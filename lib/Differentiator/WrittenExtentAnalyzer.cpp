#include "WrittenExtentAnalyzer.h"

#include "clad/Differentiator/CladUtils.h"

#include "clang/AST/Expr.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

using namespace clang;

namespace clad {

namespace {

/// A counted loop `for (v = Init; v < Bound; v++)`, recognised only in that
/// shape. `Bound` is a parameter or an integer constant; `InitIsNonNegative`
/// records whether the loop starts at or above zero, which is what makes a
/// subscript by `v` fall inside [0, Bound).
struct CountedLoop {
  const VarDecl* IV = nullptr;
  bool BoundIsParam = false;
  unsigned BoundParamIdx = 0;
  std::uint64_t BoundConst = 0;
  bool InitIsNonNegative = false;
};

class ExtentVisitor : public RecursiveASTVisitor<ExtentVisitor> {
  llvm::SmallVector<WrittenExtent, 8>& m_Extents;
  llvm::DenseMap<const ParmVarDecl*, unsigned> m_ParamIdx;
  llvm::SmallVector<CountedLoop, 4> m_Loops;
  bool m_Opaque = false;

public:
  ExtentVisitor(const FunctionDecl* FD,
                llvm::SmallVector<WrittenExtent, 8>& Extents)
      : m_Extents(Extents) {
    for (unsigned i = 0, e = FD->getNumParams(); i != e; ++i)
      m_ParamIdx[FD->getParamDecl(i)] = i;
  }

  bool TraverseForStmt(ForStmt* FS) {
    CountedLoop L;
    bool recognised = recognise(FS, L);
    if (recognised)
      m_Loops.push_back(L);
    bool res = RecursiveASTVisitor::TraverseForStmt(FS);
    if (recognised)
      m_Loops.pop_back();
    return res;
  }

  bool VisitBinaryOperator(BinaryOperator* BO) {
    if (BO->isAssignmentOp())
      recordWrite(BO->getLHS());
    return true;
  }

  bool VisitUnaryOperator(UnaryOperator* UO) {
    if (UO->isIncrementDecrementOp())
      recordWrite(UO->getSubExpr());
    return true;
  }

  /// A callee can write through anything it is handed by pointer or by
  /// non-const reference, and its body is not examined here. Such an argument
  /// therefore defeats the analysis -- unless it demonstrably designates this
  /// function's own local storage, which no parameter can alias.
  bool VisitCallExpr(CallExpr* CE) {
    for (const Expr* arg : CE->arguments()) {
      QualType argTy = arg->getType();
      bool mayWrite = (argTy->isPointerType() &&
                       !argTy->getPointeeType().isConstQualified()) ||
                      (argTy->isLValueReferenceType() &&
                       !argTy.getNonReferenceType().isConstQualified());
      if (mayWrite && !utils::designatesLocallyOwnedStorage(
                          arg, /*asPointerValue=*/argTy->isPointerType())) {
        m_Opaque = true;
        return true;
      }
    }
    return true;
  }

  bool sawOpaqueWrite() const { return m_Opaque; }

private:
  /// Matches `for (v = <init>; v < <bound>; v++)` and nothing else.
  bool recognise(const ForStmt* FS, CountedLoop& L) const {
    const Stmt* Init = FS->getInit();
    const Expr* InitVal = nullptr;
    if (const auto* DS = dyn_cast_or_null<DeclStmt>(Init)) {
      if (!DS->isSingleDecl())
        return false;
      const auto* VD = dyn_cast<VarDecl>(DS->getSingleDecl());
      if (!VD || !VD->getInit())
        return false;
      L.IV = VD;
      InitVal = VD->getInit()->IgnoreParenImpCasts();
    } else if (const auto* BO = dyn_cast_or_null<BinaryOperator>(Init)) {
      if (BO->getOpcode() != BO_Assign)
        return false;
      const auto* DRE =
          dyn_cast<DeclRefExpr>(BO->getLHS()->IgnoreParenImpCasts());
      if (!DRE)
        return false;
      L.IV = dyn_cast<VarDecl>(DRE->getDecl());
      InitVal = BO->getRHS()->IgnoreParenImpCasts();
    }
    if (!L.IV || !InitVal)
      return false;

    // `v++` or `++v`, and nothing that could step by anything else.
    const auto* Inc = dyn_cast_or_null<UnaryOperator>(FS->getInc());
    if (!Inc || (Inc->getOpcode() != UO_PostInc &&
                 Inc->getOpcode() != UO_PreInc) ||
        !refersTo(Inc->getSubExpr(), L.IV))
      return false;

    // `v < bound`.
    const auto* Cond = dyn_cast_or_null<BinaryOperator>(FS->getCond());
    if (!Cond || Cond->getOpcode() != BO_LT ||
        !refersTo(Cond->getLHS(), L.IV))
      return false;
    if (!bound(Cond->getRHS(), L))
      return false;

    L.InitIsNonNegative = isNonNegative(InitVal);
    return true;
  }

  /// The bound must be readable at a call site: a parameter, or a constant.
  bool bound(const Expr* E, CountedLoop& L) const {
    E = E->IgnoreParenImpCasts();
    if (const auto* IL = dyn_cast<IntegerLiteral>(E)) {
      L.BoundIsParam = false;
      L.BoundConst = IL->getValue().getZExtValue();
      return true;
    }
    if (const auto* DRE = dyn_cast<DeclRefExpr>(E))
      if (const auto* PVD = dyn_cast<ParmVarDecl>(DRE->getDecl())) {
        auto it = m_ParamIdx.find(PVD);
        if (it == m_ParamIdx.end())
          return false;
        L.BoundIsParam = true;
        L.BoundParamIdx = it->second;
        return true;
      }
    return false;
  }

  /// A start value of zero or more. Sums of enclosing induction variables and
  /// non-negative literals qualify, which is what `for (j = i + 1; ...)`
  /// needs.
  bool isNonNegative(const Expr* E) const {
    E = E->IgnoreParenImpCasts();
    if (const auto* IL = dyn_cast<IntegerLiteral>(E))
      return !IL->getValue().isNegative();
    if (const auto* DRE = dyn_cast<DeclRefExpr>(E)) {
      const auto* VD = dyn_cast<VarDecl>(DRE->getDecl());
      for (const CountedLoop& L : m_Loops)
        if (L.IV == VD)
          return L.InitIsNonNegative;
      return false;
    }
    if (const auto* BO = dyn_cast<BinaryOperator>(E))
      if (BO->getOpcode() == BO_Add)
        return isNonNegative(BO->getLHS()) && isNonNegative(BO->getRHS());
    return false;
  }

  static bool refersTo(const Expr* E, const VarDecl* VD) {
    const auto* DRE = dyn_cast<DeclRefExpr>(E->IgnoreParenImpCasts());
    return DRE && DRE->getDecl() == VD;
  }

  /// Widens the recorded extent for `Idx` so it also covers `New`. Two
  /// descriptions that are not identical widen to Unknown rather than to a
  /// guessed union: a wrong union would under-record.
  void widen(unsigned Idx, const WrittenExtent& New) {
    WrittenExtent& Cur = m_Extents[Idx];
    if (Cur.K == WrittenExtent::Kind::None) {
      Cur = New;
      return;
    }
    if (Cur.K == WrittenExtent::Kind::Unknown ||
        New.K == WrittenExtent::Kind::Unknown) {
      Cur.K = WrittenExtent::Kind::Unknown;
      return;
    }
    // A single element inside an already-recorded range adds nothing, and a
    // range subsumes a single element only when the element is provably
    // inside it -- which needs the bound's value, so do not assume it.
    bool same = Cur.K == New.K &&
                (Cur.K == WrittenExtent::Kind::Element
                     ? Cur.Offset == New.Offset
                     : Cur.BoundIsParam == New.BoundIsParam &&
                           Cur.BoundParamIdx == New.BoundParamIdx &&
                           Cur.BoundConst == New.BoundConst);
    if (!same)
      Cur.K = WrittenExtent::Kind::Unknown;
  }

  /// Attributes a write to a parameter and classifies the range it covers.
  void recordWrite(const Expr* LHS) {
    LHS = LHS->IgnoreParenImpCasts();
    const Expr* Base = nullptr;
    WrittenExtent E;

    if (const auto* ASE = dyn_cast<ArraySubscriptExpr>(LHS)) {
      Base = ASE->getBase();
      const Expr* Idx = ASE->getIdx()->IgnoreParenImpCasts();
      if (const auto* IL = dyn_cast<IntegerLiteral>(Idx)) {
        E.K = WrittenExtent::Kind::Element;
        E.Offset = IL->getValue().getZExtValue();
      } else if (const auto* DRE = dyn_cast<DeclRefExpr>(Idx)) {
        const auto* VD = dyn_cast<VarDecl>(DRE->getDecl());
        E.K = WrittenExtent::Kind::Unknown;
        for (const CountedLoop& L : m_Loops)
          if (L.IV == VD && L.InitIsNonNegative) {
            E.K = WrittenExtent::Kind::Range;
            E.BoundIsParam = L.BoundIsParam;
            E.BoundParamIdx = L.BoundParamIdx;
            E.BoundConst = L.BoundConst;
            break;
          }
      } else {
        E.K = WrittenExtent::Kind::Unknown;
      }
    } else if (const auto* UO = dyn_cast<UnaryOperator>(LHS)) {
      if (UO->getOpcode() != UO_Deref)
        return;
      Base = UO->getSubExpr();
      E.K = WrittenExtent::Kind::Element;
      E.Offset = 0;
    } else {
      // A write to something that is not reached through a pointer -- a local
      // scalar, a member -- cannot land in a parameter's buffer.
      return;
    }

    // Reached through a pointer, but not one of this function's parameters:
    // it may alias any of them, and nothing here rules that out.
    const auto* DRE = dyn_cast<DeclRefExpr>(Base->IgnoreParenImpCasts());
    const auto* PVD = DRE ? dyn_cast<ParmVarDecl>(DRE->getDecl()) : nullptr;
    auto it = PVD ? m_ParamIdx.find(PVD) : m_ParamIdx.end();
    if (it == m_ParamIdx.end()) {
      if (Base->getType()->isPointerType())
        m_Opaque = true;
      return;
    }
    widen(it->second, E);
  }
};

} // namespace

llvm::SmallVector<WrittenExtent, 8>
computeWrittenExtents(const FunctionDecl* FD) {
  llvm::SmallVector<WrittenExtent, 8> Extents;
  Extents.resize(FD->getNumParams());
  if (!FD->doesThisDeclarationHaveABody())
    return Extents;
  ExtentVisitor V(FD, Extents);
  V.TraverseStmt(FD->getBody());
  // Something in the body could write through a parameter without this
  // analysis seeing which one. Report every parameter it could have been as
  // unbounded rather than as untouched, so a caller that gates on isProven()
  // does not mistake silence for proof.
  if (V.sawOpaqueWrite())
    for (unsigned i = 0, e = FD->getNumParams(); i != e; ++i) {
      QualType parTy = FD->getParamDecl(i)->getType();
      bool mayWrite = (parTy->isPointerType() &&
                       !parTy->getPointeeType().isConstQualified()) ||
                      (parTy->isLValueReferenceType() &&
                       !parTy.getNonReferenceType().isConstQualified());
      if (mayWrite)
        Extents[i].K = WrittenExtent::Kind::Unknown;
    }
  return Extents;
}

} // namespace clad
