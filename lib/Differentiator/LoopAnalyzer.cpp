#include "LoopAnalyzer.h"

#include "clad/Differentiator/CladUtils.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

using namespace clang;

namespace clad {
namespace {

/// Whether \p E steps \p VD by exactly one. The increment may carry
/// unrelated work alongside, as `for (...; ...; ++i, ++p)` does.
bool stepsByOne(const Expr* E, const VarDecl* VD) {
  if (!E)
    return false;
  E = E->IgnoreParenImpCasts();
  if (const auto* UO = dyn_cast<UnaryOperator>(E)) {
    if (UO->getOpcode() != UO_PostInc && UO->getOpcode() != UO_PreInc)
      return false;
    const auto* DRE =
        dyn_cast<DeclRefExpr>(UO->getSubExpr()->IgnoreParenImpCasts());
    return DRE && DRE->getDecl() == VD;
  }
  if (const auto* BO = dyn_cast<BinaryOperator>(E))
    if (BO->getOpcode() == BO_Comma)
      return stepsByOne(BO->getLHS(), VD) || stepsByOne(BO->getRHS(), VD);
  return false;
}

} // namespace

CountedForLoop recogniseCountedForLoop(const ForStmt* FS) {
  CountedForLoop L;
  if (!FS)
    return L;

  // `v < bound` or `v <= bound`, naming the variable on the left.
  const auto* Cond = dyn_cast_or_null<BinaryOperator>(FS->getCond());
  if (!Cond)
    return L;
  bool Inclusive = Cond->getOpcode() == BO_LE;
  if (!Inclusive && Cond->getOpcode() != BO_LT)
    return L;
  const auto* CondLHS =
      dyn_cast<DeclRefExpr>(Cond->getLHS()->IgnoreParenImpCasts());
  if (!CondLHS)
    return L;
  const auto* IndVar = dyn_cast<VarDecl>(CondLHS->getDecl());
  // Integer only: a floating induction variable makes the iteration count
  // depend on rounding.
  if (!IndVar || !IndVar->getType()->isIntegerType())
    return L;

  // `T v = init` or `v = init`, naming that same variable.
  const Expr* Init = nullptr;
  if (const auto* DS = dyn_cast_or_null<DeclStmt>(FS->getInit())) {
    if (DS->isSingleDecl() && DS->getSingleDecl() == IndVar)
      Init = IndVar->getInit();
  } else if (const auto* BO = dyn_cast_or_null<BinaryOperator>(FS->getInit())) {
    const auto* LHS =
        dyn_cast<DeclRefExpr>(BO->getLHS()->IgnoreParenImpCasts());
    if (BO->getOpcode() == BO_Assign && LHS && LHS->getDecl() == IndVar)
      Init = BO->getRHS();
  }
  if (!Init)
    return L;

  if (!stepsByOne(FS->getInc(), IndVar))
    return L;

  L.IndVar = IndVar;
  L.Init = Init->IgnoreParenImpCasts();
  L.Bound = Cond->getRHS()->IgnoreParenImpCasts();
  L.Inclusive = Inclusive;
  return L;
}
bool CountedLoopStack::isNonNegative(const Expr* E) const {
  if (!E)
    return false;
  E = E->IgnoreParenImpCasts();
  if (const auto* IL = dyn_cast<IntegerLiteral>(E))
    return !IL->getValue().isNegative();
  // A loop index already on the stack is non-negative if its own start was.
  if (const auto* DRE = dyn_cast<DeclRefExpr>(E)) {
    const CountedForLoop* L = steppedBy(dyn_cast<VarDecl>(DRE->getDecl()));
    return L && L->InitIsNonNegative;
  }
  if (const auto* BO = dyn_cast<BinaryOperator>(E))
    if (BO->getOpcode() == BO_Add)
      return isNonNegative(BO->getLHS()) && isNonNegative(BO->getRHS());
  return false;
}

bool CountedLoopStack::enter(const ForStmt* FS) {
  CountedForLoop L = recogniseCountedForLoop(FS);
  if (!L)
    return false;
  // Computed here rather than at recognition: it depends on the loops this
  // one sits inside, which only the stack knows.
  L.InitIsNonNegative = isNonNegative(L.Init);
  m_Loops.push_back(L);
  return true;
}

namespace {

class ExtentVisitor : public RecursiveASTVisitor<ExtentVisitor> {
  llvm::SmallVector<WrittenExtent, 8>& m_Extents;
  llvm::DenseMap<const ParmVarDecl*, unsigned> m_ParamIdx;
  CountedLoopStack m_Loops;
  bool m_Opaque = false;

public:
  ExtentVisitor(const FunctionDecl* FD,
                llvm::SmallVector<WrittenExtent, 8>& Extents)
      : m_Extents(Extents) {
    for (unsigned i = 0, e = FD->getNumParams(); i != e; ++i)
      m_ParamIdx[FD->getParamDecl(i)] = i;
  }

  bool TraverseForStmt(ForStmt* FS) {
    bool Entered = m_Loops.enter(FS);
    bool res = RecursiveASTVisitor::TraverseForStmt(FS);
    m_Loops.leave(Entered);
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
  /// Records `[0, Bound)` in \p E when a call site can read Bound -- that is,
  /// when it is one of this function's parameters or an integer constant.
  /// Leaves \p E alone otherwise, so it stays Unknown.
  void classifyBound(const Expr* B, WrittenExtent& E) const {
    B = B->IgnoreParenImpCasts();
    if (const auto* IL = dyn_cast<IntegerLiteral>(B)) {
      E.K = WrittenExtent::Kind::Range;
      E.BoundIsParam = false;
      E.BoundConst = IL->getValue().getZExtValue();
      return;
    }
    if (const auto* DRE = dyn_cast<DeclRefExpr>(B))
      if (const auto* PVD = dyn_cast<ParmVarDecl>(DRE->getDecl())) {
        auto it = m_ParamIdx.find(PVD);
        if (it == m_ParamIdx.end())
          return;
        E.K = WrittenExtent::Kind::Range;
        E.BoundIsParam = true;
        E.BoundParamIdx = it->second;
      }
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
      const CountedForLoop* L = m_Loops.steppedBy(VD);
      return L && L->InitIsNonNegative;
    }
    if (const auto* BO = dyn_cast<BinaryOperator>(E))
      if (BO->getOpcode() == BO_Add)
        return isNonNegative(BO->getLHS()) && isNonNegative(BO->getRHS());
    return false;
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
    bool same =
        Cur.K == New.K && (Cur.K == WrittenExtent::Kind::Element
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
        const CountedForLoop* L = m_Loops.steppedBy(VD);
        // A subscript by the loop variable falls inside [0, Bound) only if
        // the loop starts at or above zero and a call site can read Bound.
        if (L && L->InitIsNonNegative)
          classifyBound(L->Bound, E);
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
