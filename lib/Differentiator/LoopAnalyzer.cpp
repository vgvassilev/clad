#include "LoopAnalyzer.h"

#include "clad/Differentiator/CladUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/OperationKinds.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

using namespace clang;

namespace clad {

/// Whether \p E steps \p VD by exactly one. The increment may carry
/// unrelated work alongside, as `for (...; ...; ++i, ++p)` does.
static bool stepsByOne(const Expr* E, const VarDecl* VD) {
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

/// Whether a loop's body changes what its header promised: assigning to the
/// induction variable or the bound, stepping either itself, or taking an
/// address through which something else could.
///
/// recogniseCountedForLoop reads the header alone, deliberately, and leaves
/// this to whoever acts on the result. An extent has to add it twice over: an
/// index the body moves need not stay under the bound, and a bound the body
/// raises is not the value a caller substituting its argument would get.
class LoopShapeBreaker : public RecursiveASTVisitor<LoopShapeBreaker> {
  const VarDecl* m_IndVar;
  const VarDecl* m_Bound;
  bool m_Broken = false;

  bool isWatched(const Expr* E) const {
    const auto* DRE = dyn_cast<DeclRefExpr>(E->IgnoreParenImpCasts());
    const auto* VD = DRE ? dyn_cast<VarDecl>(DRE->getDecl()) : nullptr;
    return VD && (VD == m_IndVar || VD == m_Bound);
  }

public:
  LoopShapeBreaker(const VarDecl* IndVar, const VarDecl* Bound)
      : m_IndVar(IndVar), m_Bound(Bound) {}
  [[nodiscard]] bool broken() const { return m_Broken; }

  bool VisitBinaryOperator(BinaryOperator* BO) {
    if (BO->isAssignmentOp() && isWatched(BO->getLHS()))
      m_Broken = true;
    return true;
  }
  bool VisitUnaryOperator(UnaryOperator* UO) {
    if ((UO->isIncrementDecrementOp() || UO->getOpcode() == UO_AddrOf) &&
        isWatched(UO->getSubExpr()))
      m_Broken = true;
    return true;
  }
};

class ExtentVisitor : public RecursiveASTVisitor<ExtentVisitor> {
  llvm::SmallVectorImpl<WrittenExtent>& m_Extents;
  llvm::DenseMap<const ParmVarDecl*, unsigned> m_ParamIdx;
  CountedLoopStack m_Loops;
  bool m_Opaque = false;
  /// The first write this analysis could not attribute; the later ones say
  /// nothing more, since every parameter is already reported unbounded.
  clang::SourceLocation m_OpaqueAt;
  const Stmt* m_FnBody;
  const ASTContext& m_Context;

public:
  ExtentVisitor(const FunctionDecl* FD,
                llvm::SmallVectorImpl<WrittenExtent>& Extents)
      : m_Extents(Extents), m_FnBody(FD->getBody()),
        m_Context(FD->getASTContext()) {
    for (unsigned i = 0, e = FD->getNumParams(); i != e; ++i)
      m_ParamIdx[FD->getParamDecl(i)] = i;
  }

  bool TraverseForStmt(ForStmt* FS) {
    bool Entered = !breaksShape(FS) && m_Loops.enter(FS);
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
  /// non-const reference -- its object included -- and its body is not
  /// examined here. Such an argument therefore defeats the analysis, unless it
  /// demonstrably designates this function's own local storage, which no
  /// parameter can alias.
  bool VisitCallExpr(CallExpr* CE) {
    // A member call does not carry its object among its arguments, but a
    // non-const method writes through it just the same.
    if (const auto* MCE = dyn_cast<CXXMemberCallExpr>(CE)) {
      const CXXMethodDecl* MD = MCE->getMethodDecl();
      const Expr* Obj = MCE->getImplicitObjectArgument();
      if (Obj && (!MD || !MD->isConst()) &&
          !utils::designatesLocallyOwnedStorage(
              Obj, /*asPointerValue=*/Obj->getType()->isPointerType())) {
        m_Opaque = true;
        if (m_OpaqueAt.isInvalid())
          m_OpaqueAt = Obj->getBeginLoc();
        return true;
      }
    }
    const FunctionDecl* Callee = CE->getDirectCallee();
    // An overloaded operator passes its object as argument zero, so the
    // arguments sit one ahead of the parameters when it is a member.
    unsigned Offset =
        isa<CXXOperatorCallExpr>(CE) && isa_and_nonnull<CXXMethodDecl>(Callee);
    for (unsigned i = Offset, e = CE->getNumArgs(); i != e; ++i) {
      const Expr* Arg = CE->getArg(i);
      QualType ArgTy = Arg->getType();
      // The parameter says whether the callee may write, not the argument: an
      // argument bound to a `double&` is still spelled `double` here. Where
      // there is no parameter to consult -- an indirect call, or the variadic
      // tail -- every argument counts as written.
      unsigned P = i - Offset;
      if (Callee && P < Callee->getNumParams())
        ArgTy = Callee->getParamDecl(P)->getType();
      bool MayWrite = !Callee || P >= Callee->getNumParams() ||
                      (ArgTy->isPointerType() &&
                       !ArgTy->getPointeeType().isConstQualified()) ||
                      (ArgTy->isLValueReferenceType() &&
                       !ArgTy.getNonReferenceType().isConstQualified());
      if (MayWrite && !utils::designatesLocallyOwnedStorage(
                          Arg, /*asPointerValue=*/ArgTy->isPointerType())) {
        m_Opaque = true;
        if (m_OpaqueAt.isInvalid())
          m_OpaqueAt = Arg->getBeginLoc();
        return true;
      }
    }
    return true;
  }

  /// Whether acting on \p FS's recognised shape would be unsound here.
  ///
  /// The index and the bound are watched over different reaches. The index
  /// only has to hold still while the loop runs. The bound has to hold still
  /// for the whole call, because a call site works the range out from the
  /// argument it passed and reads it much later.
  [[nodiscard]] bool breaksShape(const ForStmt* FS) const {
    CountedForLoop L = recogniseCountedForLoop(FS);
    if (!L)
      return false; // Not recognised anyway; nothing to break.
    LoopShapeBreaker InLoop(L.IndVar, /*Bound=*/nullptr);
    InLoop.TraverseStmt(const_cast<Stmt*>(cast<Stmt>(FS->getBody())));
    if (InLoop.broken())
      return true;
    const auto* BoundDRE = dyn_cast<DeclRefExpr>(L.Bound);
    const auto* BoundVD =
        BoundDRE ? dyn_cast<VarDecl>(BoundDRE->getDecl()) : nullptr;
    if (!BoundVD)
      return false;
    LoopShapeBreaker InFn(/*IndVar=*/nullptr, BoundVD);
    InFn.TraverseStmt(const_cast<Stmt*>(m_FnBody));
    return InFn.broken();
  }

  [[nodiscard]] bool sawOpaqueWrite() const { return m_Opaque; }
  [[nodiscard]] clang::SourceLocation opaqueWriteLoc() const {
    return m_OpaqueAt;
  }

private:
  /// Records `[0, Bound)` in \p E when a call site can work Bound out for
  /// itself -- when it folds to a constant, or names a by-value parameter it
  /// passed. Marks \p E BoundNotReadable otherwise, leaving it Unknown.
  void classifyBound(const Expr* B, WrittenExtent& E) const {
    E.Why = WrittenExtent::Refusal::BoundNotUsable;
    B = B->IgnoreParenImpCasts();
    // Folded, not matched against a literal: a dimension is usually written
    // as a constexpr variable, an enumerator or a template argument, and all
    // of those are as readable at a call site as the number itself.
    Expr::EvalResult R;
    if (B->EvaluateAsInt(R, m_Context)) {
      const llvm::APSInt& V = R.Val.getInt();
      // Zero-extending a negative bound would name almost all of memory, and
      // the loop it describes never runs anyway.
      if (V.isNegative())
        return;
      E.K = WrittenExtent::Kind::Range;
      E.Why = WrittenExtent::Refusal::None;
      E.BoundIsParam = false;
      E.BoundConst = V.getZExtValue();
      return;
    }
    if (const auto* DRE = dyn_cast<DeclRefExpr>(B))
      if (const auto* PVD = dyn_cast<ParmVarDecl>(DRE->getDecl())) {
        auto it = m_ParamIdx.find(PVD);
        // A reference parameter names storage a callee can change under us.
        if (it == m_ParamIdx.end() || PVD->getType()->isReferenceType())
          return;
        E.K = WrittenExtent::Kind::Range;
        E.Why = WrittenExtent::Refusal::None;
        E.BoundIsParam = true;
        E.BoundParamIdx = it->second;
      }
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
      // Keep the refusal that was already recorded: the first thing the
      // analysis could not bound is the one worth reporting.
      if (Cur.K != WrittenExtent::Kind::Unknown) {
        Cur.Why = New.Why;
        Cur.RefusedAt = New.RefusedAt;
      }
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
    if (!same) {
      Cur.K = WrittenExtent::Kind::Unknown;
      Cur.Why = WrittenExtent::Refusal::WritesDisagree;
      Cur.RefusedAt = New.RefusedAt;
    }
  }

  /// Attributes a write to a parameter and classifies the range it covers.
  void recordWrite(const Expr* LHS) {
    LHS = LHS->IgnoreParenImpCasts();
    const Expr* Base = nullptr;
    WrittenExtent E;
    // Every write carries where it is, not just the ones that give up: a
    // disagreement is discovered at the second write and has to point there.
    E.RefusedAt = LHS->getBeginLoc();

    if (const auto* ASE = dyn_cast<ArraySubscriptExpr>(LHS)) {
      Base = ASE->getBase();
      const Expr* Idx = ASE->getIdx()->IgnoreParenImpCasts();
      if (const auto* IL = dyn_cast<IntegerLiteral>(Idx)) {
        E.K = WrittenExtent::Kind::Element;
        E.Offset = IL->getValue().getZExtValue();
      } else if (const auto* DRE = dyn_cast<DeclRefExpr>(Idx)) {
        const auto* VD = dyn_cast<VarDecl>(DRE->getDecl());
        E.K = WrittenExtent::Kind::Unknown;
        E.Why = WrittenExtent::Refusal::IndexNotCounted;
        E.RefusedAt = Idx->getBeginLoc();
        const CountedForLoop* L = m_Loops.steppedBy(VD);
        // A subscript by the loop variable falls inside [0, Bound) only if
        // the loop starts at or above zero and a call site can read Bound.
        // Not Inclusive: `i <= d` reaches out[d], which [0, d) excludes.
        if (L && L->InitIsNonNegative && !L->Inclusive)
          classifyBound(L->Bound, E);
      } else {
        E.K = WrittenExtent::Kind::Unknown;
        E.Why = WrittenExtent::Refusal::IndexNotUnderstood;
        E.RefusedAt = Idx->getBeginLoc();
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
      if (Base->getType()->isPointerType()) {
        m_Opaque = true;
        if (m_OpaqueAt.isInvalid())
          m_OpaqueAt = LHS->getBeginLoc();
      }
      return;
    }
    widen(it->second, E);
  }
};

} // namespace

/// Whether a caller could see a write through this parameter at all: a
/// pointer or reference to something not const.
static bool parameterMayBeWritten(QualType T) {
  return (T->isPointerType() && !T->getPointeeType().isConstQualified()) ||
         (T->isLValueReferenceType() &&
          !T.getNonReferenceType().isConstQualified());
}

void computeWrittenExtents(const FunctionDecl* FD,
                           llvm::SmallVectorImpl<WrittenExtent>& Extents) {
  Extents.clear();
  Extents.resize(FD->getNumParams());
  // No body to inspect. None would read as "writes nothing", which is the one
  // thing this analysis must never say without having looked -- an extern may
  // write all of a buffer it is handed.
  if (!FD->doesThisDeclarationHaveABody()) {
    for (unsigned i = 0, e = FD->getNumParams(); i != e; ++i)
      if (parameterMayBeWritten(FD->getParamDecl(i)->getType())) {
        Extents[i].K = WrittenExtent::Kind::Unknown;
        Extents[i].Why = WrittenExtent::Refusal::NoDefinition;
        Extents[i].RefusedAt = FD->getLocation();
      }
    return;
  }
  ExtentVisitor V(FD, Extents);
  V.TraverseStmt(FD->getBody());
  // Something in the body could write through a parameter without this
  // analysis seeing which one. Report every parameter it could have been as
  // unbounded rather than as untouched, so a caller that gates on isProven()
  // does not mistake silence for proof.
  if (V.sawOpaqueWrite())
    for (unsigned i = 0, e = FD->getNumParams(); i != e; ++i) {
      if (parameterMayBeWritten(FD->getParamDecl(i)->getType())) {
        Extents[i].K = WrittenExtent::Kind::Unknown;
        Extents[i].Why = WrittenExtent::Refusal::OpaqueWrite;
        Extents[i].RefusedAt = V.opaqueWriteLoc();
      }
    }
}

} // namespace clad
