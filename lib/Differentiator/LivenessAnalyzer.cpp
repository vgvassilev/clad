#include "LivenessAnalyzer.h"

#include "Analyses.h"
#include "clad/Differentiator/DiffMode.h"
#include "clad/Differentiator/DiffPlanner.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/OperationKinds.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "clang/Basic/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <set>

using namespace clang;

namespace clad {
namespace {

/// The variable \p E names: `s`, `s.a` or `a[i]` for an array `a`. Null for
/// anything else, such as `*p`, `p->a` or `get()`.
const VarDecl* rootVar(const Expr* E) {
  while (true) {
    E = E->IgnoreParenImpCasts();
    if (const auto* ME = dyn_cast<MemberExpr>(E)) {
      if (ME->isArrow())
        return nullptr;
      E = ME->getBase();
    } else if (const auto* ASE = dyn_cast<ArraySubscriptExpr>(E)) {
      E = ASE->getBase()->IgnoreParenImpCasts();
      if (!E->getType()->isArrayType())
        return nullptr;
    } else
      break;
  }
  if (const auto* DRE = dyn_cast<DeclRefExpr>(E))
    return dyn_cast<VarDecl>(DRE->getDecl());
  return nullptr;
}

/// Whether a function writing into an object of type \p T changes nothing
/// outside it: \p T reaches no other memory.
bool reachesNoOtherMemory(const ASTContext& C, QualType T) {
  T = C.getBaseElementType(T);
  if (T->isArithmeticType() || T->isEnumeralType())
    return true;
  const auto* RD = T->getAsCXXRecordDecl();
  return RD && RD->hasDefinition() && !RD->getNumBases() && !RD->isUnion() &&
         !RD->isPolymorphic() &&
         llvm::all_of(RD->fields(), [&C](const FieldDecl* F) {
           return reachesNoOtherMemory(C, F->getType());
         });
}

bool isGlobal(const VarDecl* VD) {
  return VD && !VD->hasLocalStorage() && !VD->getType().isConstQualified();
}

/// The parameters of \p FD a call to it may write through, with the
/// arguments \p CE passes them, the object of a method first.
void writtenArgs(const CallExpr* CE, const FunctionDecl* FD,
                 llvm::SmallVectorImpl<const Expr*>& ByRef,
                 llvm::SmallVectorImpl<const Expr*>& ByPointer) {
  std::size_t skip = 0;
  if (const auto* MD = dyn_cast<CXXMethodDecl>(FD))
    if (MD->isInstance()) {
      const Expr* Obj = nullptr;
      if (const auto* MCE = dyn_cast<CXXMemberCallExpr>(CE))
        Obj = MCE->getImplicitObjectArgument();
      else if (isa<CXXOperatorCallExpr>(CE)) {
        Obj = CE->getArg(0);
        skip = 1;
      }
      if (Obj && !MD->isConst()) {
        if (Obj->getType()->isPointerType())
          ByPointer.push_back(Obj);
        else
          ByRef.push_back(Obj);
      }
    }
  // Arguments past the last parameter go to a C variadic, by value.
  std::size_t e =
      std::min<std::size_t>(CE->getNumArgs(), skip + FD->getNumParams());
  for (std::size_t i = skip; i != e; ++i) {
    QualType T = FD->getParamDecl(i - skip)->getType();
    const Expr* Arg = CE->getArg(i);
    // A temporary is gone by the next statement: nothing reads what the
    // callee writes into it.
    if (isa<MaterializeTemporaryExpr>(Arg->IgnoreParens()))
      continue;
    if (T->isReferenceType() && !T.getNonReferenceType().isConstQualified())
      ByRef.push_back(Arg);
    else if (T->isPointerType() && !T->getPointeeType().isConstQualified() &&
             !Arg->IgnoreParenImpCasts()->isNullPointerConstant(
                 FD->getASTContext(), Expr::NPC_ValueDependentIsNotNull))
      ByPointer.push_back(Arg);
  }
}

/// Collects what evaluating a call writes, its arguments included: the
/// objects and arguments of calls bound to mutable references, the arguments
/// bound to mutable pointers, and the targets of assignments and increments.
class WriteCollector : public RecursiveASTVisitor<WriteCollector> {
public:
  llvm::SmallVector<const Expr*, 4> ByRef;
  llvm::SmallVector<const Expr*, 4> ByPointer;
  bool VisitCallExpr(CallExpr* CE) {
    if (const FunctionDecl* FD = CE->getDirectCallee())
      writtenArgs(CE, FD, ByRef, ByPointer);
    return true;
  }
  bool VisitBinaryOperator(BinaryOperator* BO) {
    if (BO->isAssignmentOp())
      ByRef.push_back(BO->getLHS());
    return true;
  }
  bool VisitUnaryOperator(UnaryOperator* UO) {
    if (UO->isIncrementDecrementOp())
      ByRef.push_back(UO->getSubExpr());
    return true;
  }
};

using AddressMap =
    llvm::DenseMap<const VarDecl*, llvm::SmallVector<const Expr*, 1>>;

/// Collects the variables whose address the body takes, and so may be read
/// through something other than their name. The `&x` expressions go to
/// \p TakenAt rather than \p Escaped: a call that is the only place taking
/// the address of x may still be left out.
class EscapeFinder : public RecursiveASTVisitor<EscapeFinder> {
  llvm::SmallPtrSetImpl<const VarDecl*>& m_Escaped;
  AddressMap& m_TakenAt;

public:
  EscapeFinder(llvm::SmallPtrSetImpl<const VarDecl*>& Escaped,
               AddressMap& TakenAt)
      : m_Escaped(Escaped), m_TakenAt(TakenAt) {}
  bool VisitUnaryOperator(UnaryOperator* UO) {
    if (UO->getOpcode() == UO_AddrOf)
      if (const VarDecl* VD = rootVar(UO->getSubExpr()))
        m_TakenAt.try_emplace(VD).first->second.push_back(UO);
    return true;
  }
  bool VisitImplicitCastExpr(ImplicitCastExpr* ICE) {
    if (ICE->getCastKind() == CK_ArrayToPointerDecay)
      if (const VarDecl* VD = rootVar(ICE->getSubExpr()))
        m_Escaped.insert(VD);
    return true;
  }
  bool VisitVarDecl(VarDecl* VD) {
    if (VD->getType()->isReferenceType() && VD->getInit())
      if (const VarDecl* Bound = rootVar(VD->getInit()))
        m_Escaped.insert(Bound);
    return true;
  }
  bool VisitLambdaExpr(LambdaExpr* LE) {
    for (const LambdaCapture& C : LE->captures())
      if (C.capturesVariable() && C.getCaptureKind() == LCK_ByRef)
        if (const auto* VD = dyn_cast<VarDecl>(C.getCapturedVar()))
          m_Escaped.insert(VD);
    return true;
  }
  // `a[i]` decays `a` only to index it.
  bool TraverseArraySubscriptExpr(ArraySubscriptExpr* ASE) {
    const Expr* Base = ASE->getBase()->IgnoreParens();
    if (const auto* ICE = dyn_cast<ImplicitCastExpr>(Base))
      if (ICE->getCastKind() == CK_ArrayToPointerDecay)
        return TraverseStmt(const_cast<Expr*>(ICE->getSubExpr())) &&
               TraverseStmt(ASE->getIdx());
    return RecursiveASTVisitor::TraverseArraySubscriptExpr(ASE);
  }
};

/// Whether \p S names one of \p Vars.
bool names(const Stmt* S, const llvm::SmallPtrSetImpl<const VarDecl*>& Vars) {
  class Finder : public RecursiveASTVisitor<Finder> {
    const llvm::SmallPtrSetImpl<const VarDecl*>& m_Vars;

  public:
    bool Found = false;
    explicit Finder(const llvm::SmallPtrSetImpl<const VarDecl*>& Vars)
        : m_Vars(Vars) {}
    bool shouldVisitImplicitCode() const { return true; }
    bool VisitDeclRefExpr(DeclRefExpr* DRE) {
      if (const auto* VD = dyn_cast<VarDecl>(DRE->getDecl()))
        Found |= m_Vars.count(VD) != 0;
      return !Found;
    }
  } F(Vars);
  F.TraverseStmt(const_cast<Stmt*>(S));
  return F.Found;
}

/// Why the functions the call \p CE reaches, its arguments included, may
/// change something besides the objects passed to them: a global written, or
/// a call through a pointer. None if nothing does.
AnalysisMiss whatCalleesReach(const CallExpr* CE) {
  class Scan : public RecursiveASTVisitor<Scan> {
  public:
    AnalysisMiss Miss = AnalysisMiss::None;
    llvm::SmallVector<const FunctionDecl*, 8> Next;
    bool shouldVisitImplicitCode() const { return true; }
    bool miss(AnalysisMiss M) {
      Miss = M;
      return false;
    }
    bool writesGlobal(const Expr* E) {
      E = E->IgnoreParenImpCasts();
      // `*g = ...`, `g->a = ...` and `g[i] = ...` write where a global
      // points.
      while (true) {
        const auto* UO = dyn_cast<UnaryOperator>(E);
        if (UO && UO->getOpcode() == UO_Deref)
          E = UO->getSubExpr()->IgnoreParenImpCasts();
        else if (const auto* ME = dyn_cast<MemberExpr>(E))
          E = ME->getBase()->IgnoreParenImpCasts();
        else if (const auto* ASE = dyn_cast<ArraySubscriptExpr>(E))
          E = ASE->getBase()->IgnoreParenImpCasts();
        else
          break;
      }
      const auto* DRE = dyn_cast<DeclRefExpr>(E);
      return DRE && isGlobal(dyn_cast<VarDecl>(DRE->getDecl()));
    }
    bool VisitBinaryOperator(BinaryOperator* BO) {
      if (BO->isAssignmentOp() && writesGlobal(BO->getLHS()))
        return miss(AnalysisMiss::CalleeWritesGlobal);
      return true;
    }
    bool VisitUnaryOperator(UnaryOperator* UO) {
      if ((UO->isIncrementDecrementOp() || UO->getOpcode() == UO_AddrOf) &&
          writesGlobal(UO->getSubExpr()))
        return miss(AnalysisMiss::CalleeWritesGlobal);
      return true;
    }
    bool VisitCallExpr(CallExpr* CE) {
      const FunctionDecl* FD = CE->getDirectCallee();
      const auto* MD = dyn_cast_or_null<CXXMethodDecl>(FD);
      if (!FD || (MD && MD->isVirtual() && !MD->hasAttr<FinalAttr>()))
        return miss(AnalysisMiss::CalleeCallsIndirectly);
      llvm::SmallVector<const Expr*, 4> ByRef;
      llvm::SmallVector<const Expr*, 4> ByPointer;
      writtenArgs(CE, FD, ByRef, ByPointer);
      for (const Expr* Arg : ByRef)
        if (writesGlobal(Arg))
          return miss(AnalysisMiss::CalleeWritesGlobal);
      Next.push_back(FD);
      return true;
    }
    bool VisitCXXConstructExpr(CXXConstructExpr* CE) {
      Next.push_back(CE->getConstructor());
      return true;
    }
  } S;
  llvm::SmallPtrSet<const FunctionDecl*, 16> Seen;
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  S.TraverseStmt(const_cast<CallExpr*>(CE));
  while (S.Miss == AnalysisMiss::None && !S.Next.empty()) {
    const FunctionDecl* FD = S.Next.pop_back_val();
    if (!Seen.insert(FD).second)
      continue;
    // Without a body, assume what to-be-recorded analysis assumes: the
    // function writes only through its arguments.
    if (const FunctionDecl* Def = FD->getDefinition())
      // A constructor's member initializers are not part of the body.
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
      S.TraverseDecl(const_cast<FunctionDecl*>(Def));
  }
  return S.Miss;
}

} // namespace

/// Whether \p S is `return x;` for an arithmetic x of \p Vars. The analysis
/// runs for gradients and pullbacks, whose forward sweep does not return the
/// value, and the adjoint of the return adds to x's adjoint without reading x.
static bool returnsOnly(const Stmt* S,
                        const llvm::SmallPtrSetImpl<const VarDecl*>& Vars) {
  const auto* RS = dyn_cast<ReturnStmt>(S);
  const Expr* E = RS ? RS->getRetValue() : nullptr;
  const auto* DRE =
      E ? dyn_cast<DeclRefExpr>(E->IgnoreParenImpCasts()) : nullptr;
  const auto* VD = DRE ? dyn_cast<VarDecl>(DRE->getDecl()) : nullptr;
  return VD && Vars.contains(VD) && VD->getType()->isArithmeticType();
}

/// The pointer parameter \p E indexes, as in `p[i]`.
static const ParmVarDecl* indexedParam(const Expr* E) {
  const auto* ASE = dyn_cast<ArraySubscriptExpr>(E->IgnoreParenImpCasts());
  if (!ASE)
    return nullptr;
  const auto* DRE =
      dyn_cast<DeclRefExpr>(ASE->getBase()->IgnoreParenImpCasts());
  return DRE ? dyn_cast<ParmVarDecl>(DRE->getDecl()) : nullptr;
}

/// Whether \p S names \p P.
static bool namesParam(const Stmt* S, const ParmVarDecl* P) {
  if (const auto* DRE = dyn_cast<DeclRefExpr>(S))
    return DRE->getDecl() == P;
  return llvm::any_of(S->children(),
                      [P](const Stmt* C) { return C && namesParam(C, P); });
}

/// Whether \p E is a sum of terms, each a bare `P[i]` or an expression that
/// does not name \p P, without side effects. A call counts as a side effect.
static bool isSumOfTerms(const Expr* E, const ParmVarDecl* P,
                         const ASTContext& C) {
  E = E->IgnoreParenImpCasts();
  if (const auto* BO = dyn_cast<BinaryOperator>(E))
    if (BO->getOpcode() == BO_Add || BO->getOpcode() == BO_Sub)
      return isSumOfTerms(BO->getLHS(), P, C) &&
             isSumOfTerms(BO->getRHS(), P, C);
  if (E->HasSideEffects(C))
    return false;
  if (indexedParam(E) == P)
    return true;
  return !namesParam(E, P);
}

/// Collects into \p Stores the statements `P[i] = e;` of \p S with e a sum
/// as above. Whether \p P appears nowhere else in \p S.
static bool onlyInStores(const Stmt* S, const ParmVarDecl* P,
                         const ASTContext& C,
                         llvm::SmallVectorImpl<const Stmt*>& Stores) {
  if (!S)
    return true;
  if (const auto* BO = dyn_cast<BinaryOperator>(S))
    if (BO->getOpcode() == BO_Assign && indexedParam(BO->getLHS()) == P &&
        !BO->getLHS()->HasSideEffects(C) && isSumOfTerms(BO->getRHS(), P, C)) {
      Stores.push_back(BO);
      return true;
    }
  if (const auto* DRE = dyn_cast<DeclRefExpr>(S))
    return DRE->getDecl() != P;
  return llvm::all_of(S->children(), [&](const Stmt* Child) {
    return onlyInStores(Child, P, C, Stores);
  });
}

/// Adds to \p Dead the stores `p[i] = e;` of a pullback into a restrict
/// pointer parameter p whose values nothing reads: p appears only in such
/// stores and as terms of their sums. The pullback's reverse sweep does not
/// read p, and its caller restores what the call wrote where it reads it.
static void findDeadOutputStores(const DiffRequest& R,
                                 std::set<const Stmt*>& Dead) {
  const FunctionDecl* Fn = R.Function->getDefinition();
  const ASTContext& C = Fn->getASTContext();
  for (const ParmVarDecl* P : Fn->parameters()) {
    QualType T = P->getType();
    if (!T->isPointerType() || !T.isRestrictQualified() ||
        T->getPointeeType().isConstQualified() ||
        !T->getPointeeType()->isArithmeticType())
      continue;
    llvm::SmallVector<const Stmt*, 4> Stores;
    if (onlyInStores(Fn->getBody(), P, C, Stores))
      Dead.insert(Stores.begin(), Stores.end());
  }
}

void findUnreadCalls(const DiffRequest& R, std::set<const Stmt*>& Unread) {
  if (R.Mode == DiffMode::pullback)
    findDeadOutputStores(R, Unread);
  const FunctionDecl* Fn = R.Function->getDefinition();
  // A function body is a compound statement ([dcl.fct.def.general]).
  const auto* Body = cast<CompoundStmt>(Fn->getBody());
  const ASTContext& C = Fn->getASTContext();

  llvm::SmallPtrSet<const VarDecl*, 8> Escaped;
  AddressMap TakenAt;
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  EscapeFinder(Escaped, TakenAt).TraverseStmt(const_cast<CompoundStmt*>(Body));

  auto Stmts = Body->body();
  for (auto it = Stmts.begin(), e = Stmts.end(); it != e; ++it) {
    const auto* SE = dyn_cast<Expr>(*it);
    const auto* CE = SE ? dyn_cast<CallExpr>(SE->IgnoreImplicit()) : nullptr;
    // Only a call whose pullback needs nothing from its forward sweep can be
    // left out of it.
    const FunctionDecl* Callee = CE ? CE->getDirectCallee() : nullptr;
    if (!Callee || !R.ReplayableCalls.count(CE))
      continue;

    WriteCollector W;
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    W.TraverseStmt(const_cast<CallExpr*>(CE));
    // `&x` passed to a mutable pointer parameter writes x, as passing x by
    // reference does.
    llvm::SmallVector<const Expr*, 4> ByRef(W.ByRef.begin(), W.ByRef.end());
    llvm::SmallVector<const Expr*, 4> ByPointer;
    llvm::SmallPtrSet<const Expr*, 2> OwnAddress;
    for (const Expr* Arg : W.ByPointer) {
      const auto* UO = dyn_cast<UnaryOperator>(Arg->IgnoreParenImpCasts());
      if (UO && UO->getOpcode() == UO_AddrOf && rootVar(UO->getSubExpr())) {
        ByRef.push_back(UO->getSubExpr());
        OwnAddress.insert(UO);
      } else {
        ByPointer.push_back(Arg);
      }
    }
    llvm::SmallPtrSet<const VarDecl*, 4> Written;
    bool known = true;
    for (const Expr* Arg : ByRef) {
      const VarDecl* VD = rootVar(Arg);
      // What a gradient's caller passed by reference, the caller reads.
      // A static local outlives the call: the next one reads it. A local
      // reference writes another object, read later by its own name.
      known &= VD && VD->hasLocalStorage() &&
               !(VD->isLocalVarDecl() && VD->getType()->isReferenceType()) &&
               (VD->isLocalVarDecl() ||
                (isa<ParmVarDecl>(VD) && (R.Mode == DiffMode::pullback ||
                                          !VD->getType()->isReferenceType())));
      Written.insert(VD);
    }
    // A call that changes none of the function's variables is there for
    // something else, such as printing.
    if (!known || Written.empty() ||
        std::any_of(std::next(it), e, [&](const Stmt* Later) {
          return !returnsOnly(Later, Written) && names(Later, Written);
        }))
      continue;

    // Nothing reads what the call writes by name. Say why it still runs.
    AnalysisMiss Why = AnalysisMiss::None;
    if (!Callee->getDefinition())
      Why = AnalysisMiss::UnreadCalleeHasNoBody;
    for (auto I = Written.begin(), E = Written.end();
         I != E && Why == AnalysisMiss::None; ++I) {
      const VarDecl* VD = *I;
      // Its address taken anywhere but in this call's arguments.
      if (Escaped.contains(VD) ||
          llvm::any_of(TakenAt.lookup(VD),
                       [&](const Expr* E) { return !OwnAddress.contains(E); }))
        Why = AnalysisMiss::WrittenObjectEscapes;
      else if (!reachesNoOtherMemory(C, VD->getType().getNonReferenceType()))
        Why = AnalysisMiss::WrittenObjectHoldsPointer;
    }
    if (Why == AnalysisMiss::None && !ByPointer.empty())
      Why = AnalysisMiss::CalleeWritesThroughPointer;
    if (Why == AnalysisMiss::None)
      Why = whatCalleesReach(CE);
    if (Why != AnalysisMiss::None) {
      R.recordMiss(Why, CE->getBeginLoc());
      continue;
    }
    Unread.insert(*it);
  }
}

} // namespace clad
