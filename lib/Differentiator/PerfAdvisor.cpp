//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
//
// Where a hand-written derivative would do much less work than the generated
// one. Everything here reads code clad already finished; nothing feeds back
// into planning or generation.
//------------------------------------------------------------------------------

#include "clad/Differentiator/PerfAdvisor.h"

#include "Diagnostics.h"

#include "clad/Differentiator/CladUtils.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtCXX.h"
#include "clang/Sema/Sema.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <string>
#include <utility>

using namespace clang;

namespace clad {
namespace {

/// The share of a gradient's estimated work above which a finding is worth a
/// remark. Below it, a hand-written derivative would not repay the effort.
constexpr double ReportShare = 0.05;
/// The work, in the cost model's operations per run of the gradient, below
/// which a finding is not worth a hand-written derivative whatever its share.
constexpr double ReportWork = 1e4;
/// How many times a loop whose trip count is not a constant is taken to run.
constexpr double UnknownTrips = 10;

/// The trip counts of the loops that count a variable up, as the forward
/// sweep of a generated derivative counts the iterations its reverse sweep
/// then counts down: `for (; _t0; _t0--)` runs as often as the loop that did
/// `_t0++`.
using CounterTrips = llvm::DenseMap<const VarDecl*, double>;

/// How many times the loop \p S runs, if its bounds are constants or it
/// counts down a counter of \p Counters.
double tripsOf(const Stmt* S, const ASTContext& C,
               const CounterTrips* Counters = nullptr) {
  const auto* For = dyn_cast<ForStmt>(S);
  if (!For)
    return UnknownTrips;
  const Expr* Cond = For->getCond();
  // A loop counting a variable down to zero: `for (_t0 = 512U; _t0; _t0--)`,
  // as a counted loop's reverse sweep is generated, or from what the forward
  // sweep counted it up to.
  if (const auto* DRE = dyn_cast_or_null<DeclRefExpr>(
          Cond ? Cond->IgnoreParenImpCasts() : nullptr))
    if (const auto* VD = dyn_cast<VarDecl>(DRE->getDecl())) {
      llvm::SmallVector<const Expr*, 2> Inits;
      if (const auto* E = dyn_cast_or_null<Expr>(For->getInit()))
        Inits.push_back(E);
      while (!Inits.empty()) {
        const Expr* E = Inits.pop_back_val()->IgnoreParenImpCasts();
        const auto* BO = dyn_cast<BinaryOperator>(E);
        if (!BO)
          continue;
        if (BO->getOpcode() == BO_Comma) {
          Inits.push_back(BO->getLHS());
          Inits.push_back(BO->getRHS());
          continue;
        }
        const auto* LHS =
            dyn_cast<DeclRefExpr>(BO->getLHS()->IgnoreParenImpCasts());
        Expr::EvalResult N;
        if (BO->getOpcode() == BO_Assign && LHS && LHS->getDecl() == VD &&
            !BO->getRHS()->isValueDependent() &&
            BO->getRHS()->EvaluateAsInt(N, C))
          return N.Val.getInt().getSExtValue();
      }
      if (Counters)
        if (auto It = Counters->find(VD); It != Counters->end())
          return It->second;
    }
  const auto* Cmp = dyn_cast_or_null<BinaryOperator>(Cond);
  if (!Cmp || (Cmp->getOpcode() != BO_LT && Cmp->getOpcode() != BO_LE))
    return UnknownTrips;
  // The start: `int i = a` or, as generated code hoists it, `i = a`.
  const Expr* Start = nullptr;
  if (const auto* Init = dyn_cast_or_null<DeclStmt>(For->getInit())) {
    if (Init->isSingleDecl())
      if (const auto* VD = dyn_cast<VarDecl>(Init->getSingleDecl()))
        Start = VD->getInit();
  } else if (const auto* Assign =
                 dyn_cast_or_null<BinaryOperator>(For->getInit())) {
    if (Assign->getOpcode() == BO_Assign)
      Start = Assign->getRHS();
  }
  Expr::EvalResult From, To;
  if (!Start || Start->isValueDependent() ||
      Cmp->getRHS()->isValueDependent() || !Start->EvaluateAsInt(From, C) ||
      !Cmp->getRHS()->EvaluateAsInt(To, C))
    return UnknownTrips;
  double N = (To.Val.getInt() - From.Val.getInt()).getSExtValue() +
             (Cmp->getOpcode() == BO_LE ? 1 : 0);
  return N > 0 ? N : 0;
}

/// Fills \p Counters from the loops in \p S that count a variable up once per
/// iteration.
void collectCounters(const Stmt* S, const ASTContext& C,
                     CounterTrips& Counters) {
  if (!S)
    return;
  if (const auto* For = dyn_cast<ForStmt>(S)) {
    double Trips = tripsOf(For, C, &Counters);
    if (Trips != UnknownTrips)
      for (const Stmt* Child : For->getBody()->children())
        if (const auto* UO = dyn_cast_or_null<UnaryOperator>(Child))
          if (UO->isIncrementOp())
            if (const auto* DRE = dyn_cast<DeclRefExpr>(
                    UO->getSubExpr()->IgnoreParenImpCasts()))
              if (const auto* VD = dyn_cast<VarDecl>(DRE->getDecl()))
                Counters[VD] = Trips;
  }
  for (const Stmt* Child : S->children())
    collectCounters(Child, C, Counters);
}

/// How many scalars an object of type \p T holds, 0 if it holds anything
/// else (a pointer, a reference, a class with bases or a virtual table).
std::size_t scalarsIn(QualType T, const ASTContext& C) {
  T = T.getCanonicalType().getNonReferenceType();
  if (T->isArithmeticType() || T->isEnumeralType())
    return 1;
  if (const auto* CAT = C.getAsConstantArrayType(T))
    return CAT->getSize().getZExtValue() * scalarsIn(CAT->getElementType(), C);
  const auto* RD = T->getAsCXXRecordDecl();
  if (!RD || !RD->hasDefinition() || RD->getNumBases() || RD->isPolymorphic())
    return 0;
  std::size_t N = 0;
  for (const FieldDecl* F : RD->fields()) {
    std::size_t K = scalarsIn(F->getType(), C);
    if (!K)
      return 0;
    N += K;
  }
  return N;
}

/// A static count of the operations a function or a statement performs: one
/// per arithmetic operation, as many as an object holds scalars for a copy of
/// it, and a loop's body times its trip count. An estimate for ranking, not a
/// prediction of time.
class CostModel {
  const ASTContext& m_Context;

  static bool returns(const Stmt* S) {
    if (const auto* CS = dyn_cast_or_null<CompoundStmt>(S))
      return !CS->body_empty() && returns(CS->body_back());
    return isa_and_nonnull<ReturnStmt>(S);
  }

  llvm::DenseMap<const FunctionDecl*, double> m_Cost;
  llvm::DenseSet<const FunctionDecl*> m_Visiting;
  CounterTrips m_Counters;

public:
  explicit CostModel(const ASTContext& C) : m_Context(C) {}

  /// Learns the counters the loops of \p S count, for the loops that count
  /// them down.
  void countLoopsIn(const Stmt* S) {
    collectCounters(S, m_Context, m_Counters);
  }

  double of(const FunctionDecl* FD) {
    FD = FD->getDefinition() ? FD->getDefinition() : FD;
    if (auto It = m_Cost.find(FD); It != m_Cost.end())
      return It->second;
    if (!FD->hasBody() || !m_Visiting.insert(FD).second)
      return 1;
    collectCounters(FD->getBody(), m_Context, m_Counters);
    double C = of(FD->getBody());
    m_Visiting.erase(FD);
    return m_Cost[FD] = C;
  }

  /// \p Skip, if given, says which calls are not counted.
  double of(const Stmt* S,
            llvm::function_ref<bool(const FunctionDecl*)> Skip = nullptr) {
    if (!S)
      return 0;
    double C = 0;
    if (const auto* E = dyn_cast<Expr>(S)) {
      QualType T = E->getType();
      if (const auto* BO = dyn_cast<BinaryOperator>(E)) {
        if (!BO->isAssignmentOp() && !BO->isCommaOp() && T->isArithmeticType())
          C += 1;
        if (BO->isCompoundAssignmentOp())
          C += 1;
      } else if (const auto* UO = dyn_cast<UnaryOperator>(E)) {
        if (UO->isArithmeticOp() || UO->isIncrementDecrementOp())
          C += T->isArithmeticType() ? 1 : 0;
      } else if (const auto* CE = dyn_cast<CallExpr>(E)) {
        if (const FunctionDecl* FD = CE->getDirectCallee())
          if (!Skip || !Skip(FD))
            C += of(FD);
      } else if (const auto* Ctor = dyn_cast<CXXConstructExpr>(E)) {
        if (Ctor->getConstructor()->isCopyOrMoveConstructor())
          C += scalarsIn(T, m_Context);
        else if (!Skip || !Skip(Ctor->getConstructor()))
          C += of(Ctor->getConstructor());
      }
    }
    double Body = 0;
    if (const auto* If = dyn_cast<IfStmt>(S)) {
      // Either branch may be the one taken: count each at half.
      Body = of(If->getCond(), Skip) +
             (of(If->getThen(), Skip) + of(If->getElse(), Skip)) / 2;
    } else if (const auto* CS = dyn_cast<CompoundStmt>(S)) {
      // What follows an `if (...) return` runs only when it does not return.
      double Reach = 1;
      for (const Stmt* Child : CS->body()) {
        Body += Reach * of(Child, Skip);
        if (const auto* If = dyn_cast<IfStmt>(Child))
          if (!If->getElse() && returns(If->getThen()))
            Reach /= 2;
      }
    } else {
      for (const Stmt* Child : S->children())
        Body += of(Child, Skip);
    }
    if (isa<ForStmt, WhileStmt, DoStmt, CXXForRangeStmt>(S))
      Body *= tripsOf(S, m_Context, &m_Counters);
    return C + Body;
  }
};

/// Whether \p FD is part of clad's runtime, such as a tape operation: declared
/// in namespace clad, outside the custom derivatives.
bool isCladRuntime(const FunctionDecl* FD) {
  bool InClad = false;
  for (const DeclContext* DC = FD->getDeclContext(); DC; DC = DC->getParent())
    if (const auto* NS = dyn_cast<NamespaceDecl>(DC)) {
      if (NS->getName() == "custom_derivatives")
        return false;
      InClad |= NS->getName() == "clad" && NS->getParent()->isTranslationUnit();
    }
  return InClad;
}

/// The calls a statement makes, each with how many times one run of the
/// statement makes it.
void collectCalls(
    const Stmt* S, double Mult, const ASTContext& C,
    llvm::SmallVectorImpl<std::pair<const CallExpr*, double>>& Out,
    const CounterTrips* Counters = nullptr) {
  if (!S)
    return;
  if (const auto* CE = dyn_cast<CallExpr>(S))
    Out.emplace_back(CE, Mult);
  double Inner = Mult;
  if (isa<ForStmt, WhileStmt, DoStmt, CXXForRangeStmt>(S))
    Inner = Mult * tripsOf(S, C, Counters);
  for (const Stmt* Child : S->children())
    collectCalls(Child, Inner, C, Out, Counters);
}

/// A local object that starts out the same whatever the inputs are, and is
/// then written at a few positions named by constants only.
struct Structured {
  const VarDecl* Var = nullptr;
  std::size_t Written = 0, Scalars = 0;
};

/// \p E without the copies and moves around it.
const Expr* withoutCopies(const Expr* E) {
  while (true) {
    E = E->IgnoreImplicit()->IgnoreParens();
    const auto* CE = dyn_cast<CXXConstructExpr>(E);
    if (!CE || CE->getNumArgs() != 1 ||
        !CE->getConstructor()->isCopyOrMoveConstructor())
      return E;
    E = CE->getArg(0);
  }
}

/// Whether \p E reads nothing the function's inputs can change.
bool isInputIndependent(const Expr* E) {
  class Finder : public RecursiveASTVisitor<Finder> {
  public:
    bool Reads = false;
    bool VisitDeclRefExpr(DeclRefExpr* DRE) {
      if (const auto* VD = dyn_cast<VarDecl>(DRE->getDecl()))
        Reads |= !VD->getType().isConstQualified() || VD->hasLocalStorage();
      return !Reads;
    }
    bool VisitCXXThisExpr(CXXThisExpr*) {
      Reads = true;
      return false;
    }
  } F;
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  F.TraverseStmt(const_cast<Expr*>(E));
  return !F.Reads;
}

/// The constant position \p E names in \p V, as `V(1, 2)`, `V[3]` or `V.a`,
/// or empty if it names none.
std::string positionIn(const Expr* E, const VarDecl* V, const ASTContext& C) {
  E = E->IgnoreParenImpCasts();
  auto isV = [V](const Expr* B) {
    const auto* DRE = dyn_cast<DeclRefExpr>(B->IgnoreParenImpCasts());
    return DRE && DRE->getDecl() == V;
  };
  std::string Pos;
  auto constant = [&C, &Pos](const Expr* I) {
    Expr::EvalResult R;
    if (I->isValueDependent() || !I->EvaluateAsInt(R, C))
      return false;
    Pos += std::to_string(R.Val.getInt().getSExtValue()) + ",";
    return true;
  };
  if (const auto* OCE = dyn_cast<CXXOperatorCallExpr>(E)) {
    if ((OCE->getOperator() != OO_Call && OCE->getOperator() != OO_Subscript) ||
        !OCE->getNumArgs() || !isV(OCE->getArg(0)))
      return {};
    for (unsigned i = 1; i < OCE->getNumArgs(); ++i)
      if (!constant(OCE->getArg(i)))
        return {};
    return Pos;
  }
  if (const auto* ASE = dyn_cast<ArraySubscriptExpr>(E))
    return isV(ASE->getBase()) && constant(ASE->getIdx()) ? Pos : "";
  if (const auto* ME = dyn_cast<MemberExpr>(E))
    return isV(ME->getBase()) ? ME->getMemberDecl()->getNameAsString() : "";
  return {};
}

/// The locals of \p FD that start out input-independent and are written at
/// few constant positions afterwards. \p Returns tells which functions return
/// such an object, so that `R = f()` starts out structured too.
llvm::SmallVector<Structured, 2>
findStructured(const FunctionDecl* FD, const ASTContext& C,
               const llvm::DenseMap<const FunctionDecl*, Structured>& Returns) {
  llvm::SmallVector<Structured, 2> Found;
  if (!FD->hasBody())
    return Found;
  class Locals : public RecursiveASTVisitor<Locals> {
  public:
    llvm::SmallVector<const VarDecl*, 4> Vars;
    bool VisitVarDecl(VarDecl* VD) {
      if (VD->hasLocalStorage() && !isa<ParmVarDecl>(VD) && VD->getInit())
        Vars.push_back(VD);
      return true;
    }
  } L;
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  L.TraverseStmt(const_cast<Stmt*>(FD->getBody()));
  for (const VarDecl* V : L.Vars) {
    std::size_t N = scalarsIn(V->getType(), C);
    if (N < 9)
      continue;
    const Expr* Init = withoutCopies(V->getInit());
    std::size_t Inherited = 0;
    if (const auto* CE = dyn_cast<CallExpr>(Init))
      if (const FunctionDecl* Callee = CE->getDirectCallee())
        if (auto It = Returns.find(Callee->getCanonicalDecl());
            It != Returns.end())
          Inherited = It->second.Written;
    if (!Inherited && !isInputIndependent(Init))
      continue;
    // Every write to V names a constant position; anything else writes an
    // unknown part of it, and V is not structured.
    class Writes : public RecursiveASTVisitor<Writes> {
      const VarDecl* m_V;
      const ASTContext& m_C;

    public:
      llvm::DenseSet<llvm::StringRef> Positions;
      llvm::SmallVector<std::string, 8> Storage;
      bool Unknown = false;
      Writes(const VarDecl* V, const ASTContext& C) : m_V(V), m_C(C) {}
      /// The variable \p E is part of: `V`, `V(1, 2)`, `V.a[i]`.
      static const ValueDecl* rootOf(const Expr* E) {
        while (true) {
          E = E->IgnoreParenImpCasts();
          if (const auto* DRE = dyn_cast<DeclRefExpr>(E))
            return DRE->getDecl();
          if (const auto* ASE = dyn_cast<ArraySubscriptExpr>(E))
            E = ASE->getBase();
          else if (const auto* ME = dyn_cast<MemberExpr>(E))
            E = ME->getBase();
          else if (const auto* OCE = dyn_cast<CXXOperatorCallExpr>(E);
                   OCE && OCE->getNumArgs())
            E = OCE->getArg(0);
          else
            return nullptr;
        }
      }
      bool write(const Expr* LHS) {
        if (rootOf(LHS) != m_V)
          return true;
        std::string P = positionIn(LHS, m_V, m_C);
        Unknown |= P.empty();
        if (!P.empty()) {
          Storage.push_back(P);
          Positions.insert(Storage.back());
        }
        return !Unknown;
      }
      bool VisitBinaryOperator(BinaryOperator* BO) {
        return BO->isAssignmentOp() ? write(BO->getLHS()) : true;
      }
      bool VisitCXXOperatorCallExpr(CXXOperatorCallExpr* OCE) {
        if (OCE->isAssignmentOp() && OCE->getNumArgs())
          return write(OCE->getArg(0));
        return true;
      }
      bool VisitCallExpr(CallExpr* CE) {
        // Handing V to something that may change it.
        const FunctionDecl* Callee = CE->getDirectCallee();
        for (unsigned i = 0; i < CE->getNumArgs(); ++i) {
          const auto* DRE =
              dyn_cast<DeclRefExpr>(CE->getArg(i)->IgnoreParenImpCasts());
          if (!DRE || DRE->getDecl() != m_V || !Callee)
            continue;
          std::size_t P =
              i - (isa<CXXOperatorCallExpr>(CE) && isa<CXXMethodDecl>(Callee));
          if (P < Callee->getNumParams()) {
            QualType T = Callee->getParamDecl(P)->getType();
            Unknown |= T->isReferenceType() &&
                       !T.getNonReferenceType().isConstQualified();
          }
        }
        return !Unknown;
      }
    } W(V, C);
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    W.TraverseStmt(const_cast<Stmt*>(FD->getBody()));
    std::size_t K = Inherited + W.Positions.size();
    if (W.Unknown || !K || 3 * K > N)
      continue;
    Found.push_back({V, K, N});
  }
  return Found;
}

/// The calls \p FD makes with \p V as an argument or object, each with how
/// many times one run of \p FD makes it. A read of one constant position is
/// no such use.
llvm::SmallVector<std::pair<const FunctionDecl*, double>, 4>
usesOf(const FunctionDecl* FD, const VarDecl* V, const ASTContext& C) {
  llvm::SmallVector<std::pair<const CallExpr*, double>, 16> Calls;
  collectCalls(FD->getBody(), 1, C, Calls);
  llvm::SmallVector<std::pair<const FunctionDecl*, double>, 4> Uses;
  for (auto [CE, Mult] : Calls) {
    const FunctionDecl* Callee = CE->getDirectCallee();
    if (!Callee || !positionIn(CE, V, C).empty())
      continue;
    auto isV = [V](const Expr* A) {
      const auto* DRE = dyn_cast<DeclRefExpr>(withoutCopies(A));
      return DRE && DRE->getDecl() == V;
    };
    // V itself, or what a call made of V without anything else, as V's
    // transpose: a product of that works on V too.
    auto madeOfV = [&isV](const Expr* A) {
      if (isV(A))
        return true;
      if (const auto* MCE = dyn_cast<CXXMemberCallExpr>(withoutCopies(A)))
        return MCE->getNumArgs() == 0 && isV(MCE->getImplicitObjectArgument());
      return false;
    };
    bool Uses_ = llvm::any_of(CE->arguments(), madeOfV);
    if (const auto* MCE = dyn_cast<CXXMemberCallExpr>(CE))
      Uses_ |= isV(MCE->getImplicitObjectArgument());
    if (Uses_)
      Uses.emplace_back(Callee, Mult);
  }
  return Uses;
}

/// The function a user writes a derivative for: the template an
/// instantiation comes from, or the function itself.
const FunctionDecl* patternOf(const FunctionDecl* FD) {
  if (const FunctionDecl* P = FD->getTemplateInstantiationPattern())
    return P->getCanonicalDecl();
  return FD->getCanonicalDecl();
}

/// A finding, summed over the instantiations of one template.
struct Finding {
  double Share = 0;
  /// The work behind the share, per run of the gradient.
  double Work = 0;
  /// For a replay, what runs again, with what it costs.
  llvm::DenseMap<const FunctionDecl*, double> Rerun;
  const FunctionDecl* Fn = nullptr;
  const VarDecl* Var = nullptr;
  std::size_t Written = 0, Scalars = 0;
};

} // namespace

void adviseOnPerformance(Sema& S,
                         llvm::ArrayRef<AdvisedDerivative> Derivatives) {
  const ASTContext& C = S.getASTContext();
  CostModel Cost(C);

  // The derivatives generated here, by the canonical declaration a call names.
  llvm::DenseMap<const FunctionDecl*, const AdvisedDerivative*> Generated;
  for (const AdvisedDerivative& D : Derivatives)
    if (D.Derivative->hasBody())
      Generated[D.Derivative->getCanonicalDecl()] = &D;
  auto generated = [&Generated](const FunctionDecl* FD) {
    auto It = Generated.find(FD->getCanonicalDecl());
    return It == Generated.end() ? nullptr : It->second;
  };

  // Which functions return a structured object, for findStructured.
  llvm::DenseMap<const FunctionDecl*, Structured> Returns;
  for (const AdvisedDerivative& D : Derivatives) {
    if (!D.Original || !D.Original->hasBody())
      continue;
    for (const Structured& St : findStructured(D.Original, C, Returns)) {
      class ReturnsVar : public RecursiveASTVisitor<ReturnsVar> {
        const VarDecl* m_V;

      public:
        bool Found = false;
        explicit ReturnsVar(const VarDecl* V) : m_V(V) {}
        bool VisitReturnStmt(ReturnStmt* RS) {
          const Expr* E = RS->getRetValue();
          if (!E)
            return true;
          E = withoutCopies(E);
          // `return f(V)`, as for a pass that may give V back unchanged.
          if (const auto* CE = dyn_cast<CallExpr>(E))
            if (CE->getNumArgs() == 1)
              E = withoutCopies(CE->getArg(0));
          const auto* DRE = dyn_cast<DeclRefExpr>(E->IgnoreParenImpCasts());
          Found |= DRE && DRE->getDecl() == m_V;
          return true;
        }
      } R(St.Var);
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
      R.TraverseStmt(const_cast<Stmt*>(D.Original->getBody()));
      if (R.Found)
        Returns[D.Original->getCanonicalDecl()] = St;
    }
  }

  // One gradient at a time: how often it runs each derivative, and what the
  // runs cost in all.
  for (const AdvisedDerivative& Root : Derivatives) {
    if (Root.Mode != DiffMode::reverse || !Root.Original)
      continue;
    // Who calls whom, and how many times per run of the caller: each body is
    // read once. A call that leads back up the graph (recursion) is left out.
    llvm::DenseMap<
        const AdvisedDerivative*,
        llvm::SmallVector<std::pair<const AdvisedDerivative*, double>, 4>>
        Edges;
    llvm::SmallVector<const AdvisedDerivative*, 16> Order; // callers last
    llvm::DenseSet<const AdvisedDerivative*> Seen, OnPath;
    std::function<void(const AdvisedDerivative*)> visit =
        [&](const AdvisedDerivative* D) {
          Seen.insert(D);
          OnPath.insert(D);
          llvm::SmallVector<std::pair<const CallExpr*, double>, 16> Calls;
          CounterTrips Counters;
          collectCounters(D->Derivative->getBody(), C, Counters);
          collectCalls(D->Derivative->getBody(), 1, C, Calls, &Counters);
          for (auto [CE, Mult] : Calls)
            if (const FunctionDecl* Callee = CE->getDirectCallee())
              if (const AdvisedDerivative* Next = generated(Callee))
                if (!OnPath.count(Next)) {
                  Edges[D].emplace_back(Next, Mult);
                  if (!Seen.count(Next))
                    visit(Next);
                }
          OnPath.erase(D);
          Order.push_back(D);
        };
    visit(&Root);
    // Callers before callees, so a weight is complete before it is passed on.
    llvm::DenseMap<const AdvisedDerivative*, double> Weight;
    Weight[&Root] = 1;
    for (const AdvisedDerivative* D : llvm::reverse(Order))
      for (auto [Next, Mult] : Edges[D])
        Weight[Next] += Weight[D] * Mult;

    llvm::DenseMap<const AdvisedDerivative*, double> OwnCost;
    auto ownCost = [&](const AdvisedDerivative* D) {
      if (auto It = OwnCost.find(D); It != OwnCost.end())
        return It->second;
      Cost.countLoopsIn(D->Derivative->getBody());
      return OwnCost[D] =
                 Cost.of(D->Derivative->getBody(), [&](const FunctionDecl* FD) {
                   return generated(FD) != nullptr;
                 });
    };
    double Total = 0;
    for (auto [D, W] : Weight)
      Total += W * ownCost(D);
    if (Total <= 0)
      continue;

    // A pullback that runs its function's forward work again. The caller's
    // forward sweep ran the function already; its generated pullback runs the
    // forward sweep of the function once more before the reverse one, inline
    // or through the calls it makes, so it repeats about the function's work.
    llvm::DenseMap<const FunctionDecl*, Finding> Replays, Structures;
    for (auto [D, W] : Weight) {
      if (D->Mode != DiffMode::pullback || !D->Original)
        continue;
      // What a return computes is not run again: its pullback works out the
      // partials of that expression, which is derivative work, not replay.
      double Returned = 0;
      class Returns_ : public RecursiveASTVisitor<Returns_> {
      public:
        llvm::SmallVector<const Expr*, 2> Values;
        bool VisitReturnStmt(ReturnStmt* RS) {
          if (RS->getRetValue())
            Values.push_back(RS->getRetValue());
          return true;
        }
      } RV;
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
      RV.TraverseStmt(const_cast<Stmt*>(D->Original->getBody()));
      for (const Expr* E : RV.Values)
        Returned += Cost.of(E);
      double Replayed = std::max(0.0, Cost.of(D->Original) - Returned);
      // What in it costs most: the calls it makes, or else its own body.
      llvm::DenseMap<const FunctionDecl*, double> ReranWork;
      llvm::SmallVector<std::pair<const CallExpr*, double>, 16> Calls;
      collectCalls(D->Original->getBody(), 1, C, Calls);
      double ViaCalls = 0;
      for (auto [CE, Mult] : Calls) {
        const FunctionDecl* Callee = CE->getDirectCallee();
        if (!Callee || isCladRuntime(Callee))
          continue;
        double Work = Mult * Cost.of(Callee);
        ViaCalls += Work;
        ReranWork[patternOf(Callee)] += W * Work;
      }
      if (ViaCalls < Replayed / 2)
        ReranWork[patternOf(D->Original)] += W * (Replayed - ViaCalls);
      double Share = W * Replayed / Total;
      Finding& F = Replays[patternOf(D->Original)];
      F.Share += Share;
      F.Work += W * Replayed;
      F.Fn = patternOf(D->Original);
      for (auto [R, Work] : ReranWork)
        F.Rerun[R] += Work;
    }

    // An object constant except for a few entries, which the derivatives of
    // the function holding it treat as wholly varying.
    llvm::DenseMap<const FunctionDecl*, const AdvisedDerivative*> PullbackOf;
    for (auto [D, W] : Weight)
      if (D->Original && D->Mode == DiffMode::pullback)
        PullbackOf[D->Original->getCanonicalDecl()] = D;
    llvm::DenseMap<const FunctionDecl*, double> FnWeight;
    for (auto [D, W] : Weight)
      if (D->Original)
        FnWeight[D->Original->getCanonicalDecl()] += W;
    for (auto [FD, W] : FnWeight) {
      const FunctionDecl* Def = FD->getDefinition();
      if (!Def)
        continue;
      for (const Structured& St : findStructured(Def, C, Returns)) {
        // What the gradient spends on the derivatives of the calls that use
        // V: the pullback generated for one, or else about twice its work.
        double Work = 0;
        for (auto [Callee, Mult] : usesOf(Def, St.Var, C)) {
          auto It = PullbackOf.find(Callee->getCanonicalDecl());
          Work += Mult * (It != PullbackOf.end() ? ownCost(It->second)
                                                 : 2 * Cost.of(Callee));
        }
        double Share = W * Work / Total;
        Finding& F = Structures[patternOf(Def)];
        F.Share += Share;
        F.Work += W * Work;
        F.Fn = patternOf(Def);
        if (!F.Var || St.Written > F.Written) {
          F.Var = St.Var;
          F.Written = St.Written;
          F.Scalars = St.Scalars;
        }
      }
    }

    // Largest first, so that the remark to act on first is the first one read.
    auto report = [&](llvm::DenseMap<const FunctionDecl*, Finding>& Map,
                      llvm::function_ref<void(const Finding&)> Emit) {
      llvm::SmallVector<const Finding*, 8> Sorted;
      for (auto& [K, F] : Map)
        if (F.Share >= ReportShare && F.Work >= ReportWork)
          Sorted.push_back(&F);
      llvm::sort(Sorted, [](const Finding* A, const Finding* B) {
        return A->Share > B->Share;
      });
      for (const Finding* F : Sorted)
        Emit(*F);
    };
    report(Replays, [&](const Finding& F) {
      // The three that cost most, most first.
      using Cost = std::pair<double, const FunctionDecl*>;
      llvm::SmallVector<Cost, 8> Top;
      for (auto [FD, Work] : F.Rerun)
        Top.emplace_back(Work, FD);
      llvm::sort(
          Top, [](const auto& A, const auto& B) { return A.first > B.first; });
      std::string Names;
      bool Body = false;
      for (const auto& [Work, FD] : llvm::ArrayRef<Cost>(Top).take_front(3)) {
        if (FD == F.Fn) {
          Body = true;
          continue;
        }
        Names += (Names.empty() ? "'" : ", '") + FD->getNameAsString() + "'";
      }
      utils::diag(S, CladDiag::remark_perf_replay, F.Fn->getLocation())
          << F.Fn << unsigned(F.Share * 100 + 0.5) << Root.Original;
      if (Body)
        utils::diag(S, CladDiag::note_perf_replayed_body, F.Fn->getLocation());
      if (!Names.empty())
        utils::diag(S, CladDiag::note_perf_replayed, F.Fn->getLocation())
            << Names;
      utils::diag(S, CladDiag::note_perf_split, F.Fn->getLocation())
          << F.Fn->getNameAsString();
    });
    report(Structures, [&](const Finding& F) {
      utils::diag(S, CladDiag::remark_perf_structure, F.Var->getLocation())
          << F.Var << unsigned(F.Written) << unsigned(F.Scalars)
          << unsigned(F.Share * 100 + 0.5) << Root.Original;
      utils::diag(S, CladDiag::note_perf_structure, F.Fn->getLocation())
          << F.Fn;
    });
  }
}
} // namespace clad
