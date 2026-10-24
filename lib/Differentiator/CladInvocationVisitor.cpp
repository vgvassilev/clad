#include "clad/Differentiator/CladInvocationVisitor.h"

#include "clad/Differentiator/CladUtils.h"
#include "clad/Differentiator/DiffMode.h"
#include "clad/Differentiator/DiffPlanner.h"
#include "clad/Differentiator/DiffRequest.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclGroup.h"
#include "clang/AST/DeclarationName.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/OperationKinds.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Type.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticSema.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/OperatorKinds.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/Specifiers.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Sema.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SaveAndRestore.h"

#include <cassert>

using namespace clang;

namespace clad {
/// Updates \c DR with the target function to differentiate.
///
/// \param[in,out] DR - The DiffRequest being updated.
/// \param[in] call A clad differentiation function call expression
/// \param[in] SemaRef Reference to Sema
static bool findTargetFunction(DiffRequest& DR, CallExpr* call, Sema& SemaRef) {
  struct Finder : RecursiveASTVisitor<Finder> {
    Sema& m_SemaRef;
    DiffRequest& m_DR;
    SourceLocation m_BeginLoc;
    Finder(Sema& SemaRef, DiffRequest& DR, SourceLocation beginLoc)
        : m_SemaRef(SemaRef), m_DR(DR), m_BeginLoc(beginLoc) {}

    // Required for visiting lambda declarations.
    [[nodiscard]] bool shouldVisitImplicitCode() const { return true; }
    bool VisitExplicitCastExpr(ExplicitCastExpr* CastE) {
      if (CastE->getCastKind() != CK_ReinterpretMemberPointer)
        return true;

      // Handle the cases where the user has forced overload resolution via an
      // explicit cast: clad::differentiate(static_cast<...>(f))
      assert(!m_DR.Function && "Function already set!");
      TraverseStmt(CastE->getSubExpr());

      // We will need to update the selected function.
      if (const auto* MPTy = CastE->getType()->getAs<MemberPointerType>()) {
        CXXRecordDecl* SelectedCXXRD = MPTy->getMostRecentCXXRecordDecl();
        QualType MPPointeeTy = MPTy->getPointeeType().getCanonicalType();
        const FunctionDecl* FD = m_DR.Function;

        if (FD->getType().getCanonicalType() == MPPointeeTy)
          return false; // Type match we are done.

        // We need to find the right function that we should differentiate.
        auto BestMatch = [&FD, &MPPointeeTy](const CXXRecordDecl* RD) {
          for (NamedDecl* ND : RD->lookup(FD->getDeclName())) {
            if (FD == ND)
              continue;
            auto* NewFD = cast<FunctionDecl>(ND);
            // FIXME: Call Sema::IsOverload to ensure both decls are compatible.
            if (NewFD->getType().getCanonicalType() == MPPointeeTy) {
              FD = NewFD;
              return false; // exit
            }
          }
          return true;
        };

        if (BestMatch(SelectedCXXRD)) {
          CXXBasePaths Paths(/*FindAmbiguities=*/false, /*RecordPaths=*/false,
                             /* DetectVirtual=*/false);
          auto BestMatchInBase = [&BestMatch](const CXXBaseSpecifier* Specifier,
                                              CXXBasePath& Path) {
            const auto* Base = Specifier->getType()->getAsCXXRecordDecl();
            return !BestMatch(Base);
          };
          SelectedCXXRD->lookupInBases(BestMatchInBase, Paths);
        }

        assert(m_DR.Function != FD && "Could not find the overload");
        m_DR.Function = FD;
      }

      return false; // we traversed the subexpression already, stop.
    }

    bool VisitDeclRefExpr(DeclRefExpr* DRE) {
      if (auto* VD = dyn_cast<VarDecl>(DRE->getDecl())) {
        QualType VarTy = VD->getType();
        // If variable is of class type, set the result to the overloaded call
        // operator method of the class type.
        if (auto* CXXRD = VarTy->getAsCXXRecordDecl())
          TraverseDecl(CXXRD);
        else
          TraverseStmt(VD->getInit());
      } else if (auto* FD = dyn_cast<FunctionDecl>(DRE->getDecl()))
        m_DR.Function = FD;
      return false;
    }

    bool VisitCXXRecordDecl(CXXRecordDecl* RD) {
      auto callOperatorDeclName =
          m_SemaRef.getASTContext().DeclarationNames.getCXXOperatorName(
              OverloadedOperatorKind::OO_Call);
      LookupResult R(m_SemaRef, callOperatorDeclName, SourceLocation(),
                     Sema::LookupNameKind::LookupMemberName);
      // We do not want diagnostics that would fire because of this lookup.
      R.suppressDiagnostics();
      m_SemaRef.LookupQualifiedName(R, RD);

      // Emit error diagnostics
      if (R.empty()) {
        utils::diag(m_SemaRef, DiagnosticsEngine::Error, m_BeginLoc,
                    "%0 has no defined operator()")
            << RD << m_BeginLoc;
        return false;
      }

      if (!R.isSingleResult()) {
        utils::diag(m_SemaRef, DiagnosticsEngine::Error, m_BeginLoc,
                    "%0 has multiple definitions of operator(); "
                    "multiple definitions of call operators are not supported")
            << RD << m_BeginLoc;

        // Emit diagnostics for candidate functions
        for (auto oper = R.begin(), operEnd = R.end(); oper != operEnd;
             ++oper) {
          auto* candidateFn = cast<CXXMethodDecl>(oper.getDecl());
          m_SemaRef.NoteOverloadCandidate(candidateFn, candidateFn);
        }
        return false;
      }

      NamedDecl* FoundDecl = R.getFoundDecl();
      AccessSpecifier FoundDeclAccess = FoundDecl->getAccess();
      if (FoundDeclAccess != AccessSpecifier::AS_public) {
        // Compute access specifier name so that it can be used in
        // diagnostic message.
        const char* callOperatorAS =
            (FoundDeclAccess == AccessSpecifier::AS_private ? "private"
                                                            : "protected");
        utils::diag(m_SemaRef, DiagnosticsEngine::Error, m_BeginLoc,
                    "%0 contains %1 call operator; differentiation of "
                    "non-public call operators is not supported")
            << RD << callOperatorAS << m_BeginLoc;

        // Compute if the access specifier of the found operator is implicit.
        bool isImplicit = true;
        for (auto* decl : RD->decls()) {
          if (decl == FoundDecl)
            break;
          if (isa<AccessSpecDecl>(decl)) {
            isImplicit = false;
            break;
          }
        }
        // Emit diagnostics for the found call operator
        m_SemaRef.Diag(FoundDecl->getBeginLoc(), diag::note_access_natural)
            << (unsigned)(FoundDeclAccess == AccessSpecifier::AS_protected)
            << isImplicit;

        return false;
      }

      auto* CallOperator = cast<CXXMethodDecl>(FoundDecl);
      m_DR.Function = CallOperator;
      m_DR.Functor = CallOperator->getParent();
      // Mark the declaration as used to instantiate templates, etc.
      bool OdrUse = !CallOperator->isVirtual() ||
                    CallOperator->getDevirtualizedMethod(
                        /*Base=*/nullptr, m_SemaRef.getLangOpts().AppleKext);
      m_SemaRef.MarkAnyDeclReferenced(m_BeginLoc, CallOperator, OdrUse);
      return false;
    }
  } finder(SemaRef, DR, call->getArg(0)->getBeginLoc());
  finder.TraverseStmt(call->getArg(0));

  assert(cast<NamespaceDecl>(call->getDirectCallee()->getDeclContext())
                 ->getName() == "clad" &&
         "Should be called for clad:: special functions!");
  return finder.m_DR.Function;
}

void CladInvocationVisitor::Walk(DeclGroupRef DGR) {
  llvm::SaveAndRestore<bool> Traversing(m_Traversing, true);
  // An OFF declaration can contain ON calls; filter calls, not declarations.
  for (Decl* D : DGR)
    TraverseDecl(D);
}

bool CladInvocationVisitor::TraverseDecl(Decl* D) {
  const FunctionDecl* Enclosing = m_EnclosingFD;
  if (const auto* FD = dyn_cast_or_null<FunctionDecl>(D))
    Enclosing = FD;
  llvm::SaveAndRestore<const FunctionDecl*> Saved(m_EnclosingFD, Enclosing);
  return RecursiveASTVisitor::TraverseDecl(D);
}

/// Whether \p FD is a clad entry point. They are annotated with the first
/// letter of the mode they ask for.
static bool isCladEntryPoint(const FunctionDecl* FD) {
  if (!FD)
    return false;
  const auto* A = FD->getAttr<AnnotateAttr>();
  if (!A)
    return false;
  llvm::StringRef Mode = A->getAnnotation();
  return Mode == "D" || Mode == "G" || Mode == "H" || Mode == "J" ||
         Mode == "E";
}

/// Finds the first clad entry-point call in a statement.
class CladCallFinder : public RecursiveASTVisitor<CladCallFinder> {
public:
  const CallExpr* Found = nullptr;
  bool VisitCallExpr(CallExpr* E) {
    if (!isCladEntryPoint(E->getDirectCallee()))
      return true;
    Found = E;
    return false;
  }
};

/// Reports variables whose initialiser the compiler works out before clad
/// is handed the declaration, so that clad never gets to put the derivative
/// into the call.
class ConstantInitChecker : public RecursiveASTVisitor<ConstantInitChecker> {
  Sema& m_Sema;
  const DiffScheduler& m_Scheduler;

public:
  ConstantInitChecker(Sema& S, const DiffScheduler& Scheduler)
      : m_Sema(S), m_Scheduler(Scheduler) {}

  bool VisitVarDecl(VarDecl* VD) {
    // Only a variable the language requires to be initialised by a constant
    // expression is beyond help. An ordinary one is worked out again after
    // clad has rewritten the call, and comes out right.
    if (!VD->isConstexpr() && !VD->hasAttr<ConstInitAttr>())
      return true;
    Expr* Init = VD->getInit();
    if (!Init)
      return true;
    CladCallFinder Finder;
    Finder.TraverseStmt(Init);
    if (!Finder.Found)
      return true;
    // Use the scheduler's root policy so disabled calls do not receive a
    // constant-initializer warning for a derivative that was not requested.
    if (!m_Scheduler.isEnabled(Finder.Found->getEndLoc()))
      return true;
    utils::diag(m_Sema, DiagnosticsEngine::Warning, Finder.Found->getBeginLoc(),
                "clad cannot put the derivative into this call: the compiler "
                "works out the initialiser of '%0' before clad is handed the "
                "declaration")
        << VD->getName() << Finder.Found->getSourceRange();
    utils::diag(m_Sema, DiagnosticsEngine::Note, Finder.Found->getBeginLoc(),
                "call clad from a constexpr function, keep the result in an "
                "ordinary variable there, and evaluate that function here");
    return true;
  }
};

void DiagnoseConstantInitRequests(Sema& S, const DiffScheduler& Scheduler,
                                  DeclGroupRef DGR) {
  ConstantInitChecker Checker(S, Scheduler);
  for (Decl* D : DGR)
    Checker.TraverseDecl(D);
}

bool CladInvocationVisitor::TraverseCUDAKernelCallExpr(
    CUDAKernelCallExpr* KCE) {
  return WalkUpFromCUDAKernelCallExpr(KCE) && TraverseStmt(KCE->getCallee()) &&
         llvm::all_of(KCE->arguments(),
                      [this](Expr* arg) { return TraverseStmt(arg); });
}

bool CladInvocationVisitor::VisitCallExpr(CallExpr* E) {
  FunctionDecl* FD = E->getDirectCallee();
  if (!isCladEntryPoint(FD))
    return true;

  DiffRequest request;
  request.VerboseDiags = true;
  request.CallUpdateRequired = true;
  request.CallContext = E;
  request.ImmediateContext = m_EnclosingFD;
  m_Scheduler.PlanRoot(request, [&] { return PrepareRequest(request, E, FD); });
  return true;
}

bool CladInvocationVisitor::PrepareRequest(DiffRequest& request, CallExpr* E,
                                           const FunctionDecl* Invocation) {
  if (!findTargetFunction(request, E, m_Sema))
    return false;
  if (request.UpdateDiffOptions(m_Sema, E->getEndLoc(), m_Options, Invocation))
    return false;

  // The Jacobian ABI has no return-derivative slot. Reject non-void
  // primals before scheduling or resolving custom derivatives.
  if (request.Mode == DiffMode::jacobian &&
      !request.Function->getReturnType()->isVoidType()) {
    const FunctionDecl* PrimalFD = request.Function;
    utils::diag(m_Sema, DiagnosticsEngine::Error,
                request.CallContext->getBeginLoc(),
                "jacobian mode currently requires function %0 to return void; "
                "provide differentiable outputs through pointer, reference, or "
                "array parameters")
        << PrimalFD;
    SourceLocation NoteLoc = PrimalFD->getReturnTypeSourceRange().getBegin();
    if (NoteLoc.isInvalid())
      NoteLoc = PrimalFD->getLocation();
    utils::diag(m_Sema, DiagnosticsEngine::Note, NoteLoc,
                "%0 declared here with return type %1")
        << PrimalFD << PrimalFD->getReturnType();
    return false;
  }

  request.Args = E->getArg(1);
  request.UpdateDiffParamsInfo(m_Sema);

  // A void-returning function's derivative returns the tangent of the one
  // parameter it writes through. With several such parameters there is no
  // one tangent to return, and picking one would be a guess, so say so
  // rather than hand back a derivative whose result cannot be read.
  // Differentiating a wrapper that returns the wanted output remains the
  // way to ask for it.
  if (request.Mode == DiffMode::forward) {
    llvm::SmallVector<const ParmVarDecl*, 2> outputParams =
        request.getOutputParams();
    if (outputParams.size() > 1) {
      // Point at the differentiation that asked for this, not at the
      // function: nothing is wrong with the function, and it may live in
      // a header the user does not own. The parameters it writes through
      // get a note, so the reason is still reachable from the message.
      SourceLocation outLoc = request.Args->getBeginLoc();
      utils::diag(m_Sema, DiagnosticsEngine::Error, outLoc,
                  "attempted to differentiate '%0', which returns void and "
                  "writes through %1 parameters; forward mode returns a "
                  "single tangent, so differentiate a wrapper returning "
                  "the output you need")
          << request.Function->getNameAsString()
          << static_cast<unsigned>(outputParams.size()) << outLoc;
      for (const ParmVarDecl* PVD : outputParams)
        utils::diag(m_Sema, DiagnosticsEngine::Note, PVD->getLocation(),
                    "'%0' is written through here")
            << PVD->getNameAsString() << PVD->getLocation();
      return false;
    }
  }
  return true;
}

} // namespace clad
