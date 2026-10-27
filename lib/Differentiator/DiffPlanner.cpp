#include "clad/Differentiator/DiffPlanner.h"

#include "ActivityAnalyzer.h"
#include "TBRAnalyzer.h"
#include "UsefulAnalyzer.h"

#include "clad/Differentiator/CladUtils.h"
#include "clad/Differentiator/DiffMode.h"
#include "clad/Differentiator/DiffRequest.h"
#include "clad/Differentiator/DynamicGraph.h"
#include "clad/Differentiator/ParseDiffArgsTypes.h"
#include "clad/Differentiator/Timers.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTLambda.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/DeclarationName.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/OperationKinds.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Type.h"
#include "clang/AST/TypeLoc.h"
#include "clang/Analysis/AnalysisDeclContext.h"
#include "clang/Analysis/CFG.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/Version.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Overload.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/TemplateDeduction.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/SaveAndRestore.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>

using namespace clang;

namespace clad {
bool DiffScheduler::isEnabled(SourceLocation Loc) const {
  return Loc.isValid() &&
         utils::isInCladInterval(m_Sema.getSourceManager(), m_Interval, Loc);
}

void DiffScheduler::PlanRoot(DiffRequest& request,
                             llvm::function_ref<bool()> Prepare) {
  SourceLocation Loc = request.CallContext->getEndLoc();
  if (!isEnabled(Loc)) {
    if (Loc.isValid() && request.VerboseDiags)
      utils::diag(m_Sema, DiagnosticsEngine::Warning,
                  request.CallContext->getBeginLoc(),
                  "clad differentiation call is disabled by "
                  "'#pragma clad OFF'");
    return;
  }
  // Defer frontend validation until after the pragma check, so disabled calls
  // cannot produce target- or option-specific Clad diagnostics.
  if (Prepare && !Prepare())
    return;
  if (m_Planning) {
    m_DeferredRequests.push_back(request);
    return;
  }
  llvm::SaveAndRestore<bool> Planning(m_Planning, true);
  m_Collector.PlanRequest(request);
  // Planning a deferred root may append more requests. Copy it before the
  // vector grows so the collector keeps a stable reference.
  // NOLINTNEXTLINE(modernize-loop-convert)
  for (size_t i = 0; i < m_DeferredRequests.size(); ++i) {
    DiffRequest Deferred = m_DeferredRequests[i];
    m_Collector.PlanRequest(Deferred);
  }
  m_DeferredRequests.clear();
}

DiffCollector::DiffCollector(DynamicGraph<DiffRequest>& requestGraph, Sema& S,
                             OwnedAnalysisContexts& AllAnalysisDC)
    : m_DiffRequestGraph(requestGraph), m_AllAnalysisDC(AllAnalysisDC),
      m_Sema(S) {}

bool DiffCollector::TraverseLambdaExpr(LambdaExpr* LE) {
  if (!m_ParentReq)
    return RecursiveASTVisitor<DiffCollector>::TraverseLambdaExpr(LE);
  // Create a nested request
  DiffRequest LambdaReq = *m_ParentReq;

  LambdaReq.Function = LE->getCallOperator();
  LambdaReq.Functor = LE->getLambdaClass();
  llvm::SaveAndRestore<DiffRequest*> Saved(m_ParentReq, &LambdaReq);
  RecursiveASTVisitor<DiffCollector>::TraverseLambdaExpr(LE);

  return true;
}

bool DiffCollector::TraverseFunctionDeclOnce(const DiffRequest& request) {
  if (llvm::is_contained(m_Traversed, request))
    return true;
  const FunctionDecl* FD = request.Function;
  bool walked = llvm::any_of(
      m_Traversed, [FD](const DiffRequest& R) { return R.Function == FD; });
  m_Traversed.push_back(request);
  // The first walk of the function reported what is wrong with its body.
  DiagnosticsEngine& Diags = m_Sema.getDiagnostics();
  bool suppressed = Diags.getSuppressAllDiagnostics();
  Diags.setSuppressAllDiagnostics(suppressed || walked);
  TimedAnalysisRegion R(FD->getNameAsString());
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  bool result = TraverseDecl(const_cast<FunctionDecl*>(FD));
  Diags.setSuppressAllDiagnostics(suppressed);
  return result;
}

  bool DiffCollector::PlanNestedRequest(DiffRequest& request) {
    // Lazily-scheduled requests (pushforward/pullback/higher-order, built in
    // DerivativeBuilder::HandleNestedDiffRequest) never pass through the static
    // TU walk, so their planning must be done here on demand. For now this only
    // records the early-return flag: m_TopMostReq is left null, so
    // VisitCallExpr and VisitDeclRefExpr bail and no sub-requests are spawned
    // -- only VisitReturnStmt runs over this request's body.
    // FIXME: The varied and useful analyses are still force-disabled in
    // HandleNestedDiffRequest; only TBR is served here.
    const FunctionDecl* Def =
        request.Function ? request.Function->getDefinition() : nullptr;
    if (!Def || !Def->hasBody())
      return true;

    // Give the request the analysis context TBR needs. DiffRequest runs TBR
    // lazily out of shouldBeRecorded(), so building the context is all that is
    // required for a dynamically scheduled reverse-mode request to drop the
    // stores its reverse sweep never reads back. Hessian mode benefits most:
    // its reverse pass runs once per direction over a first-order derivative.
    //
    // Only the reverse-mode family consults shouldBeRecorded(); a context built
    // for, say, a pushforward would never be looked at. TBRAnalyzer::Analyze
    // seeds its state from request.Function's parameters and matches them
    // against the decls the CFG refers to, so the two must be the same
    // FunctionDecl -- a redeclaration has its own ParmVarDecls, which would
    // silently match nothing. shouldBeRecorded() refuses to analyze a lambda
    // call operator for the same kind of reason, so do not arm TBR for one.
    bool usesTBR = request.Mode == DiffMode::reverse ||
                   request.Mode == DiffMode::pullback ||
                   request.Mode == DiffMode::jacobian ||
                   request.Mode == DiffMode::reverse_mode_forward_pass;
    if (usesTBR && request.EnableTBRAnalysis && !request.m_AnalysisDC &&
        request.Function == Def && !isLambdaCallOperator(Def)) {
      // One context per function: the same nested request is planned again
      // whenever another call site asks for its derivative.
      AnalysisDeclContext*& ADC = m_NestedAnalysisDC[Def];
      if (!ADC) {
        clang::CFG::BuildOptions Options;
        m_AllAnalysisDC.push_back(std::make_unique<AnalysisDeclContext>(
            /*AnalysisDeclContextManager=*/nullptr, Def, Options));
        ADC = m_AllAnalysisDC.back().get();
      }
      request.m_AnalysisDC = ADC;
    }

    llvm::SaveAndRestore<DiffRequest*> Saved(m_ParentReq, &request);
    return TraverseStmt(Def->getBody());
  }

  /// Whether \p S contains a call forward mode may skip differentiating: a
  /// plain call that is not an operator or member call (see
  /// BaseForwardModeVisitor::VisitCallExpr). Missing one is safe -- the
  /// visitor then differentiates through everything.
  static bool hasPushforwardCandidate(const Stmt* S) {
    llvm::SmallVector<const Stmt*, 32> workList{S};
    while (!workList.empty()) {
      const Stmt* cur = workList.pop_back_val();
      if (!cur)
        continue;
      if (isa<CallExpr>(cur) && !isa<CXXOperatorCallExpr>(cur) &&
          !isa<CXXMemberCallExpr>(cur))
        return true;
      for (const Stmt* child : cur->children())
        workList.push_back(child);
    }
    return false;
  }

  /// Whether the planner has to work out which calls of \p request contribute
  /// to its derivative: only a forward-mode request differentiates along a
  /// single known direction, and a body without a candidate call has nothing
  /// to decide.
  static bool needsCallActivity(const DiffRequest& request) {
    return request.Mode == DiffMode::forward && request->isDefined() &&
           hasPushforwardCandidate(request->getBody());
  }

  /// Seeds the varied set with the parameters \p request differentiates with
  /// respect to. A FieldDecl (a functor differentiated w.r.t. a field) cannot
  /// be seeded; the analyzer covers it by treating everything reached through
  /// `this` as varied.
  static void seedVariedDirection(DiffRequest& request) {
    for (const DiffInputVarInfo& dParam : request.DVI)
      if (const auto* VD = dyn_cast_or_null<VarDecl>(dParam.param))
        request.addVariedDecl(VD);
  }

  /// Runs varied analysis for \p request over \p AnalysisDC, which has to
  /// describe request.Function.
  static void runVariedAnalysis(DiffRequest& request,
                                AnalysisDeclContext* AnalysisDC) {
    TimedAnalysisRegion R("VA " + request.BaseFunctionName);
    VariedAnalyzer analyzer(AnalysisDC, request, request.getVariedStmt());
    analyzer.Analyze();
  }

  /// Runs varied analysis for one direction of \p request over \p AnalysisDC.
  /// A hessian reuses one forward request for row after row, so what the
  /// previous direction concluded is dropped first.
  static void analyzeDirection(DiffRequest& request,
                               AnalysisDeclContext* AnalysisDC) {
    request.resetActivityInfo();
    if (!AnalysisDC || !needsCallActivity(request))
      return;
    seedVariedDirection(request);
    runVariedAnalysis(request, AnalysisDC);
  }

  /// Whether the argument of the call \p E for its parameter \p i is passed
  /// by reference or pointer and names a variable of \p Caller that activity
  /// analysis leaves without an adjoint: a local it gives none, or, in a
  /// pullback it planned without a parameter, that parameter. Reverse mode
  /// has none to pass for it.
  static bool passesNoAdjoint(const CallExpr* E, std::size_t i,
                              const DiffRequest& Caller) {
    if (!Caller.EnableVariedAnalysis)
      return false;
    const auto* MD = dyn_cast<CXXMethodDecl>(E->getDirectCallee());
    if (isa<CXXOperatorCallExpr>(E) && MD && MD->isInstance())
      ++i;
    if (i >= E->getNumArgs() || !utils::IsReferenceOrPointerArg(E->getArg(i)))
      return false;
    const auto* DRE =
        dyn_cast<DeclRefExpr>(E->getArg(i)->IgnoreParenImpCasts());
    const auto* VD = DRE ? dyn_cast<VarDecl>(DRE->getDecl()) : nullptr;
    if (const auto* PVD = dyn_cast_or_null<ParmVarDecl>(VD))
      return Caller.Mode == DiffMode::pullback &&
             !Caller.HasIndependentParameter(PVD);
    return VD && VD->hasLocalStorage() && !VD->getType()->isReferenceType() &&
           !Caller.shouldHaveAdjoint(VD);
  }

  /// The assignments, increments and decrements in \p E, outermost first, in
  /// the order it reads them.
  static llvm::SmallVector<const Expr*, 2> writesIn(const Expr* E) {
    class Finder : public RecursiveASTVisitor<Finder> {
    public:
      llvm::SmallVector<const Expr*, 2> Writes;
      bool TraverseBinaryOperator(BinaryOperator* BO) {
        if (BO->getOpcode() != BO_Assign)
          return RecursiveASTVisitor::TraverseBinaryOperator(BO);
        Writes.push_back(BO);
        return true;
      }
      // `+=` and its kind are not traversed as a BinaryOperator.
      bool TraverseCompoundAssignOperator(CompoundAssignOperator* CAO) {
        Writes.push_back(CAO);
        return true;
      }
      bool TraverseUnaryOperator(UnaryOperator* UO) {
        if (!UO->isIncrementDecrementOp())
          return RecursiveASTVisitor::TraverseUnaryOperator(UO);
        Writes.push_back(UO);
        return true;
      }
    } F;
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    F.TraverseStmt(const_cast<Expr*>(E));
    return F.Writes;
  }

  /// Whether the index \p Idx writes nothing, or steps an integer variable
  /// once, as `i++`, `--i`, `i = e` and `i += e` do: reverse mode then has the
  /// index from before the write.
  static bool isIndexStep(const Expr* Idx) {
    Idx = Idx->IgnoreParenImpCasts();
    if (writesIn(Idx).empty())
      return true;
    const Expr* Target = nullptr;
    const Expr* Value = nullptr;
    if (const auto* UO = dyn_cast<UnaryOperator>(Idx)) {
      if (UO->isIncrementDecrementOp())
        Target = UO->getSubExpr();
    } else if (const auto* BO = dyn_cast<BinaryOperator>(Idx)) {
      if (BO->getOpcode() == BO_Assign || BO->getOpcode() == BO_AddAssign) {
        Target = BO->getLHS();
        Value = BO->getRHS();
      }
    }
    const auto* DRE =
        Target ? dyn_cast<DeclRefExpr>(Target->IgnoreParens()) : nullptr;
    return DRE && DRE->getType()->isIntegerType() &&
           (!Value || writesIn(Value).empty());
  }

  /// Whether every write in the lvalue \p E steps one of its array indices,
  /// as in `a[i++]`: reverse mode then has its address from before them.
  static bool writesOnlyIndices(const Expr* E) {
    E = E->IgnoreParenImpCasts();
    if (const auto* ASE = dyn_cast<ArraySubscriptExpr>(E))
      return isIndexStep(ASE->getIdx()) && writesOnlyIndices(ASE->getBase());
    if (const auto* ME = dyn_cast<MemberExpr>(E))
      return writesOnlyIndices(ME->getBase());
    return writesIn(E).empty();
  }

  static bool allArgumentsAreLiterals(const CallExpr::arg_range& args,
                                      const DiffRequest* request) {
    return std::none_of(args.begin(), args.end(), [&request](const Expr* A) {
      return request->isVaried(A);
    });
  }

  static Expr* getOverloadExpr(Sema& S, DeclContext* DC, DiffRequest& R,
                               bool* foundAnyDecl = nullptr,
                               bool quiet = false) {
    if (foundAnyDecl)
      *foundAnyDecl = false;
    // Error estimation only uses forward mode derivatives if they are
    // user-prodived to handle builtin derivatives. If found, we have to change
    // the mode of the request.
    if (R.EnableErrorEstimation && R.Mode == DiffMode::pullback &&
        utils::canUsePushforwardInRevMode(
            R.Function, dyn_cast_or_null<CallExpr>(R.CallContext))) {
      R.Mode = DiffMode::pushforward;
      R.EnableErrorEstimation = false;
      if (Expr* overload = getOverloadExpr(S, DC, R, foundAnyDecl, quiet)) {
        R.DVI.clear();
        return overload;
      }
      R.Mode = DiffMode::pullback;
      R.EnableErrorEstimation = true;
    }
    llvm::SmallVector<const ValueDecl*, 4> diffParams{};
    for (const DiffInputVarInfo& VarInfo : R.DVI)
      diffParams.push_back(VarInfo.param);
    QualType dTy = utils::GetDerivativeType(S, R.Function, R.Mode, diffParams,
                                            /*forCustomDerv=*/true,
                                            /*shouldUseRestoreTracker=*/false);
    // We disable diagnostics for methods and operators because they often have
    // ideantical names: `constructor_pullback`, `operator_star_pushforward`,
    // etc. If we turn it on, every such operator will trigger diagnostics
    // because of our STL and Kokkos custom derivatives.
    // FIXME: Add a way to silence the diagnostics.
    bool enableDiagnostics = !quiet && !isa<CXXMethodDecl>(R.Function) &&
                             !R->isOverloadedOperator() &&
                             R.BaseFunctionName != "forward" &&
                             !R.EnableErrorEstimation;

    ASTContext& C = S.getASTContext();
    auto LookupPropagator = [&C, &S, &DC](const std::string& Name) {
      IdentifierInfo* II = &C.Idents.get(Name);
      DeclarationNameInfo DNInfo(II, utils::GetValidSLoc(S));
      LookupResult Found(S, DNInfo, Sema::LookupOrdinaryName);
      S.LookupQualifiedName(Found, DC);
      return Found;
    };

    std::string Name = R.ComputeDerivativeName();
    LookupResult Found = LookupPropagator(Name);
    // This is a hack to reuse the builtin derivatives for vector mode.
    if (Found.empty() && R.Mode == DiffMode::vector_pushforward)
      Found = LookupPropagator(R.BaseFunctionName + "_pushforward");

    if (Found.empty())
      return nullptr; // Nothing found.

    if (foundAnyDecl)
      *foundAnyDecl = true;

    // A custom reverse_forw / pullback carries a trailing pullback_state<S>
    // clad does not synthesize. Extend the expected signature with it before
    // matching, or the declaration is rejected and the diagnostic omits the
    // state the pullback must accept. S comes from the reverse_forw: the
    // candidate itself, or -- for a pullback request -- its paired
    // reverse_forw.
    auto findStateParam = [](const LookupResult& LR) -> QualType {
      for (const NamedDecl* ND : LR)
        // getAsFunction sees through a FunctionTemplateDecl to its templated
        // FunctionDecl -- a templated reverse_forw (e.g. the thrust
        // derivatives) declares the state param there. The payload itself must
        // still be non-dependent for GetPullbackStatePayload to recognize it.
        if (const FunctionDecl* FD = ND->getUnderlyingDecl()->getAsFunction())
          for (const ParmVarDecl* PVD : FD->parameters()) {
            QualType PT = PVD->getType();
            if (PT->isLValueReferenceType() &&
                !utils::GetPullbackStatePayload(PT.getNonReferenceType())
                     .isNull())
              return PT.getNonReferenceType();
          }
      return {};
    };
    // A payload that owns storage -- a clad::tape, say -- has no copy
    // constructor, so a pullback carrying one must take the carrier by
    // reference. Whichever form the candidate declares is the one to expect.
    auto pullbackTakesStateByRef = [](const LookupResult& LR) {
      for (const NamedDecl* ND : LR)
        if (const FunctionDecl* FD = ND->getUnderlyingDecl()->getAsFunction())
          for (const ParmVarDecl* PVD : FD->parameters()) {
            QualType PT = PVD->getType();
            if (!utils::GetPullbackStatePayload(PT.getNonReferenceType())
                     .isNull())
              return PT->isLValueReferenceType();
          }
      return false;
    };
    QualType stateParam;
    if (R.Mode == DiffMode::reverse_mode_forward_pass)
      stateParam = findStateParam(Found);
    else if (R.Mode == DiffMode::pullback)
      stateParam = findStateParam(
          LookupPropagator(R.BaseFunctionName + "_reverse_forw"));
    if (!stateParam.isNull())
      if (const auto* FPT = dTy->getAs<FunctionProtoType>()) {
        // The reverse_forw always takes the state by reference: it is an
        // out-param it fills during the forward sweep.
        QualType stateArg = (R.Mode == DiffMode::reverse_mode_forward_pass ||
                             pullbackTakesStateByRef(Found))
                                ? C.getLValueReferenceType(stateParam)
                                : stateParam;
        llvm::SmallVector<QualType, 8> params(FPT->param_types().begin(),
                                              FPT->param_types().end());
        params.push_back(stateArg);
        dTy = C.getFunctionType(FPT->getReturnType(), params,
                                FPT->getExtProtoInfo());
        // Remember it so Derive appends a matching argument when it builds the
        // overload call for this custom derivative.
        R.PullbackStateParam = stateArg;
      }

    TemplateSpecCandidateSet FailedCandidates(R.CallContext->getBeginLoc(),
                                              /*ForTakingAddress=*/false);
    if (Expr* overload =
            utils::MatchOverloadType(S, dTy, Found, FailedCandidates))
      return overload;

    // For an original that returns `const T&`, the expected pushforward
    // returns ValueAndPushforward<const T&, const T&>. With two reference
    // members that struct is neither default-constructible nor assignable,
    // which breaks differentiating the generated code again at higher
    // orders (issue #1872), so the min/max builtins return
    // ValueAndPushforward<T, T> instead. Accept them by retrying the lookup
    // with the expectation relaxed the same way. Retrying only after the
    // exact match fails keeps genuinely reference-returning pushforwards
    // (e.g. the STL builtins for operator[], at, front, back) matching as
    // before.
    if (const auto* FPT = dTy->getAs<FunctionProtoType>()) {
      auto* CTSD = dyn_cast_or_null<ClassTemplateSpecializationDecl>(
          FPT->getReturnType()->getAsCXXRecordDecl());
      TemplateDecl* VPDecl =
          utils::LookupTemplateDeclInCladNamespace(S, "ValueAndPushforward");
      if (CTSD && CTSD->getSpecializedTemplate() == VPDecl) {
        const auto& tArgs = CTSD->getTemplateArgs();
        QualType valueTy = tArgs[0].getAsType();
        QualType pushforwardTy = tArgs[1].getAsType();
        if (valueTy->isLValueReferenceType() &&
            valueTy.getNonReferenceType().isConstQualified() &&
            C.hasSameType(valueTy, pushforwardTy)) {
          QualType relaxedTy =
              valueTy.getNonReferenceType().getUnqualifiedType();
          QualType relaxedRetTy =
              utils::InstantiateTemplate(S, VPDecl, {relaxedTy, relaxedTy});
          QualType relaxedDTy = C.getFunctionType(
              relaxedRetTy, FPT->getParamTypes(), FPT->getExtProtoInfo());
          TemplateSpecCandidateSet RelaxedCandidates(
              R.CallContext->getBeginLoc(),
              /*ForTakingAddress=*/false);
          if (Expr* overload = utils::MatchOverloadType(S, relaxedDTy, Found,
                                                        RelaxedCandidates))
            return overload;
        }
      }
    }

    if (!enableDiagnostics)
      return nullptr;

    // We did not match the found candidates. Warn and offer the user hints.
    auto errId = S.Diags.getCustomDiagID(
        DiagnosticsEngine::Error,
        "user-defined derivative for %0 was provided but not used; "
        "expected signature %1 does not match");
    S.Diag(R.CallContext->getBeginLoc(), errId) << R.Function << dTy;
    FailedCandidates.NoteCandidates(S, R.CallContext->getBeginLoc());
    utils::DiagnoseSignatureMismatch(S, dTy, Found);

    return nullptr;
  }

  bool DiffCollector::LookupCustomDerivativeDecl(DiffRequest& request,
                                                 bool* foundAnyDecl) {
    if (foundAnyDecl)
      *foundAnyDecl = false;
    NamespaceDecl* cladNS =
        utils::LookupNSD(m_Sema, "clad", /*shouldExist=*/true);
    NamespaceDecl* customDerNS =
        utils::LookupNSD(m_Sema, "custom_derivatives",
                         /*shouldExist=*/false, cladNS);
    if (!customDerNS)
      return false;
    if (request.Mode == DiffMode::unknown)
      return true;

    const Expr* callSite = request.CallContext;
    assert(callSite && "Called lookup without CallContext");

    const Decl* fnDecl = nullptr;
    // Check if the callSite is not associated with a shadow declaration.
    if (request.Mode == DiffMode::pushforward ||
        request.Mode == DiffMode::pullback ||
        request.Mode == DiffMode::vector_pushforward) {
      if (const auto* ME = dyn_cast<CXXMemberCallExpr>(callSite)) {
        fnDecl = ME->getMethodDecl();
      } else if (const auto* CE = dyn_cast<CallExpr>(callSite)) {
        const Expr* Callee = CE->getCallee()->IgnoreParenCasts();
        if (const auto* DRE = dyn_cast<DeclRefExpr>(Callee))
          fnDecl = DRE->getFoundDecl();
        else if (const auto* MemberE = dyn_cast<MemberExpr>(Callee))
          fnDecl = MemberE->getFoundDecl().getDecl();
      } else if (const auto* CtorExpr = dyn_cast<CXXConstructExpr>(callSite)) {
        fnDecl = CtorExpr->getConstructor();
      }
      // A synthesized request -- the pushforward of a hessian assembled from
      // vector products -- carries the clad::hessian call as its context. Its
      // callee is clad's own template, not the function being differentiated,
      // and must not anchor the namespace lookup.
      bool anchorsRequestFn = false;
      if (const auto* ND = dyn_cast_or_null<NamedDecl>(fnDecl)) {
        const Decl* underlying = ND->getUnderlyingDecl();
        if (const auto* FTD = dyn_cast<FunctionTemplateDecl>(underlying))
          underlying = FTD->getTemplatedDecl();
        if (const auto* UFD = dyn_cast<FunctionDecl>(underlying)) {
          const FunctionDecl* pattern =
              request.Function->getTemplateInstantiationPattern();
          anchorsRequestFn =
              UFD->getCanonicalDecl() == request.Function->getCanonicalDecl() ||
              (pattern &&
               UFD->getCanonicalDecl() == pattern->getCanonicalDecl());
        }
      }
      if (!anchorsRequestFn)
        fnDecl = request.Function;
    } else
      fnDecl = request.Function;
    assert(request.Mode != DiffMode::unknown &&
           "Called lookup without specified DiffMode");

    auto LookupOverload = [this, fnDecl, &request,
                           quiet = foundAnyDecl != nullptr](NamespaceDecl* NSD,
                                                            bool* foundDecl) {
      DeclContext* DC = NSD;
      if (isa<CXXMethodDecl>(request.Function))
        DC = utils::LookupNSD(m_Sema, "class_functions",
                              /*shouldExist=*/false, DC);
      else
        DC = utils::FindDeclContext(
            m_Sema, DC, utils::GetCustomDerivativeLookupContext(fnDecl));
      if (!DC)
        return static_cast<Expr*>(nullptr);
      return getOverloadExpr(m_Sema, DC, request, foundDecl, quiet);
    };

    // Look for the user overrides namespace nested inside custom_derivatives
    NamespaceDecl* overridesNS = utils::LookupNSD(
        m_Sema, "overrides", /*shouldExist=*/false, customDerNS);

    // clad::custom_derivatives::overrides (user overrides, higher priority).
    bool foundOverridesDecl = false;
    if (overridesNS) {
      if (Expr* overload = LookupOverload(overridesNS, &foundOverridesDecl)) {
        // Overload found. Mark the request as custom derivative overrides and
        // save the set of overloads to process later.
        request.CustomDerivative = overload;
        return true;
      }
      // Overrides declaration found but signature mismatch; do not fall back to
      // builtin.
      if (foundOverridesDecl) {
        if (foundAnyDecl)
          *foundAnyDecl = true;
        return false;
      }
    }

    // clad::custom_derivatives (builtins, fallback).
    if (Expr* overload = LookupOverload(customDerNS, foundAnyDecl)) {
      // Overload found. Mark the request as custom derivative and save
      // the set of overloads to process later.
      request.CustomDerivative = overload;
      return true;
    }

    return false;
  }

  bool DiffCollector::TraverseCUDAKernelCallExpr(CUDAKernelCallExpr* KCE) {
    return WalkUpFromCUDAKernelCallExpr(KCE) &&
           TraverseStmt(KCE->getCallee()) &&
           llvm::all_of(KCE->arguments(),
                        [this](Expr* arg) { return TraverseStmt(arg); });
  }

  bool DiffCollector::VisitCallExpr(CallExpr* E) {
    if (!m_TopMostReq)
      return true;

    // Check if we should look into this.
    DiffRequest request;

    FunctionDecl* FD = E->getDirectCallee();
    if (!FD)
      return true;

    bool nonDiff = false;
    bool hasNoMemoryInputForPointerOrRefReturn = false;
    // If the function contains annotation of non_differentiable, then Clad
    // should not produce any derivative expression for that function call,
    // and the function call in the primal should be used as it is.
    if (clad::utils::hasNonDifferentiableAttribute(E))
      nonDiff = true;

    // A call whose object is a marked non-differentiable type -- a stream
    // operation (`os << x`) or a std::streambuf method -- is opaque. Do not
    // schedule a derivative for it or recurse into its body; otherwise clad
    // descends into the whole non-differentiable I/O machinery
    // (sentry/streambuf/scope_guard/char_traits). The primal call is emitted
    // as-is in the derivative.
    if (clad::utils::callOperatesOnNonDifferentiableType(m_Sema, E))
      return true;

    if (diagnoseByteCopy(E))
      return true;

    request.VerboseDiags = false;
    request.inheritAnalysesFrom(*m_TopMostReq);
    request.EmitPortingHints = m_TopMostReq->EmitPortingHints;
    request.EnableErrorEstimation = m_TopMostReq->EnableErrorEstimation;
    request.CallContext = E;

    const auto* MD = dyn_cast<CXXMethodDecl>(FD);
    if (MD) {
      if (isLambdaCallOperator(MD) &&
          (m_TopMostReq->Mode == DiffMode::reverse ||
           m_TopMostReq->Mode == DiffMode::pullback)) {
        request.EnableVariedAnalysis = false;
        return true;
      }
      // Captured lambda pushforward is built directly
      if (isLambdaCallOperator(MD) && !MD->getParent()->field_empty() &&
          (m_TopMostReq->Mode == DiffMode::forward ||
           m_TopMostReq->Mode == DiffMode::vector_forward_mode))
        return true;
      const CXXRecordDecl* CD = MD->getParent();
      if (clad::utils::hasNonDifferentiableAttribute(CD))
        nonDiff = true;
    }

    QualType returnType = FD->getReturnType();
    bool hasPointerOrRefReturn = utils::isNonConstReferenceType(returnType) ||
                                 returnType->isPointerType();
    // Such a call moves a derivative through a global, whatever its
    // arguments.
    bool touchesGlobal = touchesActiveGlobal(E, FD);
    // Don't build propagators for calls that do not contribute in
    // differentiable way to the result.
    if (!(MD && MD->isInstance()) && !hasPointerOrRefReturn && !touchesGlobal &&
        allArgumentsAreLiterals(E->arguments(), m_ParentReq))
      nonDiff = true;
    // In reverse mode, calls without memory parameters normally have no
    // adjoint destination for a pointer or reference return. Defer the final
    // decision for instance calls until custom derivatives are known because
    // their implicit object can carry the adjoint.
    hasNoMemoryInputForPointerOrRefReturn =
        !utils::hasMemoryTypeParams(FD) && hasPointerOrRefReturn &&
        m_TopMostReq->Mode == DiffMode::reverse;
    // Skip reverse-mode scheduling for integral-return helper calls that
    // cannot accumulate through memory arguments. Keep this narrow to avoid
    // suppressing diagnostics on variadic/non-helper calls.
    const bool HasPointerOrReferenceParam = std::any_of(
        FD->parameters().begin(), FD->parameters().end(),
        [](const ParmVarDecl* PVD) {
          QualType ParamType = PVD->getType();
          return ParamType->isPointerType() || ParamType->isReferenceType();
        });
    if (m_TopMostReq->Mode == DiffMode::reverse && !FD->isVariadic() &&
        HasPointerOrReferenceParam && !utils::hasMemoryTypeParams(FD) &&
        returnType->isIntegralOrEnumerationType())
      nonDiff = true;

    if (nonDiff && m_TopMostReq->Mode != DiffMode::reverse)
      return true;

    request.Function = FD;
    request.CallContext = E;
    // A pushforward has no way to hand an adjoint to a global.
    bool canUsePushforwardInRevMode = m_TopMostReq->Mode == DiffMode::reverse &&
                                      !request.EnableErrorEstimation &&
                                      !touchesGlobal &&
                                      utils::canUsePushforwardInRevMode(FD, E);

    std::string FDName = FD->getNameAsString();
#if CLANG_VERSION_MAJOR < 16
    if (clang::AnalysisDeclContext::isInStdNamespace(FD) &&
        (FDName == "move" || FDName == "forward"))
      return true;
#endif
    // FIXME: hessians require second derivatives, i.e. apart from the
    // pushforward, we also need to schedule pushforward_pullback.
    if (m_ParentReq->CustomDerivative || m_ParentReq->Mode == DiffMode::unknown)
      request.Mode = DiffMode::unknown;
    else if (m_TopMostReq->Mode == DiffMode::forward ||
             m_TopMostReq->Mode == DiffMode::hessian ||
             canUsePushforwardInRevMode)
      request.Mode = DiffMode::pushforward;
    else if (m_TopMostReq->Mode == DiffMode::reverse ||
             m_TopMostReq->Mode == DiffMode::pullback)
      request.Mode = DiffMode::pullback;
    else if (m_TopMostReq->Mode == DiffMode::vector_forward_mode ||
             m_TopMostReq->Mode == DiffMode::jacobian ||
             m_TopMostReq->Mode == DiffMode::vector_pushforward) {
      request.Mode = DiffMode::vector_pushforward;
    } else {
      assert(0 && "unexpected mode.");
      return true;
    }

    // FIXME: Use CLAD_ELIDABLE_REVERSE_FORW
    if (request.Mode == DiffMode::pullback &&
        (FDName == "cudaMemcpy" || FDName == "begin" || FDName == "end"))
      return true;

    if (request.Mode != DiffMode::pushforward &&
        request.Mode != DiffMode::vector_pushforward) {
      // CUDA device function call in global kernel gradient
      bool useCUDA = !m_TopMostReq->CUDAGlobalArgsIndexes.empty();
      bool droppedNoAdjoint = false;
      auto planParams = [&](DiffRequest& R, bool dropNoAdjoint) {
        R.DVI.clear();
        R.CUDAGlobalArgsIndexes.clear();
        for (size_t i = 0, e = FD->getNumParams(); i < e; ++i) {
          const auto* paramDecl = FD->getParamDecl(i);
          if (clad::utils::hasNonDifferentiableAttribute(paramDecl))
            continue;

          // Reverse mode asks for the pullback without a parameter it has
          // no adjoint to pass for. Plan that pullback, not one it never
          // calls.
          if (dropNoAdjoint && passesNoAdjoint(E, i, *m_ParentReq)) {
            droppedNoAdjoint = true;
            continue;
          }

          // If a parameter is non-differentiable, don't include
          // it in the DVI to avoid dynamic partial derivatives.
          const ParmVarDecl* PVD = nullptr;
          Expr* ArgE = E->getArg(i)->IgnoreParens()->IgnoreParenCasts();
          if (const auto* DRE = dyn_cast<DeclRefExpr>(ArgE))
            PVD = dyn_cast<ParmVarDecl>(DRE->getDecl());
          bool IsDifferentiableArg =
              !PVD || m_ParentReq->HasIndependentParameter(PVD);

          if (!utils::isArrayOrPointerType(paramDecl->getType()) ||
              IsDifferentiableArg)
            R.DVI.push_back(paramDecl);
          // FIXME: If we cannot deduce whether the argument is
          // differentiable, we should still add it to CUDAGlobalArgsIndexes.
          // i.e. remove `&& PVD`.
          // We know we should use atomic ops here
          if (useCUDA && PVD && IsDifferentiableArg)
            R.CUDAGlobalArgsIndexes.push_back(i);
        }
      };
      planParams(request, /*dropNoAdjoint=*/true);
      // A user's pullback takes every parameter. Where there is one, plan it,
      // and reverse mode passes a zero adjoint for what activity analysis
      // leaves without one: every call then uses the user's derivative. A
      // pullback whose signature is wrong is then reported against the one
      // the user has to write.
      if (droppedNoAdjoint) {
        DiffRequest full = request;
        planParams(full, /*dropNoAdjoint=*/false);
        full.BaseFunctionName = utils::ComputeEffectiveFnName(FD);
        bool declared = false;
        if (LookupCustomDerivativeDecl(full, &declared) || declared)
          planParams(request, /*dropNoAdjoint=*/false);
      }
    }

    // Warn if we find pullbacks.
    if (canUsePushforwardInRevMode && m_TopMostReq->Mode == DiffMode::reverse) {
      DiffRequest R = request;
      R.BaseFunctionName = utils::ComputeEffectiveFnName(R.Function);
      R.Mode = DiffMode::pullback;
      if (LookupCustomDerivativeDecl(R)) {
        utils::diag(m_Sema, DiagnosticsEngine::Warning,
                    R.CallContext->getBeginLoc(),
                    "unused function '%0_pullback'; '%0' is a real-domain, "
                    "single-argument function and only pushforward is required")
            << R.Function->getName() << R.CallContext->getSourceRange();
        // Collect the unused decls.
        llvm::SmallVector<const NamedDecl*, 2> UnusedDecls;
        if (const auto* OvE = dyn_cast<OverloadExpr>(R.CustomDerivative)) {
          UnusedDecls.append(OvE->decls_begin(), OvE->decls_end());
        } else {
          const auto* DRE = cast<DeclRefExpr>(R.CustomDerivative);
          const auto* UnusedD = cast<FunctionDecl>(DRE->getDecl());
          UnusedDecls.push_back(UnusedD);
        }
        unsigned noteId = m_Sema.Diags.getCustomDiagID(DiagnosticsEngine::Note,
                                                       "%0 is unused");
        for (const NamedDecl* UnusedD : UnusedDecls) {
          SourceLocation L = UnusedD->getLocation();
          m_Sema.Diag(L, noteId) << UnusedD << L;
        }
      }
    }

    return CollectRequest(request, nonDiff,
                          hasNoMemoryInputForPointerOrRefReturn);
  }

  bool DiffCollector::PlanRequest(DiffRequest& request) {
    llvm::SaveAndRestore<const DiffRequest*> Top(m_TopMostReq, &request);
    llvm::SaveAndRestore<DiffRequest*> Parent(m_ParentReq, nullptr);
    m_Traversed.clear();
    if ((request.Mode == DiffMode::reverse ||
         request.Mode == DiffMode::pullback) &&
        request.EnableVariedAnalysis)
      seedVariedDirection(request);
    if (request.Function->hasAttr<CUDAGlobalAttr>())
      for (size_t I = 0; I < request.Function->getNumParams(); ++I)
        request.CUDAGlobalArgsIndexes.push_back(I);
    bool Result = CollectRequest(request);
    m_Traversed.clear();
    return Result;
  }

  bool
  DiffCollector::CollectRequest(DiffRequest& request, bool nonDiff,
                                bool hasNoMemoryInputForPointerOrRefReturn) {
    const auto* E = cast<CallExpr>(request.CallContext);
    const auto* FD = request.Function;
    if (request.BaseFunctionName.empty())
      request.BaseFunctionName =
          utils::ComputeEffectiveFnName(request.Function);

    // FIXME: Here we copy all varied declarations down to the pullback, has to
    // be removed once AA and TBR are completely reworked, with better
    // branch-merging.
    if (m_ParentReq)
      for (const auto& decl : m_ParentReq->getVariedDecls())
        request.addVariedDecl(decl);

    llvm::SaveAndRestore<DiffRequest*> Saved(m_ParentReq, &request);
    if (request.Function->getDefinition())
      request.Function = request.Function->getDefinition();

    // Functions with side-effects require TBR.
    bool isNonConstMethod = false;
    if (const auto* MD = dyn_cast<CXXMethodDecl>(FD))
      isNonConstMethod = MD && MD->isInstance() && !MD->isConst();
    else if (const auto* OCE = dyn_cast<CXXOperatorCallExpr>(E))
      isNonConstMethod =
          utils::isNonConstReferenceType(OCE->getArg(0)->getType());
    bool requestTBR =
        request.EnableTBRAnalysis &&
        (request.Mode == DiffMode::pullback || isNonConstMethod) &&
        utils::hasMemoryTypeParams(request.Function) && request->isDefined() &&
        E->getDirectCallee();
    bool shouldUseRestoreTracker =
        utils::shouldUseRestoreTracker(request.Function);
    // Literal-only calls like log(2) are non-differentiable, so their
    // pushforward is never called. Builtin pushforwards use floating-point
    // types, so log<int> reports a type mismatch.
    bool hasCustomPullback =
        (!nonDiff || request.Mode != DiffMode::pushforward) &&
        LookupCustomDerivativeDecl(request);
    // Share one request between early classification and final scheduling.
    // Only nested pullbacks have a caller that consumes a reverse-forward pair.
    const bool isNestedPullback =
        request.Mode == DiffMode::pullback && Saved.get();
    DiffRequest forwPassRequest;
    if (isNestedPullback) {
      forwPassRequest.Function = request.Function;
      forwPassRequest.BaseFunctionName = request.BaseFunctionName;
      forwPassRequest.Mode = DiffMode::reverse_mode_forward_pass;
      forwPassRequest.CallContext = request.CallContext;
      forwPassRequest.UseRestoreTracker = shouldUseRestoreTracker;
      forwPassRequest.EnableLoopAnalysis = request.EnableLoopAnalysis;
    }

    if (hasNoMemoryInputForPointerOrRefReturn) {
      const auto* calledMethod = dyn_cast<CXXMethodDecl>(FD);
      bool isRefReturningInstanceCall =
          request.Mode == DiffMode::pullback && calledMethod &&
          calledMethod->isInstance() &&
          utils::isNonConstReferenceType(request->getReturnType());
      bool hasCustomReverseForw = false;
      if (isRefReturningInstanceCall)
        hasCustomReverseForw = LookupCustomDerivativeDecl(forwPassRequest);

      // A reverse_forw still needs a matching pullback. Preserve the existing
      // forward-only contract only when a custom reverse_forw has no custom
      // pullback, as with smart-pointer and reference-wrapper helpers.
      if (!isRefReturningInstanceCall ||
          (hasCustomReverseForw && !hasCustomPullback))
        nonDiff = true;
    }

    // Enzyme differentiates the emitted IR, so the only thing clad produces
    // for such a request is the __enzyme_autodiff call. Analysing the body and
    // recursing into the call graph would schedule pullbacks nothing consumes,
    // and would fail on callees clad cannot differentiate but Enzyme can. The
    // request itself is still scheduled below.
    if (!request.use_enzyme &&
        (!(hasCustomPullback || nonDiff) || requestTBR)) {
      clang::CFG::BuildOptions Options;
      std::unique_ptr<AnalysisDeclContext> AnalysisDC =
          std::make_unique<AnalysisDeclContext>(
              /*AnalysisDeclContextManager=*/nullptr, request.Function,
              Options);

      if (needsCallActivity(request)) {
        seedVariedDirection(request);
        runVariedAnalysis(request, AnalysisDC.get());
      } else if (request.EnableVariedAnalysis && request->isDefined()) {
        // The opt-in analysis; requests it seeds are seeded by the collector.
        runVariedAnalysis(request, AnalysisDC.get());
      }

      if (m_TopMostReq->EnableUsefulAnalysis) {
        TimedAnalysisRegion R("UA " + request.BaseFunctionName);
        UsefulAnalyzer analyzer(AnalysisDC.get(), request.getUsefulDecls());
        analyzer.Analyze(request.Function);
      }

      m_AllAnalysisDC.push_back(std::move(AnalysisDC));
      request.m_AnalysisDC = m_AllAnalysisDC.back().get();

      //  Recurse into call graph. With a custom derivative the body is walked
      // for TBR alone and never differentiated.
      llvm::SaveAndRestore<bool> SavedReplaced(m_BodyIsReplaced,
                                               hasCustomPullback || nonDiff);
      TraverseFunctionDeclOnce(request);

      if (requestTBR) {
        TimedAnalysisRegion R("TBR " + request.BaseFunctionName);
        ParamInfo& modifiedParams = request.getModifiedParams();
        ParamInfo& usedParams = request.getUsedParams();
        TBRAnalyzer analyzer(request.m_AnalysisDC, request.getToBeRecorded(),
                             &modifiedParams, &usedParams);
        analyzer.Analyze(request);
        if (modifiedParams[FD].empty())
          shouldUseRestoreTracker = false;
        if (DiffRequest* Parent = Saved.get()) {
          Parent->addFunctionModifiedParams(FD, modifiedParams[FD]);
          Parent->addFunctionUsedParams(FD, usedParams[FD]);
        }
      }

      if (request.Mode == DiffMode::hessian ||
          request.Mode == DiffMode::hessian_diagonal) {
        // A hessian assembled from hessian-vector products needs one
        // pushforward of the function, whose direction is a run-time
        // argument. Schedule it here so that it is derived, and not merely
        // declared, by the time HessianModeVisitor asks for its pullback: a
        // pullback of a function whose body does not exist yet comes out
        // empty. The decision is recorded on the request: the visitor
        // consumes it and does not decide again, so the two sides cannot
        // disagree.
        request.UseHessianVectorProducts =
            request.Mode == DiffMode::hessian &&
            utils::canUseHessianVectorProducts(request.Function);

        // A pointer parameter outside the requested set would get a null
        // tangent, which forward mode only reads as zero through a const
        // pointee. Derive per direction instead, which diagnoses the
        // dependent non-const pointer rather than dereferencing null at run
        // time.
        if (request.UseHessianVectorProducts && request.Args)
          for (const ParmVarDecl* PVD : request.Function->parameters()) {
            QualType T = PVD->getType();
            if (!utils::isArrayOrPointerType(T) ||
                utils::GetValueType(T).isConstQualified())
              continue;
            bool requested = std::any_of(request.DVI.begin(), request.DVI.end(),
                                         [PVD](const DiffInputVarInfo& dVar) {
                                           return dVar.param == PVD;
                                         });
            if (!requested) {
              request.UseHessianVectorProducts = false;
              break;
            }
          }

        // Build the per-direction forward requests up front: deriving per
        // direction consumes them, and the vector-product scheme must first
        // know that none of them resolves to a custom forward derivative,
        // which only the per-direction scheme honors.
        llvm::SmallVector<DiffRequest, 8> forwRequests;
        bool hasCustomForwardDerivative = false;
        DiffRequest forwRequest = request;
        forwRequest.Mode = DiffMode::forward;
        forwRequest.CallUpdateRequired = false;
        if (request.Mode == DiffMode::hessian_diagonal)
          forwRequest.RequestedDerivativeOrder = 2;
        for (const auto& dParam : request.DVI) {
          const auto* PVD = cast<ParmVarDecl>(dParam.param);
          auto indexInterval = dParam.paramIndexInterval;
          bool perIndex = utils::isArrayOrPointerType(PVD->getType());
          std::size_t start = perIndex ? indexInterval.Start : 0;
          std::size_t finish = perIndex ? indexInterval.Finish : 1;
          for (std::size_t i = start; i < finish; ++i) {
            // FIXME: We shouldn't synthesize Args strings.
            std::string independentArgString = PVD->getNameAsString();
            if (perIndex)
              independentArgString += "[" + std::to_string(i) + "]";
            forwRequest.Args = utils::CreateStringLiteral(
                m_Sema.getASTContext(), independentArgString);
            forwRequest.UpdateDiffParamsInfo(m_Sema);
            if (perIndex && !forwRequest.DVI.empty())
              forwRequest.DVI.back().TotalCapacity = indexInterval.Finish;
            // Every row of an array parameter seeds the same VarDecl, so the
            // first row's analysis holds for all of them.
            if (i == start)
              analyzeDirection(forwRequest, request.m_AnalysisDC);
            // The request is reused across directions and the lookup only
            // writes on success; without this reset a direction with a
            // custom derivative leaks it into every direction after it.
            forwRequest.CustomDerivative = nullptr;
            hasCustomForwardDerivative |=
                LookupCustomDerivativeDecl(forwRequest);
            forwRequests.push_back(forwRequest);
          }
        }
        if (hasCustomForwardDerivative)
          request.UseHessianVectorProducts = false;

        if (request.UseHessianVectorProducts) {
          DiffRequest pushforwardRequest =
              request.pushforwardRequestForHessian(m_Sema);
          // A custom pushforward is free to take any signature, and the
          // wrapper builds its calls by position; derive per direction
          // instead.
          if (LookupCustomDerivativeDecl(pushforwardRequest))
            request.UseHessianVectorProducts = false;
          else
            m_DiffRequestGraph.addNode(pushforwardRequest, /*isSource=*/true);
        }

        if (!request.UseHessianVectorProducts)
          for (DiffRequest& fr : forwRequests)
            m_DiffRequestGraph.addNode(fr, /*isSource=*/true);
      }
    }

    if (isNestedPullback) {
      // TBR can prove that no state needs restoring, so refresh this before
      // scheduling the request.
      forwPassRequest.UseRestoreTracker = shouldUseRestoreTracker;
      QualType returnType = request->getReturnType();
      bool hasCustomReverseForw = LookupCustomDerivativeDecl(forwPassRequest);

      if (hasCustomReverseForw ||
          (!hasCustomPullback &&
           (utils::returnsAdjoint(returnType) || shouldUseRestoreTracker ||
            utils::allocatesIntoRecordParam(request.Function)))) {
        m_DiffRequestGraph.addNode(forwPassRequest, /*isSource=*/true);
      }
      if (!nonDiff && Saved.get() && request.PullbackStateParam.isNull() &&
          forwPassRequest.PullbackStateParam.isNull() &&
          !utils::allocatesIntoRecordParam(request.Function))
        Saved.get()->ReplayableCalls.insert(E);
    }

    // The call, the stores around it and its pullback each use an argument.
    // One that writes something is evaluated once: record its writes for the
    // caller, and say so where reverse mode cannot keep it from before them.
    if (!nonDiff && Saved.get() &&
        (Saved.get()->Mode == DiffMode::reverse ||
         Saved.get()->Mode == DiffMode::pullback ||
         Saved.get()->Mode == DiffMode::jacobian)) {
      const FunctionDecl* Callee = request.Function;
      std::size_t skip =
          isa<CXXOperatorCallExpr>(E) && isa<CXXMethodDecl>(Callee);
      for (std::size_t i = skip, e = E->getNumArgs();
           i < e && i - skip < Callee->getNumParams(); ++i) {
        const Expr* Arg = E->getArg(i);
        llvm::SmallVector<const Expr*, 2> Writes = writesIn(Arg);
        if (Writes.empty())
          continue;
        QualType T = Callee->getParamDecl(i - skip)->getType();
        if (T->isReferenceType() && Arg->isGLValue() && writesOnlyIndices(Arg))
          Saved.get()->ArgWrites[Arg] = std::move(Writes);
        else if (!T->isReferenceType() && Arg->getType()->isScalarType())
          Saved.get()->ArgWrites[Arg] = {};
        else
          utils::diag(m_Sema, DiagnosticsEngine::Warning, Arg->getBeginLoc(),
                      "clad evaluates this argument more than once, and "
                      "repeats what it writes each time")
              << Arg->getSourceRange();
      }
    }

    // The pullback replays the globals the call assigns: the caller puts them
    // back around it, in the order they are declared. Only those something
    // reads: the callee, or the caller. A custom pullback replays nothing.
    if (!nonDiff && !hasCustomPullback && request.Mode == DiffMode::pullback &&
        Saved.get()) {
      const GlobalAccess& Callee = getReachableGlobalAccess(request.Function);
      const GlobalAccess& Caller =
          getReachableGlobalAccess(Saved.get()->Function);
      std::vector<const VarDecl*> Globals;
      for (const VarDecl* VD : Callee.AnyWrites)
        if (Callee.AnyReads.count(VD) || Caller.AnyReads.count(VD))
          Globals.push_back(VD);
      llvm::sort(Globals, [](const VarDecl* L, const VarDecl* R) {
        return L->getLocation() < R->getLocation();
      });
      if (!Globals.empty())
        Saved.get()->CalleeGlobalWrites[E] = std::move(Globals);
    }

    if (!nonDiff && request.Mode != DiffMode::unknown)
      m_DiffRequestGraph.addNode(request, /*isSource=*/true);

    return true;
  }

  bool DiffCollector::diagnoseByteCopy(const CallExpr* E) {
    QualType copied = utils::GetByteCopiedFloatingType(E);
    if (copied.isNull() || !m_ParentReq || m_BodyIsReplaced)
      return false;
    // A defaulted assignment copies an array member this way; point at the
    // assignment that called it.
    const auto* MD = dyn_cast<CXXMethodDecl>(m_ParentReq->Function);
    bool defaultedAssign =
        MD && MD->isDefaulted() &&
        (MD->isCopyAssignmentOperator() || MD->isMoveAssignmentOperator());
    const Expr* site = defaultedAssign && m_ParentReq->CallContext
                           ? m_ParentReq->CallContext
                           : E;
    SourceLocation L = site->getBeginLoc();
    if (defaultedAssign) {
      DiffMode mode = m_ParentReq->Mode;
      bool forward = mode == DiffMode::forward ||
                     mode == DiffMode::pushforward ||
                     mode == DiffMode::vector_forward_mode ||
                     mode == DiffMode::vector_pushforward;
      utils::diag(m_Sema, DiagnosticsEngine::Error, L,
                  "cannot differentiate the defaulted %select{copy|move}0 "
                  "assignment of %1: it copies %2 with %3, which has no "
                  "derivative")
          << MD->isMoveAssignmentOperator() << MD->getParent() << copied
          << E->getDirectCallee();
      utils::diag(m_Sema, DiagnosticsEngine::Note, L,
                  "provide custom %select{'operator_equal_reverse_forw' and "
                  "'operator_equal_pullback'|'operator_equal_pushforward'}0 "
                  "for %1")
          << forward << MD->getParent();
      return true;
    }
    utils::diag(m_Sema, DiagnosticsEngine::Error, L,
                "cannot differentiate %0 copying %1, which holds "
                "floating-point values; its derivative would be zero")
        << E->getDirectCallee() << copied;
    utils::diag(m_Sema, DiagnosticsEngine::Note, L,
                "copy the values with assignments, which clad can "
                "differentiate");
    return true;
  }

  bool DiffCollector::VisitBinaryOperator(BinaryOperator* BO) {
    // Runs while TraverseFunctionDeclOnce walks a request's body, with
    // m_ParentReq pointing at that request (still local, added to the graph
    // only after the walk) -- so this eagerly records into the very request
    // reverse mode will consume. Outside a request body m_ParentReq is null.
    if (m_ParentReq && BO->getOpcode() == BO_Assign)
      if (const VarDecl* p = DiffRequest::AllocCallInfo::recognize(BO->getRHS())
                                 .getInPlaceReallocPtr(BO->getLHS()))
        m_ParentReq->InPlaceReallocPtrs.insert(p);
    return true;
  }

  /// Whether reverse mode can carry a derivative through a global of type
  /// \p T: a floating-point scalar, or a plain struct made only of those.
  /// Pointers, references and integers (counters, flags) cannot, and neither
  /// can an array yet: reverse mode does not compile a global array's adjoint.
  static bool isFloatingAggregate(QualType T) {
    T = T.getCanonicalType();
    if (T->isRealFloatingType())
      return true;
    if (const auto* RD = T->getAsCXXRecordDecl()) {
      if (!RD->hasDefinition() || !RD->isAggregate() || RD->getNumBases())
        return false;
      return !RD->field_empty() &&
             std::all_of(RD->field_begin(), RD->field_end(),
                         [](const FieldDecl* F) {
                           return isFloatingAggregate(F->getType());
                         });
    }
    return false;
  }

  const DiffCollector::GlobalAccess&
  DiffCollector::getGlobalAccess(const FunctionDecl* FD) {
    auto found = m_GlobalAccess.find(FD);
    if (found != m_GlobalAccess.end())
      return found->second;

    class Collector : public RecursiveASTVisitor<Collector> {
      GlobalAccess& m_Access;
      /// The variable an lvalue stores into: `g`, `g.v`, `g[i]`.
      static const DeclRefExpr* assignedVar(const Expr* E) {
        while (true) {
          E = E->IgnoreParenImpCasts();
          if (const auto* ME = dyn_cast<MemberExpr>(E)) {
            if (ME->isArrow())
              return nullptr;
            E = ME->getBase();
          } else if (const auto* ASE = dyn_cast<ArraySubscriptExpr>(E)) {
            // `p[i]` stores through a pointer, not into p.
            E = ASE->getBase()->IgnoreParenImpCasts();
            if (!E->getType()->isArrayType())
              return nullptr;
          } else
            break;
        }
        return dyn_cast<DeclRefExpr>(E);
      }
      static bool isTracked(const DeclRefExpr* DRE) {
        const auto* VD = DRE ? dyn_cast<VarDecl>(DRE->getDecl()) : nullptr;
        return VD && VD->isFileVarDecl() && !VD->getType().isConstQualified() &&
               isFloatingAggregate(VD->getType());
      }
      /// The variables a plain assignment stores into without reading them.
      llvm::SmallPtrSet<const DeclRefExpr*, 4> m_StoreOnly;
      /// A global the caller can copy aside and assign back: not an array,
      /// an atomic, or memory that lives on a device.
      static bool isCopyableGlobal(const DeclRefExpr* DRE) {
        const auto* VD = DRE ? dyn_cast<VarDecl>(DRE->getDecl()) : nullptr;
        return VD && (VD->isFileVarDecl() || VD->isStaticDataMember()) &&
               !VD->getType().isConstQualified() &&
               !VD->getType()->isArrayType() &&
               VD->getType().isTriviallyCopyableType(VD->getASTContext()) &&
               !VD->hasAttr<CUDADeviceAttr>() &&
               !VD->hasAttr<CUDASharedAttr>() &&
               !VD->hasAttr<CUDAConstantAttr>();
      }
      void addWrite(const DeclRefExpr* DRE) {
        if (isTracked(DRE))
          m_Access.Writes.insert(cast<VarDecl>(DRE->getDecl()));
        if (isCopyableGlobal(DRE))
          m_Access.AnyWrites.insert(cast<VarDecl>(DRE->getDecl()));
      }

    public:
      explicit Collector(GlobalAccess& A) : m_Access(A) {}
      [[nodiscard]] bool shouldVisitImplicitCode() const { return true; }
      // `sizeof(g)` and the like read nothing.
      bool TraverseUnaryExprOrTypeTraitExpr(UnaryExprOrTypeTraitExpr*) {
        return true;
      }
      // clang-22 added a trailing bool (TraverseQualifier); a defaulted
      // parameter matches both calls.
      bool TraverseDecltypeTypeLoc(DecltypeTypeLoc, bool = false) {
        return true;
      }
      bool VisitBinaryOperator(BinaryOperator* BO) {
        if (!BO->isAssignmentOp())
          return true;
        const DeclRefExpr* DRE = assignedVar(BO->getLHS());
        addWrite(DRE);
        if (DRE && BO->getOpcode() == BO_Assign)
          m_StoreOnly.insert(DRE);
        return true;
      }
      bool VisitUnaryOperator(UnaryOperator* UO) {
        if (UO->isIncrementDecrementOp())
          addWrite(assignedVar(UO->getSubExpr()));
        return true;
      }
      bool VisitDeclRefExpr(DeclRefExpr* DRE) {
        if (isTracked(DRE) && !m_StoreOnly.contains(DRE))
          m_Access.Reads.insert(cast<VarDecl>(DRE->getDecl()));
        if (isCopyableGlobal(DRE) && !m_StoreOnly.contains(DRE))
          m_Access.AnyReads.insert(cast<VarDecl>(DRE->getDecl()));
        // A global pointer or reference may alias a tracked global.
        const auto* VD = dyn_cast<VarDecl>(DRE->getDecl());
        if (VD && VD->isFileVarDecl() &&
            (VD->getType()->isPointerType() ||
             VD->getType()->isReferenceType()) &&
            VD->getType()->getPointeeType()->isRealFloatingType())
          m_Access.IndirectReads.insert(VD);
        return true;
      }
      bool VisitCallExpr(CallExpr* CE) {
        if (const FunctionDecl* Callee = CE->getDirectCallee())
          m_Access.Callees.push_back(Callee);
        return true;
      }
      bool VisitCXXConstructExpr(CXXConstructExpr* CE) {
        m_Access.Callees.push_back(CE->getConstructor());
        return true;
      }
    };

    GlobalAccess Access;
    if (const FunctionDecl* Def = FD->getDefinition())
      // A constructor's member initializers are part of the declaration, not
      // of the body; traverse the whole definition.
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
      Collector(Access).TraverseDecl(const_cast<FunctionDecl*>(Def));
    return m_GlobalAccess[FD] = std::move(Access);
  }

  const DiffCollector::GlobalAccess&
  DiffCollector::getReachableGlobalAccess(const FunctionDecl* Callee) {
    auto found = m_ReachableGlobalAccess.find(Callee);
    if (found != m_ReachableGlobalAccess.end())
      return found->second;
    GlobalAccess Reachable;
    llvm::SmallPtrSet<const FunctionDecl*, 8> Visited;
    llvm::SmallVector<const FunctionDecl*, 8> Worklist{Callee};
    while (!Worklist.empty()) {
      const FunctionDecl* FD = Worklist.pop_back_val();
      if (!FD || !Visited.insert(FD).second)
        continue;
      const GlobalAccess& A = getGlobalAccess(FD);
      Reachable.Reads.insert(A.Reads.begin(), A.Reads.end());
      Reachable.Writes.insert(A.Writes.begin(), A.Writes.end());
      Reachable.AnyWrites.insert(A.AnyWrites.begin(), A.AnyWrites.end());
      Reachable.AnyReads.insert(A.AnyReads.begin(), A.AnyReads.end());
      Reachable.IndirectReads.insert(A.IndirectReads.begin(),
                                     A.IndirectReads.end());
      Worklist.append(A.Callees.begin(), A.Callees.end());
    }
    return m_ReachableGlobalAccess[Callee] = std::move(Reachable);
  }

  bool DiffCollector::touchesActiveGlobal(const Expr* E,
                                          const FunctionDecl* Callee) {
    // Only reverse mode gives a global an adjoint; see VisitDeclRefExpr.
    if (!m_TopMostReq || !m_ParentReq ||
        (m_ParentReq->Mode != DiffMode::reverse &&
         m_ParentReq->Mode != DiffMode::pullback))
      return false;
    // A global is active when the differentiated function, anything it
    // calls, or the caller assigns it, as in `set_g(x); return read_g();`.
    const GlobalAccess& Top = getReachableGlobalAccess(m_TopMostReq->Function);
    const GlobalAccess& Parent = getGlobalAccess(m_ParentReq->Function);
    auto isActive = [&](const VarDecl* VD) {
      return Top.Writes.count(VD) || Parent.Writes.count(VD);
    };
    auto isReadByCaller = [&](const VarDecl* VD) {
      return Top.Reads.count(VD) || Parent.Reads.count(VD);
    };
    // A callee assigning a global the caller reads sends, or kills, the
    // derivative the caller reads.
    const GlobalAccess& Callees = getReachableGlobalAccess(Callee);
    if (std::any_of(Callees.Reads.begin(), Callees.Reads.end(), isActive) ||
        std::any_of(Callees.Writes.begin(), Callees.Writes.end(),
                    isReadByCaller)) {
      m_ParentReq->ActiveGlobalCalls.insert(E);
      return true;
    }

    // A global pointer or reference may alias an active global.
    if (!Callees.IndirectReads.empty() &&
        (!Top.Writes.empty() || !Parent.Writes.empty()))
      utils::diag(m_Sema, DiagnosticsEngine::Warning, E->getBeginLoc(),
                  "%0 reads the global %1 through a pointer or reference; "
                  "clad does not differentiate through it")
          << Callee << *Callees.IndirectReads.begin();
    return false;
  }

  bool DiffCollector::VisitDeclRefExpr(DeclRefExpr* DRE) {
    // m_TopMostReq is dereferenced below; it is null when PlanNestedRequest
    // walks a lazy request's body just to record its early-return flag, and no
    // global-adjoint discovery is wanted there.
    if (!m_ParentReq || !m_TopMostReq)
      return true;
    // FIXME: Add support for globals in other modes.
    if (m_ParentReq->Mode != DiffMode::reverse &&
        m_ParentReq->Mode != DiffMode::pullback)
      return true;

    // FIXME: In some cases, custom overloads are not found by DiffPlanner and
    // clad starts traversing builtin functions. This leads to some unnecessary
    // global adjoints being built and produces warnings in files where it's
    // impossible to expect them by tests. Because of this, global adjoints
    // are only created in the same file as the differentiated function for now.
    const clang::SourceManager& SM = m_Sema.getASTContext().getSourceManager();
    SourceLocation parentLoc = (*m_ParentReq)->getLocation();
    SourceLocation topMostLoc = (*m_TopMostReq)->getLocation();
    if (SM.getFileID(parentLoc) != SM.getFileID(topMostLoc))
      return true;

    if (const auto* VD = dyn_cast<VarDecl>(DRE->getDecl())) {
      if (VD->isFileVarDecl() && !VD->getType().isConstQualified()) {
        DiffRequest request;
        request.DeclarationOnly = true;
        request.Global = VD;
        m_DiffRequestGraph.addNode(request, /*isSource=*/true);
      }
    }
    return true;
  }

  bool DiffCollector::VisitCXXConstructExpr(CXXConstructExpr* E) {
    // Don't visit CXXConstructExpr in outside differentiated functions.
    if (!m_TopMostReq)
      return true;

    // FIXME: add support for the forward mode
    if ((m_ParentReq->Mode != DiffMode::reverse) &&
        (m_ParentReq->Mode != DiffMode::pullback))
      return true;

    // FIXME: This only happens to perform nested TBR.
    // Constructors are not yet suported
    if (m_ParentReq->CustomDerivative)
      return true;

    CXXConstructorDecl* CD = E->getConstructor();
    DiffRequest forwPassRequest;
    forwPassRequest.Function = CD;
    forwPassRequest.BaseFunctionName = "constructor";
    forwPassRequest.Mode = DiffMode::reverse_mode_forward_pass;
    forwPassRequest.CallContext = E;
    forwPassRequest.EmitPortingHints = m_TopMostReq->EmitPortingHints;
    forwPassRequest.EnableLoopAnalysis = m_TopMostReq->EnableLoopAnalysis;
    QualType recordTy = CD->getThisType()->getPointeeType();
    bool elideRevForw =
        utils::constructorReverseForwIsElidable(CD, m_Sema.getASTContext());
    // Cloning the body of a non-differentiable constructor yields an ill-formed
    // reverse-forward propagator: e.g. std::string's char-pointer constructor
    // (reached through a Kokkos::View label) delegates to a private member
    // (this->__init(...)), but the propagator is a static function with no
    // `this`, so CodeGen crashes. Build a propagator for a marked type only
    // when the user supplied a custom derivative, never by cloning the library
    // body.
    bool cloneBodyUnsafe =
        utils::isNonDifferentiableType(m_Sema, CD->getParent());
    if (LookupCustomDerivativeDecl(forwPassRequest) ||
        (!elideRevForw && !cloneBodyUnsafe))
      m_DiffRequestGraph.addNode(forwPassRequest, /*isSource=*/true);

    // Don't build propagators for calls that do not contribute in
    // differentiable way to the result.
    if (!touchesActiveGlobal(E, CD) &&
        allArgumentsAreLiterals(E->arguments(), m_ParentReq))
      return true;

    if (clad::utils::hasNonDifferentiableAttribute(CD->getParent()))
      return true;

    DiffRequest request;
    request.Function = CD;
    request.Mode = DiffMode::pullback;
    request.VerboseDiags = false;
    request.inheritAnalysesFrom(*m_TopMostReq);
    request.EmitPortingHints = m_TopMostReq->EmitPortingHints;

    for (const auto* paramDecl : CD->parameters())
      request.DVI.push_back(paramDecl);

    request.BaseFunctionName = "constructor";
    request.CallContext = E;

    if (m_ParentReq)
      for (const auto& decl : m_ParentReq->getVariedDecls())
        request.addVariedDecl(decl);

    llvm::SaveAndRestore<DiffRequest*> Saved(m_ParentReq, &request);
    if (request.Function->getDefinition())
      request.Function = request.Function->getDefinition();

    if (m_Sema.isStdInitializerList(recordTy, /*elemType=*/nullptr))
      return true;

    if (!LookupCustomDerivativeDecl(request)) {
      clang::CFG::BuildOptions Options;
      std::unique_ptr<AnalysisDeclContext> AnalysisDC =
          std::make_unique<AnalysisDeclContext>(
              /*AnalysisDeclContextManager=*/nullptr, request.Function,
              Options);
      if (request.EnableVariedAnalysis) {
        TimedAnalysisRegion R("VA " + request.BaseFunctionName);
        VariedAnalyzer analyzer(AnalysisDC.get(), request,
                                request.getVariedStmt());
        analyzer.Analyze();
      }
      // FIXME: Add proper support for objects in VA and UA.
      m_AllAnalysisDC.push_back(std::move(AnalysisDC));
      request.m_AnalysisDC = m_AllAnalysisDC.back().get();

      // Recurse into call graph.
      TraverseFunctionDeclOnce(request);
    }
    m_DiffRequestGraph.addNode(request, /*isSource=*/true);

    return true;
  }
  } // namespace clad
