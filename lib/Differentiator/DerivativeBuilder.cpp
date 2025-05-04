//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// version: $Id: ClangPlugin.cpp 7 2013-06-01 22:48:03Z v.g.vassilev@gmail.com $
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#include "clad/Differentiator/DerivativeBuilder.h"

#include "JacobianModeVisitor.h"

#include "clad/Differentiator/BaseForwardModeVisitor.h"
#include "clad/Differentiator/CladUtils.h"
#include "clad/Differentiator/Compatibility.h"
#include "clad/Differentiator/DiffMode.h"
#include "clad/Differentiator/DiffPlanner.h"
#include "clad/Differentiator/DynamicGraph.h"
#include "clad/Differentiator/ErrorEstimator.h"
#include "clad/Differentiator/HessianModeVisitor.h"
#include "clad/Differentiator/PushForwardModeVisitor.h"
#include "clad/Differentiator/ReverseModeForwPassVisitor.h"
#include "clad/Differentiator/ReverseModeVisitor.h"
#include "clad/Differentiator/StmtClone.h"
#include "clad/Differentiator/Timers.h"
#include "clad/Differentiator/VectorForwardModeVisitor.h"
#include "clad/Differentiator/VectorPushForwardModeVisitor.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/OperationKinds.h"
#include "clang/AST/TemplateBase.h"
#include "clang/AST/Type.h"
#include "clang/Basic/LLVM.h" // isa, dyn_cast
#include "clang/Basic/Specifiers.h"
#include "clang/Basic/TokenKinds.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Overload.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaInternal.h"
#include "clang/Sema/Template.h"

#include "llvm/Support/SaveAndRestore.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>

using namespace clang;

namespace clad {

DerivativeBuilder::DerivativeBuilder(clang::Sema& S, plugin::CladPlugin& P,
                                     DerivedFnCollector& DFC,
                                     clad::DynamicGraph<DiffRequest>& G)
    : m_Sema(S), m_CladPlugin(P), m_Context(S.getASTContext()), m_DFC(DFC),
      m_DiffRequestGraph(G),
      m_NodeCloner(new utils::StmtClone(m_Sema, m_Context)),
      m_BuiltinDerivativesNSD(nullptr), m_NumericalDiffNSD(nullptr) {}

DerivativeBuilder::~DerivativeBuilder() {}

static void registerDerivative(Decl* D, Sema& S, const DiffRequest& R) {
  DeclContext* DC = D->getLexicalDeclContext();
  if (auto* dFD = dyn_cast<FunctionDecl>(D)) {
    LookupResult Previous(S, dFD->getNameInfo(), Sema::LookupOrdinaryName);
    // Template instantiations of function templates should not be considered
    // redeclarations.
    // FIXME: Currently we produce a FunctionDecl per instantiation, however, we
    // should follow closer what clang does, namely building a
    // FunctionTemplateDecl and then we should instantiate it with the
    // particular template parameters.
    if (R.Function && !R.Function->getPrimaryTemplate())
      S.LookupQualifiedName(Previous, dFD->getParent());

    // Check if we created a top-level decl with the same name for another
    // class.
    // FIXME: This case should be addressed by providing proper names and
    // function implementation that does not rely on accessing private data from
    // the class.
    bool IsBrokenDecl = isa<RecordDecl>(DC);
    if (!IsBrokenDecl) {
      S.CheckFunctionDeclaration(
          /*Scope=*/nullptr, dFD, Previous,
          /*IsMemberSpecialization=*/
          false
          /*DeclIsDefn*/
          CLAD_COMPAT_CheckFunctionDeclaration_DeclIsDefn_ExtraParam(dFD));
    } else if (R.DerivedFDPrototypes.size() >= R.CurrentDerivativeOrder) {
      // Size >= current derivative order means that there exists a declaration
      // or prototype for the currently derived function.
      dFD->setPreviousDecl(R.DerivedFDPrototypes[R.CurrentDerivativeOrder - 1]);
    }
  } else if (auto* dVD = dyn_cast<VarDecl>(D))
    // Add the identifier to the scope and IdResolver
    S.PushOnScopeChains(dVD, S.TUScope, /*AddToContext*/ false);

  if (D->isInvalidDecl())
    return; // CheckFunctionDeclaration was unhappy about derivedFD

  DC->addDecl(D);
}

  static bool hasAttribute(const Decl *D, attr::Kind Kind) {
    for (const auto *Attribute : D->attrs())
      if (Attribute->getKind() == Kind)
        return true;
    return false;
  }

  DeclWithContext DerivativeBuilder::cloneFunction(
      const clang::FunctionDecl* FD, clad::VisitorBase& VB,
      clang::DeclContext* DC, clang::SourceLocation& noLoc,
      clang::DeclarationNameInfo name, clang::QualType functionType) {
    FunctionDecl* returnedFD = nullptr;
    NamespaceDecl* enclosingNS = nullptr;
    TypeSourceInfo* TSI = m_Context.getTrivialTypeSourceInfo(functionType);
    if (isa<CXXMethodDecl>(FD)) {
      CXXRecordDecl* CXXRD = cast<CXXRecordDecl>(DC);
      // For constructor derivatives, `this` object is not provided.
      // Therefore, we need to make the derivative static.
      StorageClass SC = isa<CXXConstructorDecl>(FD)
                            ? SC_Static
                            : FD->getCanonicalDecl()->getStorageClass();
      returnedFD = CXXMethodDecl::Create(
          m_Context, CXXRD, noLoc, name, functionType, TSI,
          SC CLAD_COMPAT_FunctionDecl_UsesFPIntrin_Param(FD),
          FD->isInlineSpecified(), FD->getConstexprKind(), noLoc);
      // Generated member function should be called outside of class definitions
      // even if their original function had different access specifier.
      returnedFD->setAccess(AS_public);
    } else {
      assert (isa<FunctionDecl>(FD) && "Unexpected!");
      enclosingNS = VB.RebuildEnclosingNamespaces(DC);
      returnedFD = FunctionDecl::Create(
          m_Context, m_Sema.CurContext, noLoc, name, functionType, TSI,
          FD->getCanonicalDecl()->getStorageClass()
              CLAD_COMPAT_FunctionDecl_UsesFPIntrin_Param(FD),
          FD->isInlineSpecified(), FD->hasWrittenPrototype(),
          FD->getConstexprKind(),
          FD->getTrailingRequiresClause()
              ? VB.Clone(FD->getTrailingRequiresClause())
              : nullptr);

      returnedFD->setAccess(FD->getAccess());
    }

    returnedFD->setImplicitlyInline(FD->isInlined());

    for (const FunctionDecl* NFD : FD->redecls()) {
      for (const auto* Attr : NFD->attrs()) {
        // We only need the keywords final and override in the tag declaration.
        if (isa<OverrideAttr>(Attr) || isa<FinalAttr>(Attr))
          continue;
        if (!hasAttribute(returnedFD, Attr->getKind()))
          returnedFD->addAttr(Attr->clone(m_Context));
      }
    }

    return { returnedFD, enclosingNS };
  }

  // This method is derived from the source code of both
  // buildOverloadedCallSet() in SemaOverload.cpp and ActOnCallExpr() in
  // SemaExpr.cpp.
  bool
  DerivativeBuilder::noOverloadExists(Expr* UnresolvedLookup,
                                      llvm::MutableArrayRef<Expr*> ARargs) {
    auto NeedsMoreArgs = [](const FunctionDecl* FD, size_t Size) {
      return FD->getMinRequiredArguments() > Size || FD->getNumParams() < Size;
    };
    if (UnresolvedLookup->hasPlaceholderType(BuiltinType::BoundMember)) {
      // See Sema::BuildCallToMemberFunction.
      if (auto* ME = dyn_cast<MemberExpr>(UnresolvedLookup->IgnoreParens())) {
        auto* M = cast<CXXMethodDecl>(ME->getMemberDecl());
        return NeedsMoreArgs(M, ARargs.size());
      }
      return false;
    }
    if (UnresolvedLookup->hasPlaceholderType(BuiltinType::Overload)) {
      OverloadExpr::FindResult find = OverloadExpr::find(UnresolvedLookup);

      if (!find.HasFormOfMemberPointer) {
        OverloadExpr* ovl = find.Expression;

        if (isa<UnresolvedLookupExpr>(ovl)) {
          ExprResult result;
          SourceLocation Loc;
          OverloadCandidateSet CandidateSet(Loc,
                                            OverloadCandidateSet::CSK_Normal);
          Scope* S = m_Sema.getScopeForContext(m_Sema.CurContext);
          auto* ULE = cast<UnresolvedLookupExpr>(ovl);
          // Populate CandidateSet.
          m_Sema.buildOverloadedCallSet(S, UnresolvedLookup, ULE, ARargs, Loc,
                                        &CandidateSet, &result);
          OverloadCandidateSet::iterator Best = nullptr;
          OverloadingResult OverloadResult = CandidateSet.BestViableFunction(
              m_Sema, UnresolvedLookup->getBeginLoc(), Best);
          if (OverloadResult != 0U) // No overloads were found.
            return true;
        }
      }
      return false;
    }

    if (!isa<DeclRefExpr>(UnresolvedLookup))
      return false;

    const auto* DRE = cast<DeclRefExpr>(UnresolvedLookup);
    if (const auto* FD = dyn_cast<FunctionDecl>(DRE->getDecl()))
      return NeedsMoreArgs(FD, ARargs.size());

    return false;
  }

  LookupResult DerivativeBuilder::LookupCustomDerivativeOrNumericalDiff(
      const std::string& Name, const clang::DeclContext* originalFnDC,
      CXXScopeSpec& SS, bool forCustomDerv /*=true*/,
      bool namespaceShouldExist /*=true*/) {

    IdentifierInfo* II = &m_Context.Idents.get(Name);
    DeclarationName name(II);
    DeclarationNameInfo DNInfo(name, utils::GetValidSLoc(m_Sema));
    LookupResult R(m_Sema, DNInfo, Sema::LookupOrdinaryName);

    NamespaceDecl* NSD = nullptr;
    std::string namespaceID;
    if (forCustomDerv) {
      namespaceID = "custom_derivatives";
      NamespaceDecl* cladNS = nullptr;
      if (m_BuiltinDerivativesNSD)
        NSD = m_BuiltinDerivativesNSD;
      else {
        cladNS = utils::LookupNSD(m_Sema, "clad", /*shouldExist=*/true);
        NSD =
            utils::LookupNSD(m_Sema, namespaceID, namespaceShouldExist, cladNS);
        m_BuiltinDerivativesNSD = NSD;
      }
    } else {
      NSD = m_NumericalDiffNSD;
      namespaceID = "numerical_diff";
    }
    if (!NSD) {
      NSD = utils::LookupNSD(m_Sema, namespaceID, namespaceShouldExist);
      if (!NSD)
        return R;
    }
    DeclContext* DC = NSD;

    // FIXME: Here `if` branch should be removed once we update
    // numerical diff to use correct declaration context.
    if (forCustomDerv) {
      // FIXME: We should ideally construct nested name specifier from the
      // found custom derivative function. Current way will compute incorrect
      // nested name specifier in some cases.
      if (isa<RecordDecl>(originalFnDC))
        DC = utils::LookupNSD(m_Sema, "class_functions",
                              /*shouldExist=*/false, NSD);
      else
        DC = utils::FindDeclContext(m_Sema, NSD, originalFnDC);
      if (DC)
        utils::BuildNNS(m_Sema, DC, SS);
    } else {
      SS.Extend(m_Context, NSD, noLoc, noLoc);
    }
    if (DC)
      m_Sema.LookupQualifiedName(R, DC);

    if (R.empty())
      SS.clear();

    return R;
  }

  FunctionDecl* DerivativeBuilder::LookupCustomDerivativeDecl(
      const std::string& Name, const clang::DeclContext* originalFnDC,
      QualType functionType) {
    CXXScopeSpec SS;
    LookupResult R =
        LookupCustomDerivativeOrNumericalDiff(Name, originalFnDC, SS);

    for (NamedDecl* ND : R)
      if (auto* FD = dyn_cast<FunctionDecl>(ND))
        // Check if FD and functionType have the same signature.
        if (utils::SameCanonicalType(FD->getType(), functionType))
          // Make sure that it is not the case that FD is the forward
          // declaration generated by Clad. It should be user defined custom
          // derivative (either within the same translation unit or linked in
          // from another translation unit).
          if (FD->isDefined() || !m_DFC.IsCladDerivative(FD)) {
            m_DFC.AddToCustomDerivativeSet(FD);
            return FD;
          }

    return nullptr;
  }

  Expr* DerivativeBuilder::BuildCallToCustomDerivativeOrNumericalDiff(
      const std::string& Name, llvm::SmallVectorImpl<Expr*>& CallArgs,
      clang::Scope* S, const clang::Expr* callSite,
      bool forCustomDerv /*=true*/, bool namespaceShouldExist /*=true*/,
      Expr* CUDAExecConfig /*=nullptr*/) {
    DeclContext* originalFnDC = nullptr;

    // FIXME: callSite must not be null but it comes when we try to build
    // a numerical diff call. We should merge both paths and remove the
    // special branches being taken for propagators and numerical diff.
    if (callSite) {
      // Check if the callSite is not associated with a shadow declaration.
      if (const auto* ME = dyn_cast<CXXMemberCallExpr>(callSite)) {
        originalFnDC = ME->getMethodDecl()->getParent();
      } else if (const auto* CE = dyn_cast<CallExpr>(callSite)) {
        const Expr* Callee = CE->getCallee()->IgnoreParenCasts();
        if (const auto* DRE = dyn_cast<DeclRefExpr>(Callee))
          originalFnDC =
              const_cast<DeclContext*>(DRE->getFoundDecl()->getDeclContext());
        else if (const auto* MemberE = dyn_cast<MemberExpr>(Callee))
          originalFnDC = MemberE->getFoundDecl().getDecl()->getDeclContext();
      } else if (const auto* CtorExpr = dyn_cast<CXXConstructExpr>(callSite)) {
        originalFnDC = CtorExpr->getConstructor()->getDeclContext();
      }
    }

    CXXScopeSpec SS;
    LookupResult R = LookupCustomDerivativeOrNumericalDiff(
        Name, originalFnDC, SS, forCustomDerv, namespaceShouldExist);
    bool hasUserDefinedPropagator = false;
    if (!R.empty()) {
      hasUserDefinedPropagator = true;
#ifndef NDEBUG
      // FIXME: Figure out how to assert here in cases where we have provided
      // both a clad-generated derivative and a user-provided one.
      // LookupResult R1 = utils::LookupQualifiedName(Name, m_Sema,
      // originalFnDC); assert((R1.empty() || R1.getFoundDecl() ==
      // R.getFoundDecl()) &&
      //        "We clad built a derivative for entity which"
      //        "has a custom derivative!");
#endif // NDEBUG
    } else {
      // Try to find if clad already built a derivative.
      R = utils::LookupQualifiedName(Name, m_Sema, originalFnDC);
      if (originalFnDC && !originalFnDC->isRecord())
        utils::BuildNNS(m_Sema, originalFnDC, SS);
    }

    Expr* OverloadedFn = nullptr;
    if (!R.empty()) {
      auto MARargs = llvm::MutableArrayRef<Expr*>(CallArgs);
      SourceLocation Loc;

      if (forCustomDerv && (isa<CXXMemberCallExpr>(callSite) ||
                            isa<CXXOperatorCallExpr>(callSite))) {
        if (R.getNamingClass()) {
          Expr* Base = CallArgs[0];
          // if (Loc.isInvalid())
          //   Loc = m_DiffReq->getLocation();
          UnqualifiedId Member;
          Member.setIdentifier(&m_Context.Idents.get(Name), Loc);
          bool isArrow = Base->getType()->isPointerType();
          // FIXME: update SS here?
          auto* ME =
              m_Sema
                  .ActOnMemberAccessExpr(S, Base, Loc,
                                         isArrow ? tok::TokenKind::arrow
                                                 : tok::TokenKind::period,
                                         SS, noLoc, Member,
                                         /*ObjCImpDecl=*/nullptr)
                  .get();
          if (noOverloadExists(ME, MARargs.drop_front()))
            return nullptr;

          return m_Sema
              .ActOnCallExpr(S, ME, Loc, MARargs.drop_front(), Loc,
                             CUDAExecConfig)
              .get();
        }
        // If we have user-defined propagators we pass the first argument by
        // pointer.
        if (hasUserDefinedPropagator &&
            !CallArgs[0]->getType()->isPointerType())
          CallArgs[0] =
              m_Sema.BuildUnaryOp(S, noLoc, UO_AddrOf, CallArgs[0]).get();
      }
      Expr* UnresolvedLookup =
          m_Sema.BuildDeclarationNameExpr(SS, R, /*ADL*/ false).get();

      if (noOverloadExists(UnresolvedLookup, MARargs))
        return nullptr;

      OverloadedFn = m_Sema
                         .ActOnCallExpr(S, UnresolvedLookup, Loc, MARargs, Loc,
                                        CUDAExecConfig)
                         .get();

      // Add the custom derivative to the set of derivatives.
      // This is required in case the definition of the custom derivative
      // is not found in the current translation unit and is linked in
      // from another translation unit.
      // Adding it to the set of derivatives ensures that the custom
      // derivative is not differentiated again using numerical
      // differentiation due to unavailable definition.
      if (auto* CE = dyn_cast_or_null<CallExpr>(OverloadedFn))
        if (FunctionDecl* FD = CE->getDirectCallee())
          m_DFC.AddToCustomDerivativeSet(FD);
    }
    return OverloadedFn;
  }

  clang::FunctionDecl*
  DerivativeBuilder::HandleNestedDiffRequest(DiffRequest& request) {
    // FIXME: Find a way to do this without accessing plugin namespace functions
    bool alreadyDerived = true;
    FunctionDecl* derivative = this->FindDerivedFunction(request);
    if (!derivative) {
      alreadyDerived = false;

      {
        // Store and restore the original function and its order.
        llvm::SaveAndRestore<const FunctionDecl*> origFn(request.Function);
        llvm::SaveAndRestore<unsigned> origFnOrder(
            request.CurrentDerivativeOrder);

        // Derive declaration of the the forward mode derivative.
        request.DeclarationOnly = true;
        derivative = plugin::ProcessDiffRequest(m_CladPlugin, request);
      }

      // It is possible that user has provided a custom derivative for the
      // derivative function. In that case, we should not derive the definition
      // again.
      if (derivative &&
          (derivative->isDefined() || m_DFC.IsCustomDerivative(derivative)))
        alreadyDerived = true;

      // Add the request to derive the definition of the forward mode derivative
      // to the schedule.
      request.DeclarationOnly = false;
    }
    this->AddEdgeToGraph(request, alreadyDerived);
    return derivative;
  }

  void DerivativeBuilder::AddErrorEstimationModel(
      std::unique_ptr<FPErrorEstimationModel> estModel) {
    m_EstModel.push_back(std::move(estModel));
  }

  void InitErrorEstimation(
      llvm::SmallVectorImpl<std::unique_ptr<ErrorEstimationHandler>>& handler,
      llvm::SmallVectorImpl<std::unique_ptr<FPErrorEstimationModel>>& model,
      DerivativeBuilder& builder, const DiffRequest& request) {
    // Set the handler.
    std::unique_ptr<ErrorEstimationHandler> pHandler(
        new ErrorEstimationHandler());
    handler.push_back(std::move(pHandler));
    // Set error estimation model. If no custom model provided by user,
    // use the built in Taylor approximation model.
    if (model.size() != handler.size()) {
      std::unique_ptr<FPErrorEstimationModel> pModel(
          new TaylorApprox(builder, request));
      model.push_back(std::move(pModel));
    }
    handler.back()->SetErrorEstimationModel(model.back().get());
  }

  void CleanupErrorEstimation(
      llvm::SmallVectorImpl<std::unique_ptr<ErrorEstimationHandler>>& handler,
      llvm::SmallVectorImpl<std::unique_ptr<FPErrorEstimationModel>>& model) {
    model.back()->clearEstimationVariables();
    model.pop_back();
    handler.pop_back();
  }

  DerivativeAndOverload
  DerivativeBuilder::Derive(const DiffRequest& request) {
    TimedGenerationRegion G([&request]() { return (std::string)request; });
    if (const FunctionDecl* FD = request.Function) {
      // Perform diagnostics for functions
      // If FD is only a declaration, try to find its definition.
      if (!FD->getDefinition()) {
        // If only declaration is requested, allow this for clad-generated
        // functions or custom derivatives.
        if (!request.DeclarationOnly ||
            !(m_DFC.IsCladDerivative(FD) || m_DFC.IsCustomDerivative(FD))) {
          if (request.VerboseDiags)
            diag(DiagnosticsEngine::Error,
                 request.CallContext ? request.CallContext->getBeginLoc()
                                     : noLoc,
                 "attempted differentiation of function '%0', which does not "
                 "have a "
                 "definition",
                 {FD->getNameAsString()});
          return {};
        }
      }

      if (!request.DeclarationOnly)
        FD = FD->getDefinition();

      // check if the function is non-differentiable.
      if (clad::utils::hasNonDifferentiableAttribute(FD)) {
        diag(DiagnosticsEngine::Error,
             request.CallContext ? request.CallContext->getBeginLoc() : noLoc,
             "attempted differentiation of function '%0', which is marked as "
             "non-differentiable",
             {FD->getNameAsString()});
        return {};
      }

      // If the function is a method of a class, check if the class is
      // non-differentiable.
      if (const CXXMethodDecl* MD = dyn_cast<CXXMethodDecl>(FD)) {
        const CXXRecordDecl* CD = MD->getParent();
        if (clad::utils::hasNonDifferentiableAttribute(CD)) {
          diag(DiagnosticsEngine::Error, MD->getLocation(),
               "attempted differentiation of method '%0' in class '%1', which "
               "is "
               "marked as "
               "non-differentiable",
               {MD->getNameAsString(), CD->getNameAsString()});
          return {};
        }
      }
    } else if (const VarDecl* VD = request.Global) {
      // Warn the user about the usage of global variables.
      auto diagId = m_Sema.Diags.getCustomDiagID(
          DiagnosticsEngine::Warning,
          "The gradient utilizes a global variable '%0'"
          ". Please make sure to properly reset '%0' before re-running "
          "the gradient.");
      m_Sema.Diag(VD->getLocation(), diagId) << VD->getName();
    }

    DerivativeAndOverload result{};
    if (request.Mode == DiffMode::forward) {
      BaseForwardModeVisitor V(*this, request);
      result = V.Derive();
    } else if (request.Mode == DiffMode::pushforward) {
      PushForwardModeVisitor V(*this, request);
      result = V.Derive();
    } else if (request.Mode == DiffMode::vector_forward_mode) {
      VectorForwardModeVisitor V(*this, request);
      result = V.Derive();
    } else if (request.Mode == DiffMode::vector_pushforward) {
      VectorPushForwardModeVisitor V(*this, request);
      result = V.Derive();
    } else if (request.Mode == DiffMode::reverse) {
      ReverseModeVisitor V(*this, request);
      result = V.Derive();
    } else if (request.Mode == DiffMode::pullback) {
      ReverseModeVisitor V(*this, request);
      if (!m_ErrorEstHandler.empty()) {
        InitErrorEstimation(m_ErrorEstHandler, m_EstModel, *this, request);
        V.AddExternalSource(*m_ErrorEstHandler.back());
      }
      result = V.Derive();
      if (!m_ErrorEstHandler.empty())
        CleanupErrorEstimation(m_ErrorEstHandler, m_EstModel);
    } else if (request.Mode == DiffMode::reverse_mode_forward_pass) {
      ReverseModeForwPassVisitor V(*this, request);
      result = V.Derive();
    } else if (request.Mode == DiffMode::hessian ||
               request.Mode == DiffMode::hessian_diagonal) {
      HessianModeVisitor H(*this, request);
      result = H.Derive();
    } else if (request.Mode == DiffMode::jacobian) {
      JacobianModeVisitor J(*this, request);
      result = J.Derive();
    } else if (request.Mode == DiffMode::error_estimation) {
      ReverseModeVisitor R(*this, request);
      InitErrorEstimation(m_ErrorEstHandler, m_EstModel, *this, request);
      R.AddExternalSource(*m_ErrorEstHandler.back());
      // Finally begin estimation.
      result = R.Derive();
      // Once we are done, we want to clear the model for any further
      // calls to estimate_error.
      CleanupErrorEstimation(m_ErrorEstHandler, m_EstModel);
    } else if (const VarDecl* VD = request.Global) {
      // The request represents a global variable, construct the adjoint and
      // register it.
      QualType type = VD->getType();
      // add namespace specifier in variable declaration if needed.
      type = utils::AddNamespaceSpecifier(m_Sema, m_Context, type);
      IdentifierInfo* II = &m_Context.Idents.get("_d_" + VD->getNameAsString());
      auto* DC = const_cast<DeclContext*>(VD->getDeclContext());
      auto* VDDiff =
          VarDecl::Create(m_Context, DC, VD->getLocation(), VD->getLocation(),
                          II, type, /*TSI=*/nullptr, SC_None);
      m_Sema.AddInitializerToDecl(VDDiff, utils::getZeroInit(type, m_Sema),
                                  /*DirectInit=*/false);
      m_Sema.FinalizeDeclaration(VDDiff);
      result = VDDiff;
    }

    // FIXME: if the derivatives aren't registered in this order and the
    //   derivative is a member function it goes into an infinite loop
    bool isCustomDerivative = false;
    if (auto* FD = dyn_cast_or_null<FunctionDecl>(result.derivative))
      isCustomDerivative = m_DFC.IsCustomDerivative(FD);
    if (!isCustomDerivative) {
      if (auto* FD = result.derivative)
        registerDerivative(FD, m_Sema, request);
      if (auto* OFD = result.overload)
        registerDerivative(OFD, m_Sema, request);
    }

    return result;
  }

  FunctionDecl*
  DerivativeBuilder::FindDerivedFunction(const DiffRequest& request) {
    auto DFI = m_DFC.Find(request);
    if (DFI.IsValid())
      return DFI.DerivedFn();
    return nullptr;
  }

  void DerivativeBuilder::AddEdgeToGraph(const DiffRequest& request,
                                         bool alreadyDerived /*=false*/) {
    m_DiffRequestGraph.addEdgeToCurrentNode(request, alreadyDerived);
  }
  } // end namespace clad
