//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// version: $Id: ClangPlugin.cpp 7 2013-06-01 22:48:03Z v.g.vassilev@gmail.com $
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#include "clad/Differentiator/DerivativeBuilder.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/TemplateBase.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Overload.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaInternal.h"
#include "clang/Sema/Template.h"
#include "clad/Differentiator/BaseForwardModeVisitor.h"
#include "clad/Differentiator/CladUtils.h"
#include "clad/Differentiator/DiffPlanner.h"
#include "clad/Differentiator/ErrorEstimator.h"
#include "clad/Differentiator/HessianModeVisitor.h"
#include "clad/Differentiator/PushForwardModeVisitor.h"
#include "clad/Differentiator/ReverseModeForwPassVisitor.h"
#include "clad/Differentiator/ReverseModeVisitor.h"
#include "clad/Differentiator/StmtClone.h"
#include "clad/Differentiator/VectorForwardModeVisitor.h"
#include "clad/Differentiator/VectorPushForwardModeVisitor.h"

#include "llvm/Support/SaveAndRestore.h"

#include <algorithm>

#include "clad/Differentiator/CladUtils.h"
#include "clad/Differentiator/Compatibility.h"

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

static void registerDerivative(FunctionDecl* derivedFD, Sema& semaRef) {
  LookupResult R(semaRef, derivedFD->getNameInfo(), Sema::LookupOrdinaryName);
  // FIXME: Attach out-of-line virtual function definitions to the TUScope.
  Scope* S = semaRef.getScopeForContext(derivedFD->getDeclContext());
  semaRef.CheckFunctionDeclaration(
      S, derivedFD, R,
      /*IsMemberSpecialization=*/
      false
      /*DeclIsDefn*/ CLAD_COMPAT_CheckFunctionDeclaration_DeclIsDefn_ExtraParam(
          derivedFD));

  // FIXME: Avoid the DeclContext lookup and the manual setPreviousDecl.
  // Consider out-of-line virtual functions.
  {
    DeclContext* LookupCtx = derivedFD->getDeclContext();
    auto R = LookupCtx->noload_lookup(derivedFD->getDeclName());

    for (NamedDecl* I : R) {
      if (auto* FD = dyn_cast<FunctionDecl>(I)) {
        // FIXME: We still do extra work in creating a derivative and throwing
        // it away.
        if (FD->getDefinition())
          return;

        if (derivedFD->getASTContext().hasSameFunctionTypeIgnoringExceptionSpec(
                derivedFD->getType(), FD->getType())) {
          // Register the function on the redecl chain.
          derivedFD->setPreviousDecl(FD);
          break;
        }
      }
    }
    // Inform the decl's decl context for its existance after the lookup,
    // otherwise it would end up in the LookupResult.
    derivedFD->getDeclContext()->addDecl(derivedFD);

    // FIXME: Rebuild VTable to remove requirements for "forward" declared
    // virtual methods
  }
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
    if (isa<CXXMethodDecl>(FD)) {
      CXXRecordDecl* CXXRD = cast<CXXRecordDecl>(DC);
      returnedFD = CXXMethodDecl::Create(
          m_Context, CXXRD, noLoc, name, functionType, FD->getTypeSourceInfo(),
          FD->getCanonicalDecl()->getStorageClass()
              CLAD_COMPAT_FunctionDecl_UsesFPIntrin_Param(FD),
          FD->isInlineSpecified(), clad_compat::Function_GetConstexprKind(FD),
          noLoc);
      returnedFD->setAccess(FD->getAccess());
    } else {
      assert (isa<FunctionDecl>(FD) && "Unexpected!");
      enclosingNS = VB.RebuildEnclosingNamespaces(DC);
      returnedFD = FunctionDecl::Create(
          m_Context, m_Sema.CurContext, noLoc, name, functionType,
          FD->getTypeSourceInfo(),
          FD->getCanonicalDecl()->getStorageClass()
              CLAD_COMPAT_FunctionDecl_UsesFPIntrin_Param(FD),
          FD->isInlineSpecified(), FD->hasWrittenPrototype(),
          clad_compat::Function_GetConstexprKind(FD)
              CLAD_COMPAT_CLANG10_FunctionDecl_Create_ExtraParams(
                  FD->getTrailingRequiresClause()));
    } 

    for (const FunctionDecl* NFD : FD->redecls())
      for (const auto* Attr : NFD->attrs())
        if (!hasAttribute(returnedFD, Attr->getKind()))
          returnedFD->addAttr(Attr->clone(m_Context));

    return { returnedFD, enclosingNS };
  }

  // This method is derived from the source code of both
  // buildOverloadedCallSet() in SemaOverload.cpp
  // and ActOnCallExpr() in SemaExpr.cpp.
  bool
  DerivativeBuilder::noOverloadExists(Expr* UnresolvedLookup,
                                      llvm::MutableArrayRef<Expr*> ARargs) {
    if (UnresolvedLookup->getType() == m_Context.OverloadTy) {
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
    }
    return false;
  }

  LookupResult DerivativeBuilder::LookupCustomDerivativeOrNumericalDiff(
      const std::string& Name, clang::DeclContext* originalFnDC,
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
    return R;
  }

  FunctionDecl* DerivativeBuilder::LookupCustomDerivativeDecl(
      const std::string& Name, clang::DeclContext* originalFnDC,
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
      clang::Scope* S, clang::DeclContext* originalFnDC,
      bool forCustomDerv /*=true*/, bool namespaceShouldExist /*=true*/) {

    CXXScopeSpec SS;
    LookupResult R = LookupCustomDerivativeOrNumericalDiff(
        Name, originalFnDC, SS, forCustomDerv, namespaceShouldExist);

    Expr* OverloadedFn = nullptr;
    if (!R.empty()) {
      // FIXME: We should find a way to specify nested name specifier
      // after finding the custom derivative.
      Expr* UnresolvedLookup =
          m_Sema.BuildDeclarationNameExpr(SS, R, /*ADL*/ false).get();

      auto MARargs = llvm::MutableArrayRef<Expr*>(CallArgs);

      SourceLocation Loc;

      if (noOverloadExists(UnresolvedLookup, MARargs))
        return nullptr;

      OverloadedFn =
          m_Sema.ActOnCallExpr(S, UnresolvedLookup, Loc, MARargs, Loc).get();

      // Add the custom derivative to the set of derivatives.
      // This is required in case the definition of the custom derivative
      // is not found in the current translation unit and is linked in
      // from another translation unit.
      // Adding it to the set of derivatives ensures that the custom
      // derivative is not differentiated again using numerical
      // differentiation due to unavailable definition.
      if (auto* CE = dyn_cast<CallExpr>(OverloadedFn))
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
    const FunctionDecl* FD = request.Function;
    //m_Sema.CurContext = m_Context.getTranslationUnitDecl();
    assert(FD && "Must not be null.");
    // If FD is only a declaration, try to find its definition.
    if (!FD->getDefinition()) {
      // If only declaration is requested, allow this for clad-generated
      // functions or custom derivatives.
      if (!request.DeclarationOnly ||
          !(m_DFC.IsCladDerivative(FD) || m_DFC.IsCustomDerivative(FD))) {
        if (request.VerboseDiags)
          diag(DiagnosticsEngine::Error,
               request.CallContext ? request.CallContext->getBeginLoc() : noLoc,
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
             "attempted differentiation of method '%0' in class '%1', which is "
             "marked as "
             "non-differentiable",
             {MD->getNameAsString(), CD->getNameAsString()});
        return {};
      }
    }

    DerivativeAndOverload result{};
    if (request.Mode == DiffMode::forward) {
      BaseForwardModeVisitor V(*this, request);
      result = V.Derive(FD, request);
    } else if (request.Mode == DiffMode::experimental_pushforward) {
      PushForwardModeVisitor V(*this, request);
      result = V.DerivePushforward(FD, request);
    } else if (request.Mode == DiffMode::vector_forward_mode) {
      VectorForwardModeVisitor V(*this, request);
      result = V.DeriveVectorMode(FD, request);
    } else if (request.Mode == DiffMode::experimental_vector_pushforward) {
      VectorPushForwardModeVisitor V(*this, request);
      result = V.DerivePushforward(FD, request);
    } else if (request.Mode == DiffMode::reverse) {
      ReverseModeVisitor V(*this, request);
      result = V.Derive(FD, request);
    } else if (request.Mode == DiffMode::experimental_pullback) {
      ReverseModeVisitor V(*this, request);
      if (!m_ErrorEstHandler.empty()) {
        InitErrorEstimation(m_ErrorEstHandler, m_EstModel, *this, request);
        V.AddExternalSource(*m_ErrorEstHandler.back());
      }
      result = V.DerivePullback(FD, request);
      if (!m_ErrorEstHandler.empty())
        CleanupErrorEstimation(m_ErrorEstHandler, m_EstModel);
    } else if (request.Mode == DiffMode::reverse_mode_forward_pass) {
      ReverseModeForwPassVisitor V(*this, request);
      result = V.Derive(FD, request);
    } else if (request.Mode == DiffMode::hessian ||
               request.Mode == DiffMode::hessian_diagonal) {
      HessianModeVisitor H(*this, request);
      result = H.Derive(FD, request);
    } else if (request.Mode == DiffMode::jacobian) {
      ReverseModeVisitor R(*this, request);
      result = R.Derive(FD, request);
    } else if (request.Mode == DiffMode::error_estimation) {
      ReverseModeVisitor R(*this, request);
      InitErrorEstimation(m_ErrorEstHandler, m_EstModel, *this, request);
      R.AddExternalSource(*m_ErrorEstHandler.back());
      // Finally begin estimation.
      result = R.Derive(FD, request);
      // Once we are done, we want to clear the model for any further
      // calls to estimate_error.
      CleanupErrorEstimation(m_ErrorEstHandler, m_EstModel);
    }

    // FIXME: if the derivatives aren't registered in this order and the
    //   derivative is a member function it goes into an infinite loop
    if (!m_DFC.IsCustomDerivative(result.derivative)) {
      if (auto* FD = result.derivative)
        registerDerivative(FD, m_Sema);
      if (auto* OFD = result.overload)
        registerDerivative(OFD, m_Sema);
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
}// end namespace clad
