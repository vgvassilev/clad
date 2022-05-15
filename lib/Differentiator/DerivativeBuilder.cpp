//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// version: $Id: ClangPlugin.cpp 7 2013-06-01 22:48:03Z v.g.vassilev@gmail.com $
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#include "clad/Differentiator/DerivativeBuilder.h"

#include "clad/Differentiator/ErrorEstimator.h"
#include "clad/Differentiator/ForwardModeVisitor.h"
#include "clad/Differentiator/HessianModeVisitor.h"
#include "clad/Differentiator/JacobianModeVisitor.h"
#include "clad/Differentiator/ReverseModeVisitor.h"

#include "clad/Differentiator/DiffPlanner.h"
#include "clad/Differentiator/StmtClone.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/TemplateBase.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Overload.h"
#include "clang/Sema/SemaInternal.h"
#include "clang/Sema/Template.h"

#include <algorithm>

#include "clad/Differentiator/Compatibility.h"

using namespace clang;

namespace clad {

namespace plugin {
void DumpRequestedInfo(CladPlugin& P, const FunctionDecl* sourceFn,
                       const FunctionDecl* derivedFn);
void ProcessTopLevelDecl(CladPlugin& P, Decl* D);
}

  DerivativeBuilder::DerivativeBuilder(clang::Sema& S, plugin::CladPlugin& P)
    : m_Sema(S), m_CladPlugin(P), m_Context(S.getASTContext()),
      m_NodeCloner(new utils::StmtClone(m_Sema, m_Context)),
      m_BuiltinDerivativesNSD(nullptr), m_NumericalDiffNSD(nullptr), m_ErrorEstHandler(nullptr) {}

  DerivativeBuilder::~DerivativeBuilder() {}

  void DerivativeBuilder::RegisterFunction(FunctionDecl* FD) {
    LookupResult R(m_Sema, FD->getNameInfo(), Sema::LookupOrdinaryName);
    // FIXME: Attach out-of-line virtual function definitions to the TUScope.
    Scope* S = m_Sema.getScopeForContext(FD->getDeclContext());
    m_Sema.CheckFunctionDeclaration(S, FD, R,
                                    /*IsMemberSpecialization=*/false);

    // FIXME: Avoid the DeclContext lookup and the manual setPreviousDecl.
    // Consider out-of-line virtual functions.
    {
      DeclContext* LookupCtx = FD->getDeclContext();
      auto R = LookupCtx->noload_lookup(FD->getDeclName());

      for (NamedDecl* I : R) {
        if (auto* prevFD = dyn_cast<FunctionDecl>(I)) {
          // FIXME: We still do extra work in creating a derivative and throwing
          // it away.
          if (prevFD->getDefinition())
            return;

          if (FD->getASTContext().hasSameFunctionTypeIgnoringExceptionSpec(
                  FD->getType(), prevFD->getType())) {
            // Register the function on the redecl chain.
            FD->setPreviousDecl(prevFD);
            break;
          }
        }
      }
      // Inform the decl's decl context for its existance after the lookup,
      // otherwise it would end up in the LookupResult.
      FD->getDeclContext()->addDecl(FD);
      m_Sema.MarkFunctionReferenced(noLoc, FD);
      // CallHandleTopLevelDeclIfRequired(FD);
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

  DeclWithContext 
  DerivativeBuilder::cloneFunction(const clang::FunctionDecl* FD,
                                   clad::VisitorBase VD, 
                                   clang::DeclContext* DC,
                                   clang::Sema& m_Sema,
                                   clang::ASTContext& m_Context,
                                   clang::SourceLocation& noLoc,
                                   clang::DeclarationNameInfo name,
                                   clang::QualType functionType) {
    FunctionDecl* returnedFD = nullptr;
    NamespaceDecl* enclosingNS = nullptr;
    if (isa<CXXMethodDecl>(FD)) {
      CXXRecordDecl* CXXRD = cast<CXXRecordDecl>(DC);
      returnedFD = CXXMethodDecl::Create(m_Context, 
                                         CXXRD, 
                                         noLoc, 
                                         name,
                                         functionType, 
                                         FD->getTypeSourceInfo(),
                                         FD->getStorageClass(),
                                         FD->isInlineSpecified(),
                                         clad_compat::Function_GetConstexprKind
                                         (FD), noLoc);
      returnedFD->setAccess(FD->getAccess());
    } else {
      assert (isa<FunctionDecl>(FD) && "Unexpected!");
      enclosingNS = VD.RebuildEnclosingNamespaces(DC);
      returnedFD = FunctionDecl::Create(m_Context, 
                                        m_Sema.CurContext, 
                                        noLoc,
                                        name, 
                                        functionType,
                                        FD->getTypeSourceInfo(),
                                        FD->getStorageClass(),
                                        FD->isInlineSpecified(),
                                        FD->hasWrittenPrototype(),
                                        clad_compat::Function_GetConstexprKind(FD)CLAD_COMPAT_CLANG10_FunctionDecl_Create_ExtraParams(FD->getTrailingRequiresClause()));
    } 

    for (const FunctionDecl* NFD : FD->redecls())
      for (const auto* Attr : NFD->attrs())
        if (!hasAttribute(returnedFD, Attr->getKind()))
          returnedFD->addAttr(Attr->clone(m_Context));

    return { returnedFD, enclosingNS };
  }

  void DerivativeBuilder::SetErrorEstimationModel(
      std::unique_ptr<FPErrorEstimationModel> estModel) {
    m_EstModel = std::move(estModel);
  }

  DerivativeAndOverload
  DerivativeBuilder::Derive(const DiffRequest& request) {
    const FunctionDecl* FD = request.Function;
    //m_Sema.CurContext = m_Context.getTranslationUnitDecl();
    assert(FD && "Must not be null.");
    // If FD is only a declaration, try to find its definition.
    if (!FD->getDefinition()) {
      if (request.VerboseDiags)
        diag(DiagnosticsEngine::Error,
             request.CallContext ? request.CallContext->getBeginLoc() : noLoc,
             "attempted differentiation of function '%0', which does not have a "
             "definition", { FD->getNameAsString() });
      return {};
    }
    FD = FD->getDefinition();
    DerivativeAndOverload result{};
    if (request.Mode == DiffMode::forward) {
      ForwardModeVisitor V(*this);
      result = V.Derive(FD, request);
    } else if (request.Mode == DiffMode::experimental_pushforward) {
      ForwardModeVisitor V(*this);
      result = V.DerivePushforward(FD, request);
    } else if (request.Mode == DiffMode::reverse) {
      ReverseModeVisitor V(*this);
      result = V.Derive(FD, request);
    } else if (request.Mode == DiffMode::experimental_pullback) {
      ReverseModeVisitor V(*this);
      result = V.DerivePullback(FD, request);
    } else if (request.Mode == DiffMode::hessian) {
      HessianModeVisitor H(*this);
      result = H.Derive(FD, request);
    } else if (request.Mode == DiffMode::jacobian) {
      JacobianModeVisitor J(*this);
      result = J.Derive(FD, request);
    } else if (request.Mode == DiffMode::error_estimation) {
      ReverseModeVisitor R(*this);
      // Set the handler.
      m_ErrorEstHandler.reset(new ErrorEstimationHandler());
      // Set error estimation model. If no custom model provided by user,
      // use the built in Taylor approximation model.
      if (!m_EstModel) {
        m_EstModel.reset(new TaylorApprox(*this));
      }
      m_ErrorEstHandler->SetErrorEstimationModel(m_EstModel.get());
      R.AddExternalSource(*m_ErrorEstHandler);
      // Finally begin estimation.
      result = R.Derive(FD, request);
      // Once we are done, we want to clear the model for any further
      // calls to estimate_error.
      m_EstModel->clearEstimationVariables();
    }

    // FIXME: if the derivatives aren't registered in this order and the
    //   derivative is a member function it goes into an infinite loop
    if (auto FD = result.derivative)
      RegisterFunction(FD);
    if (auto OFD = result.overload)
      RegisterFunction(OFD);

    return result;
  }

  void DerivativeBuilder::CallHandleTopLevelDeclIfRequired(Decl* D) {
    if (!D)
      return;
    // We ideally should not call `HandleTopLevelDecl` for declarations
    // inside a namespace. After parsing a namespace that is defined
    // directly in translation unit context , clang calls
    // `BackendConsumer::HandleTopLevelDecl`.
    // `BackendConsumer::HandleTopLevelDecl` emits LLVM IR of each
    // declaration inside the namespace using CodeGen. We need to manually
    // call `HandleTopLevelDecl` for each new declaration added to a
    // namespace because `HandleTopLevelDecl` has already been called for
    // a namespace by Clang when the namespace is parsed.

    // Call CodeGen only if the produced Decl is a top-most
    // decl or is contained in a namespace decl.
    DeclContext* derivativeDC = D->getDeclContext();
    bool isTUorND =
        derivativeDC->isTranslationUnit() || derivativeDC->isNamespace();
    if (isTUorND) {
      plugin::ProcessTopLevelDecl(m_CladPlugin, D);
    }
  }

  FunctionDecl* DerivativeBuilder::ProcessDiffRequest(DiffRequest& request) {
    m_Sema.PerformPendingInstantiations();
    if (request.Function->getDefinition())
      request.Function = request.Function->getDefinition();
    request.UpdateDiffParamsInfo(m_Sema);
    const FunctionDecl* FD = request.Function;

    FunctionDecl* derivativeDecl = nullptr;
    FunctionDecl* overloadedDecl = nullptr;
    auto DFI = m_DFC.Find(request);

    if (DFI.IsValid()) {
      derivativeDecl = DFI.DerivedFn();
      overloadedDecl = DFI.OverloadedDerivedFn();
    } else {
      auto deriveRes = Derive(request);
      derivativeDecl = deriveRes.derivative;
      overloadedDecl = deriveRes.overload;
      // Differentiation successful, save differentiation information and dump
      // requested information.
      if (derivativeDecl) {
        m_DFC.Add(DerivedFnInfo(request, derivativeDecl, overloadedDecl));
        plugin::DumpRequestedInfo(m_CladPlugin, FD, derivativeDecl);
      }
      CallHandleTopLevelDeclIfRequired(derivativeDecl);
      CallHandleTopLevelDeclIfRequired(overloadedDecl);
    }

    // `*Visitor` classes in-charge of differentiation are responsible for
    // giving error messages if differentiation fails.
    if (!derivativeDecl)
      return nullptr;

    bool isLastDerivativeOrder =
        (request.CurrentDerivativeOrder == request.RequestedDerivativeOrder);

    // If this is the last required derivative order, replace the function
    // inside a call to clad::differentiate/gradient with its derivative.
    if (request.CallUpdateRequired && isLastDerivativeOrder)
      request.updateCall(derivativeDecl, overloadedDecl, m_Sema);

    // Last requested order was computed, return the result.
    if (isLastDerivativeOrder)
      return derivativeDecl;

    // If higher order derivatives are required, proceed to compute them
    // recursively.
    request.Function = derivativeDecl;
    request.CurrentDerivativeOrder += 1;
    return ProcessDiffRequest(request);
  }
}// end namespace clad
