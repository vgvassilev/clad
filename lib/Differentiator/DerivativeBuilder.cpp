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
#include "clad/Differentiator/ForwardModeVisitor.h"
#include "clad/Differentiator/HessianModeVisitor.h"
#include "clad/Differentiator/JacobianModeVisitor.h"
#include "clad/Differentiator/ReverseModeVisitor.h"
#include "clad/Differentiator/StmtClone.h"
#include "clad/Differentiator/VectorForwardModeVisitor.h"

#include <algorithm>

#include "clad/Differentiator/CladUtils.h"
#include "clad/Differentiator/Compatibility.h"

using namespace clang;

namespace clad {

  DerivativeBuilder::DerivativeBuilder(clang::Sema& S, plugin::CladPlugin& P)
    : m_Sema(S), m_CladPlugin(P), m_Context(S.getASTContext()),
      m_NodeCloner(new utils::StmtClone(m_Sema, m_Context)),
      m_BuiltinDerivativesNSD(nullptr), m_NumericalDiffNSD(nullptr) {}

  DerivativeBuilder::~DerivativeBuilder() {}

  static void registerDerivative(FunctionDecl* derivedFD, Sema& semaRef) {
    LookupResult R(semaRef, derivedFD->getNameInfo(), Sema::LookupOrdinaryName);
    // FIXME: Attach out-of-line virtual function definitions to the TUScope.
    Scope* S = semaRef.getScopeForContext(derivedFD->getDeclContext());
    semaRef.CheckFunctionDeclaration(S, derivedFD, R,
                                     /*IsMemberSpecialization=*/false
                                     /*DeclIsDefn*/CLAD_COMPAT_CheckFunctionDeclaration_DeclIsDefn_ExtraParam(derivedFD));

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

          if (derivedFD->getASTContext()
                  .hasSameFunctionTypeIgnoringExceptionSpec(derivedFD
                                                                ->getType(),
                                                            FD->getType())) {
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
                                         FD->getStorageClass()
                                         CLAD_COMPAT_FunctionDecl_UsesFPIntrin_Param(FD),
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
                                        FD->getStorageClass()
                                        CLAD_COMPAT_FunctionDecl_UsesFPIntrin_Param(FD),
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

  void DerivativeBuilder::AddErrorEstimationModel(
      std::unique_ptr<FPErrorEstimationModel> estModel) {
    m_EstModel.push_back(std::move(estModel));
  }

  void InitErrorEstimation(
      llvm::SmallVectorImpl<std::unique_ptr<ErrorEstimationHandler>>& handler,
      llvm::SmallVectorImpl<std::unique_ptr<FPErrorEstimationModel>>& model,
      DerivativeBuilder& builder) {
    // Set the handler.
    std::unique_ptr<ErrorEstimationHandler> pHandler(
        new ErrorEstimationHandler());
    handler.push_back(std::move(pHandler));
    // Set error estimation model. If no custom model provided by user,
    // use the built in Taylor approximation model.
    if (model.size() != handler.size()) {
      std::unique_ptr<FPErrorEstimationModel> pModel(new TaylorApprox(builder));
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
      if (request.VerboseDiags)
        diag(DiagnosticsEngine::Error,
             request.CallContext ? request.CallContext->getBeginLoc() : noLoc,
             "attempted differentiation of function '%0', which does not have a "
             "definition", { FD->getNameAsString() });
      return {};
    }

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
      BaseForwardModeVisitor V(*this);
      result = V.Derive(FD, request);
    } else if (request.Mode == DiffMode::experimental_pushforward) {
      ForwardModeVisitor V(*this);
      result = V.DerivePushforward(FD, request);
    } else if (request.Mode == DiffMode::vector_forward_mode) {
      VectorForwardModeVisitor V(*this);
      result = V.DeriveVectorMode(FD, request);
    } else if (request.Mode == DiffMode::reverse) {
      ReverseModeVisitor V(*this);
      result = V.Derive(FD, request);
    } else if (request.Mode == DiffMode::experimental_pullback) {
      ReverseModeVisitor V(*this);
      if (!m_ErrorEstHandler.empty()) {
        InitErrorEstimation(m_ErrorEstHandler, m_EstModel, *this);
        V.AddExternalSource(*m_ErrorEstHandler.back());
      }
      result = V.DerivePullback(FD, request);
      if (!m_ErrorEstHandler.empty())
        CleanupErrorEstimation(m_ErrorEstHandler, m_EstModel);
    } else if (request.Mode == DiffMode::hessian) {
      HessianModeVisitor H(*this);
      result = H.Derive(FD, request);
    } else if (request.Mode == DiffMode::jacobian) {
      JacobianModeVisitor J(*this);
      result = J.Derive(FD, request);
    } else if (request.Mode == DiffMode::error_estimation) {
      ReverseModeVisitor R(*this);
      InitErrorEstimation(m_ErrorEstHandler, m_EstModel, *this);
      R.AddExternalSource(*m_ErrorEstHandler.back());
      // Finally begin estimation.
      result = R.Derive(FD, request);
      // Once we are done, we want to clear the model for any further
      // calls to estimate_error.
      CleanupErrorEstimation(m_ErrorEstHandler, m_EstModel);
    }

    // FIXME: if the derivatives aren't registered in this order and the
    //   derivative is a member function it goes into an infinite loop
    if (auto FD = result.derivative)
      registerDerivative(FD, m_Sema);
    if (auto OFD = result.overload)
      registerDerivative(OFD, m_Sema);

    return result;
  }
}// end namespace clad
