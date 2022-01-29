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

  std::unique_ptr<ErrorEstimationHandler> errorEstHandler = nullptr;

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
                                     /*IsMemberSpecialization=*/false);

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

  static FunctionDecl* getOriginalFD(OverloadedDeclWithContext& ODWC) {
    return std::get<0>(ODWC);
  }

  static FunctionDecl* getOverloadFD(OverloadedDeclWithContext& ODWC) {
    return std::get<2>(ODWC);
  }

  void DerivativeBuilder::SetErrorEstimationModel(
      std::unique_ptr<FPErrorEstimationModel> estModel) {
    m_EstModel = std::move(estModel);
  }

  OverloadedDeclWithContext
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
    OverloadedDeclWithContext result{};
    if (request.Mode == DiffMode::forward) {
      ForwardModeVisitor V(*this);
      result = V.Derive(FD, request);
    } else if (request.Mode == DiffMode::experimental_pushforward) {
      ForwardModeVisitor V(*this);
      result = V.DerivePushforward(FD, request);
    } else if (request.Mode == DiffMode::reverse) {
      ReverseModeVisitor V(*this);
      result = V.Derive(FD, request);
    } else if (request.Mode == DiffMode::hessian) {
      HessianModeVisitor H(*this);
      result = H.Derive(FD, request);
    } else if (request.Mode == DiffMode::jacobian) {
      JacobianModeVisitor J(*this);
      result = J.Derive(FD, request);
    } else if (request.Mode == DiffMode::error_estimation) {
      // Set the handler.
      errorEstHandler.reset(new ErrorEstimationHandler(*this));
      // Set error estimation model. If no custom model provided by user,
      // use the built in Taylor approximation model.
      if (!m_EstModel) {
        m_EstModel.reset(new TaylorApprox(*this));
      }
      errorEstHandler->SetErrorEstimationModel(m_EstModel.get());
      // Finally begin estimation.
      result = errorEstHandler->Derive(FD, request);
      // Once we are done, we want to clear the model for any further
      // calls to estimate_error.
      m_EstModel->clearEstimationVariables();
    }

    // FIXME: if the derivatives aren't registered in this order and the
    //   derivative is a member function it goes into an infinite loop
    if (auto FD = getOriginalFD(result))
      registerDerivative(FD, m_Sema);
    if (auto OFD = getOverloadFD(result))
      registerDerivative(OFD, m_Sema);

    return result;
  }
}// end namespace clad
