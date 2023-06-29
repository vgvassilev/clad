//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// version: $Id: ClangPlugin.cpp 7 2013-06-01 22:48:03Z v.g.vassilev@gmail.com $
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#include "clad/Differentiator/ForwardModeVisitor.h"
#include "clad/Differentiator/BaseForwardModeVisitor.h"

#include "clad/Differentiator/CladUtils.h"

#include "llvm/Support/SaveAndRestore.h"

using namespace clang;

namespace clad {
ForwardModeVisitor::ForwardModeVisitor(DerivativeBuilder& builder)
    : BaseForwardModeVisitor(builder) {}

ForwardModeVisitor::~ForwardModeVisitor() {}

clang::QualType ForwardModeVisitor::ComputePushforwardFnReturnType() {
  assert(m_Mode == DiffMode::experimental_pushforward);
  QualType originalFnRT = m_Function->getReturnType();
  if (originalFnRT->isVoidType())
    return m_Context.VoidTy;
  TemplateDecl* valueAndPushforward =
      LookupTemplateDeclInCladNamespace("ValueAndPushforward");
  assert(valueAndPushforward &&
         "clad::ValueAndPushforward template not found!!");
  QualType RT =
      InstantiateTemplate(valueAndPushforward, {originalFnRT, originalFnRT});
  return RT;
}

  DerivativeAndOverload
  ForwardModeVisitor::DerivePushforward(const FunctionDecl* FD,
                                        const DiffRequest& request) {
    m_Function = FD;
    m_Functor = request.Functor;
    m_DerivativeOrder = request.CurrentDerivativeOrder;
    m_Mode = DiffMode::experimental_pushforward;
    assert(!m_DerivativeInFlight &&
           "Doesn't support recursive diff. Use DiffPlan.");
    m_DerivativeInFlight = true;

    auto originalFnEffectiveName = utils::ComputeEffectiveFnName(m_Function);

    IdentifierInfo* derivedFnII =
        &m_Context.Idents.get(originalFnEffectiveName + "_pushforward");
    DeclarationNameInfo derivedFnName(derivedFnII, noLoc);
    llvm::SmallVector<QualType, 16> paramTypes, derivedParamTypes;

    // If we are differentiating an instance member function then
    // create a parameter type for the parameter that will represent the
    // derivative of `this` pointer with respect to the independent parameter.
    if (auto MD = dyn_cast<CXXMethodDecl>(FD)) {
      if (MD->isInstance()) {
        QualType thisType = clad_compat::CXXMethodDecl_getThisType(m_Sema, MD);
        derivedParamTypes.push_back(thisType);
      }
    }

    for (auto* PVD : m_Function->parameters()) {
      paramTypes.push_back(PVD->getType());

      if (BaseForwardModeVisitor::IsDifferentiableType(PVD->getType()))
        derivedParamTypes.push_back(PVD->getType());
    }

    paramTypes.insert(paramTypes.end(), derivedParamTypes.begin(),
                      derivedParamTypes.end());

    auto originalFnType = dyn_cast<FunctionProtoType>(m_Function->getType());
    QualType returnType = ComputePushforwardFnReturnType();
    QualType derivedFnType =
        m_Context.getFunctionType(returnType, paramTypes,
                                  originalFnType->getExtProtoInfo());
    llvm::SaveAndRestore<DeclContext*> saveContext(m_Sema.CurContext);
    llvm::SaveAndRestore<Scope*> saveScope(m_CurScope);
    DeclContext* DC = const_cast<DeclContext*>(m_Function->getDeclContext());
    m_Sema.CurContext = DC;

    DeclWithContext cloneFunctionResult =
        m_Builder.cloneFunction(m_Function, *this, DC, m_Sema, m_Context, noLoc,
                                derivedFnName, derivedFnType);
    m_Derivative = cloneFunctionResult.first;

    llvm::SmallVector<ParmVarDecl*, 16> params;
    llvm::SmallVector<ParmVarDecl*, 16> derivedParams;
    beginScope(Scope::FunctionPrototypeScope | Scope::FunctionDeclarationScope |
               Scope::DeclScope);
    m_Sema.PushFunctionScope();
    m_Sema.PushDeclContext(getCurrentScope(), m_Derivative);

    // If we are differentiating an instance member function then
    // create a parameter for representing derivative of
    // `this` pointer with respect to the independent parameter.
    if (auto MFD = dyn_cast<CXXMethodDecl>(FD)) {
      if (MFD->isInstance()) {
        auto thisType = clad_compat::CXXMethodDecl_getThisType(m_Sema, MFD);
        IdentifierInfo* derivedPVDII = CreateUniqueIdentifier("_d_this");
        auto derivedPVD = utils::BuildParmVarDecl(m_Sema, m_Sema.CurContext,
                                                  derivedPVDII, thisType);
        m_Sema.PushOnScopeChains(derivedPVD, getCurrentScope(),
                                 /*AddToContext=*/false);
        derivedParams.push_back(derivedPVD);
        m_ThisExprDerivative = BuildDeclRef(derivedPVD);
      }
    }

    std::size_t numParamsOriginalFn = m_Function->getNumParams();
    for (std::size_t i = 0; i < numParamsOriginalFn; ++i) {
      auto PVD = m_Function->getParamDecl(i);
      // Some of the special member functions created implicitly by compilers
      // have missing parameter identifier.
      bool identifierMissing = false;
      IdentifierInfo* PVDII = PVD->getIdentifier();
      if (!PVDII || PVDII->getLength() == 0) {
        PVDII = CreateUniqueIdentifier("param");
        identifierMissing = true;
      }
      auto newPVD = CloneParmVarDecl(PVD, PVDII,
                                     /*pushOnScopeChains=*/true,
                                     /*cloneDefaultArg=*/false);
      params.push_back(newPVD);

      if (identifierMissing)
        m_DeclReplacements[PVD] = newPVD;

      if (!BaseForwardModeVisitor::IsDifferentiableType(PVD->getType()))
        continue;
      auto derivedPVDName = "_d_" + std::string(PVDII->getName());
      IdentifierInfo* derivedPVDII = CreateUniqueIdentifier(derivedPVDName);
      auto derivedPVD = CloneParmVarDecl(PVD, derivedPVDII,
                                         /*pushOnScopeChains=*/true,
                                         /*cloneDefaultArg=*/false);
      derivedParams.push_back(derivedPVD);
      m_Variables[newPVD] = BuildDeclRef(derivedPVD);
    }

    params.insert(params.end(), derivedParams.begin(), derivedParams.end());
    m_Derivative->setParams(params);
    m_Derivative->setBody(nullptr);

    beginScope(Scope::FnScope | Scope::DeclScope);
    m_DerivativeFnScope = getCurrentScope();
    beginBlock();

    Stmt* bodyDiff = Visit(FD->getBody()).getStmt();
    CompoundStmt* CS = cast<CompoundStmt>(bodyDiff);
    for (Stmt* S : CS->body())
      addToCurrentBlock(S);

    Stmt* derivativeBody = endBlock();
    m_Derivative->setBody(derivativeBody);

    endScope(); // Function body scope
    m_Sema.PopFunctionScopeInfo();
    m_Sema.PopDeclContext();
    endScope(); // Function decl scope

    m_DerivativeInFlight = false;
    return DerivativeAndOverload{cloneFunctionResult.first};
  }

  StmtDiff ForwardModeVisitor::VisitReturnStmt(const ReturnStmt* RS) {
    //If there is no return value, we must not attempt to differentiate
    if (!RS->getRetValue())
      return nullptr;
    
    StmtDiff retValDiff = Visit(RS->getRetValue());
    llvm::SmallVector<Expr*, 2> returnValues = {retValDiff.getExpr(),
                                                retValDiff.getExpr_dx()};
    SourceLocation fakeInitLoc = utils::GetValidSLoc(m_Sema);
    // This can instantiate as part of the move or copy initialization and
    // needs a fake source location.
    Expr* initList =
        m_Sema.ActOnInitList(fakeInitLoc, returnValues, noLoc).get();

    SourceLocation fakeRetLoc = utils::GetValidSLoc(m_Sema);
    Stmt* returnStmt =
        m_Sema.ActOnReturnStmt(fakeRetLoc, initList, getCurrentScope()).get();
    return StmtDiff(returnStmt);
  }
} // end namespace clad
