//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// version: $Id: ClangPlugin.cpp 7 2013-06-01 22:48:03Z v.g.vassilev@gmail.com $
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#include "clad/Differentiator/ReverseModeVisitor.h"

#include "ConstantFolder.h"

#include "TBRAnalyzer.h"
#include "clad/Differentiator/DerivativeBuilder.h"
#include "clad/Differentiator/DiffPlanner.h"
#include "clad/Differentiator/ErrorEstimator.h"
#include "clad/Differentiator/ExternalRMVSource.h"
#include "clad/Differentiator/MultiplexExternalRMVSource.h"
#include "clad/Differentiator/StmtClone.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTLambda.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/TemplateBase.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/TokenKinds.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Overload.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaInternal.h"
#include "clang/Sema/Template.h"
#include <clang/AST/DeclCXX.h>
#include <clang/AST/ExprCXX.h>
#include <clang/AST/OperationKinds.h>
#include <clang/Sema/Ownership.h>

#include "llvm/Support/SaveAndRestore.h"

#include <algorithm>
#include <numeric>

#include "clad/Differentiator/CladUtils.h"
#include "clad/Differentiator/Compatibility.h"

using namespace clang;

namespace clad {

Expr* getArraySizeExpr(const ArrayType* AT, ASTContext& context,
                       ReverseModeVisitor& rvm) {
  if (const auto* const CAT = dyn_cast<ConstantArrayType>(AT))
    return ConstantFolder::synthesizeLiteral(context.getSizeType(), context,
                                             CAT->getSize().getZExtValue());
  if (const auto* VSAT = dyn_cast<VariableArrayType>(AT))
    return rvm.Clone(VSAT->getSizeExpr());

  return nullptr;
}

  Expr* ReverseModeVisitor::CladTapeResult::Last() {
    LookupResult& Back = V.GetCladTapeBack();
    CXXScopeSpec CSS;
    CSS.Extend(V.m_Context, V.GetCladNamespace(), noLoc, noLoc);
    Expr* BackDRE = V.m_Sema
                        .BuildDeclarationNameExpr(CSS, Back,
                                                  /*AcceptInvalidDecl=*/false)
                        .get();
    Expr* Call =
        V.m_Sema.ActOnCallExpr(V.getCurrentScope(), BackDRE, noLoc, Ref, noLoc)
            .get();
    return Call;
  }

  ReverseModeVisitor::CladTapeResult
  ReverseModeVisitor::MakeCladTapeFor(Expr* E, llvm::StringRef prefix) {
    assert(E && "must be provided");
    E = E->IgnoreImplicit();
    QualType TapeType =
        GetCladTapeOfType(getNonConstType(E->getType(), m_Context, m_Sema));
    LookupResult& Push = GetCladTapePush();
    LookupResult& Pop = GetCladTapePop();
    Expr* TapeRef =
        BuildDeclRef(GlobalStoreImpl(TapeType, prefix, getZeroInit(TapeType)));
    auto* VD = cast<VarDecl>(cast<DeclRefExpr>(TapeRef)->getDecl());
    // Add fake location, since Clang AST does assert(Loc.isValid()) somewhere.
    VD->setLocation(m_DiffReq->getLocation());
    CXXScopeSpec CSS;
    CSS.Extend(m_Context, GetCladNamespace(), noLoc, noLoc);
    auto* PopDRE = m_Sema
                       .BuildDeclarationNameExpr(CSS, Pop,
                                                 /*AcceptInvalidDecl=*/false)
                       .get();
    auto* PushDRE = m_Sema
                        .BuildDeclarationNameExpr(CSS, Push,
                                                  /*AcceptInvalidDecl=*/false)
                        .get();
    Expr* PopExpr =
        m_Sema.ActOnCallExpr(getCurrentScope(), PopDRE, noLoc, TapeRef, noLoc)
            .get();
    Expr* CallArgs[] = {TapeRef, E};
    Expr* PushExpr =
        m_Sema.ActOnCallExpr(getCurrentScope(), PushDRE, noLoc, CallArgs, noLoc)
            .get();
    return CladTapeResult{*this, PushExpr, PopExpr, TapeRef};
  }

  bool ReverseModeVisitor::shouldUseCudaAtomicOps(const Expr* E) {
    // Same as checking whether this is a function executed by the GPU
    if (!m_CUDAGlobalArgs.empty())
      if (const auto* DRE = dyn_cast<DeclRefExpr>(E))
        if (const auto* PVD = dyn_cast<ParmVarDecl>(DRE->getDecl()))
          // Check whether this param is in the global memory of the GPU
          return m_CUDAGlobalArgs.find(PVD) != m_CUDAGlobalArgs.end();

    return false;
  }

  clang::Expr* ReverseModeVisitor::BuildCallToCudaAtomicAdd(clang::Expr* LHS,
                                                            clang::Expr* RHS) {
    DeclarationName atomicAddId = &m_Context.Idents.get("atomicAdd");
    LookupResult lookupResult(m_Sema, atomicAddId, SourceLocation(),
                              Sema::LookupOrdinaryName);
    m_Sema.LookupQualifiedName(lookupResult,
                               m_Context.getTranslationUnitDecl());

    CXXScopeSpec SS;
    Expr* UnresolvedLookup =
        m_Sema.BuildDeclarationNameExpr(SS, lookupResult, /*ADL=*/true).get();

    Expr* finalLHS = LHS;
    if (auto* UO = dyn_cast<UnaryOperator>(LHS)) {
      if (UO->getOpcode() == UnaryOperatorKind::UO_Deref)
        finalLHS = UO->getSubExpr()->IgnoreImplicit();
    } else if (!LHS->getType()->isPointerType() &&
               !LHS->getType()->isReferenceType())
      finalLHS = BuildOp(UnaryOperatorKind::UO_AddrOf, LHS);

    llvm::SmallVector<Expr*, 2> atomicArgs = {finalLHS, RHS};

    assert(!m_Builder.noOverloadExists(UnresolvedLookup, atomicArgs) &&
           "atomicAdd function not found");

    Expr* atomicAddCall =
        m_Sema
            .ActOnCallExpr(
                getCurrentScope(),
                /*Fn=*/UnresolvedLookup,
                /*LParenLoc=*/noLoc,
                /*ArgExprs=*/llvm::MutableArrayRef<Expr*>(atomicArgs),
                /*RParenLoc=*/m_DiffReq->getLocation())
            .get();

    return atomicAddCall;
  }

  ReverseModeVisitor::ReverseModeVisitor(DerivativeBuilder& builder,
                                         const DiffRequest& request)
      : VisitorBase(builder, request), m_Result(nullptr) {}

  ReverseModeVisitor::~ReverseModeVisitor() {
    if (m_ExternalSource) {
      // Inform external sources that `ReverseModeVisitor` object no longer
      // exists.
      // FIXME: Make this so the lifetime scope of the source matches.
      // m_ExternalSource->ForgetRMV();
      // Free the external sources multiplexer since we own this resource.
      delete m_ExternalSource;
    }
  }

  FunctionDecl* ReverseModeVisitor::CreateGradientOverload() {
    auto gradientParams = m_Derivative->parameters();
    auto gradientNameInfo = m_Derivative->getNameInfo();
    // Calculate the total number of parameters that would be required for
    // automatic differentiation in the derived function if all args are
    // requested.
    // FIXME: Here we are assuming all function parameters are of differentiable
    // type. Ideally, we should not make any such assumption.
    std::size_t totalDerivedParamsSize = m_DiffReq->getNumParams() * 2;
    std::size_t numOfDerivativeParams = m_DiffReq->getNumParams();

    // Account for the this pointer.
    if (isa<CXXMethodDecl>(m_DiffReq.Function) &&
        !utils::IsStaticMethod(m_DiffReq.Function))
      ++numOfDerivativeParams;
    // All output parameters will be of type `void*`. These
    // parameters will be casted to correct type before the call to the actual
    // derived function.
    // We require each output parameter to be of same type in the overloaded
    // derived function due to limitations of generating the exact derived
    // function type at the compile-time (without clad plugin help).
    QualType outputParamType = m_Context.getPointerType(m_Context.VoidTy);

    llvm::SmallVector<QualType, 16> paramTypes;

    // Add types for representing original function parameters.
    for (auto* PVD : m_DiffReq->parameters())
      paramTypes.push_back(PVD->getType());
    // Add types for representing parameter derivatives.
    // FIXME: We are assuming all function parameters are differentiable. We
    // should not make any such assumptions.
    for (std::size_t i = 0; i < numOfDerivativeParams; ++i)
      paramTypes.push_back(outputParamType);

    auto gradFuncOverloadEPI =
        dyn_cast<FunctionProtoType>(m_DiffReq->getType())->getExtProtoInfo();
    QualType gradientFunctionOverloadType =
        m_Context.getFunctionType(m_Context.VoidTy, paramTypes,
                                  // Cast to function pointer.
                                  gradFuncOverloadEPI);

    // FIXME: We should not use const_cast to get the decl context here.
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    auto* DC = const_cast<DeclContext*>(m_DiffReq->getDeclContext());
    m_Sema.CurContext = DC;
    DeclWithContext gradientOverloadFDWC =
        m_Builder.cloneFunction(m_DiffReq.Function, *this, DC, noLoc,
                                gradientNameInfo, gradientFunctionOverloadType);
    FunctionDecl* gradientOverloadFD = gradientOverloadFDWC.first;

    beginScope(Scope::FunctionPrototypeScope | Scope::FunctionDeclarationScope |
               Scope::DeclScope);
    m_Sema.PushFunctionScope();
    m_Sema.PushDeclContext(getCurrentScope(), gradientOverloadFD);

    llvm::SmallVector<ParmVarDecl*, 4> overloadParams;
    llvm::SmallVector<Expr*, 4> callArgs;

    overloadParams.reserve(totalDerivedParamsSize);
    callArgs.reserve(gradientParams.size());

    for (auto* PVD : m_DiffReq->parameters()) {
      auto* VD = utils::BuildParmVarDecl(
          m_Sema, gradientOverloadFD, PVD->getIdentifier(), PVD->getType(),
          PVD->getStorageClass(), /*defArg=*/nullptr, PVD->getTypeSourceInfo());
      overloadParams.push_back(VD);
      callArgs.push_back(BuildDeclRef(VD));
    }

    for (std::size_t i = 0; i < numOfDerivativeParams; ++i) {
      IdentifierInfo* II = nullptr;
      StorageClass SC = StorageClass::SC_None;
      std::size_t effectiveGradientIndex = m_DiffReq->getNumParams() + i;
      // `effectiveGradientIndex < gradientParams.size()` implies that this
      // parameter represents an actual derivative of one of the function
      // original parameters.
      if (effectiveGradientIndex < gradientParams.size()) {
        auto* GVD = gradientParams[effectiveGradientIndex];
        II = CreateUniqueIdentifier("_temp_" + GVD->getNameAsString());
        SC = GVD->getStorageClass();
      } else {
        II = CreateUniqueIdentifier("_d_" + std::to_string(i));
      }
      auto* PVD = utils::BuildParmVarDecl(m_Sema, gradientOverloadFD, II,
                                          outputParamType, SC);
      overloadParams.push_back(PVD);
    }

    for (auto* PVD : overloadParams)
      if (PVD->getIdentifier())
        m_Sema.PushOnScopeChains(PVD, getCurrentScope(),
                                 /*AddToContext=*/false);

    gradientOverloadFD->setParams(overloadParams);
    gradientOverloadFD->setBody(/*B=*/nullptr);

    beginScope(Scope::FnScope | Scope::DeclScope);
    m_DerivativeFnScope = getCurrentScope();
    beginBlock();

    // Build derivatives to be used in the call to the actual derived function.
    // These are initialised by effectively casting the derivative parameters of
    // overloaded derived function to the correct type.
    for (std::size_t i = m_DiffReq->getNumParams(); i < gradientParams.size();
         ++i) {
      auto* overloadParam = overloadParams[i];
      auto* gradientParam = gradientParams[i];
      TypeSourceInfo* typeInfo =
          m_Context.getTrivialTypeSourceInfo(gradientParam->getType());
      SourceLocation fakeLoc = utils::GetValidSLoc(m_Sema);
      auto* init = m_Sema
                       .BuildCStyleCastExpr(fakeLoc, typeInfo, fakeLoc,
                                            BuildDeclRef(overloadParam))
                       .get();

      auto* gradientVD = BuildGlobalVarDecl(gradientParam->getType(),
                                            gradientParam->getName(), init);
      callArgs.push_back(BuildDeclRef(gradientVD));
      addToCurrentBlock(BuildDeclStmt(gradientVD));
    }

    // If the function is a global kernel, we need to transform it
    // into a device function when calling it inside the overload function
    // which is the final global kernel returned.
    if (m_Derivative->hasAttr<clang::CUDAGlobalAttr>()) {
      m_Derivative->dropAttr<clang::CUDAGlobalAttr>();
      m_Derivative->addAttr(clang::CUDADeviceAttr::CreateImplicit(m_Context));
    }

    Expr* callExpr = BuildCallExprToFunction(m_Derivative, callArgs,
                                             /*UseRefQualifiedThisObj=*/true);
    addToCurrentBlock(callExpr);
    Stmt* gradientOverloadBody = endBlock();

    gradientOverloadFD->setBody(gradientOverloadBody);

    endScope(); // Function body scope
    m_Sema.PopFunctionScopeInfo();
    m_Sema.PopDeclContext();
    endScope(); // Function decl scope

    return gradientOverloadFD;
  }

  DerivativeAndOverload ReverseModeVisitor::Derive() {
    const FunctionDecl* FD = m_DiffReq.Function;
    if (m_ExternalSource)
      m_ExternalSource->ActOnStartOfDerive();

    // FIXME: reverse mode plugins may have request mode other than
    // `DiffMode::reverse`, but they still need the `DiffMode::reverse` mode
    // specific behaviour, because they are "reverse" mode plugins.
    // assert(m_DiffReq.Mode == DiffMode::reverse ||
    //        m_DiffReq.Mode == DiffMode::jacobian && "Unexpected Mode.");
    if (m_DiffReq.Mode == DiffMode::error_estimation)
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
      const_cast<DiffRequest&>(m_DiffReq).Mode = DiffMode::reverse;

    m_Pullback =
        ConstantFolder::synthesizeLiteral(m_Context.IntTy, m_Context, 1);
    assert(m_DiffReq.Function && "Must not be null.");

    DiffParams args{};
    if (m_DiffReq.Args)
      for (const auto& dParam : m_DiffReq.DVI)
        args.push_back(dParam.param);
    else
      std::copy(FD->param_begin(), FD->param_end(), std::back_inserter(args));
    if (args.empty())
      return {};

    if (m_ExternalSource)
      m_ExternalSource->ActAfterParsingDiffArgs(m_DiffReq, args);
    // Save the type of the output parameter(s) that is add by clad to the
    // derived function
    if (m_DiffReq.Mode == DiffMode::jacobian) {
      unsigned lastArgN = m_DiffReq->getNumParams() - 1;
      outputArrayStr = m_DiffReq->getParamDecl(lastArgN)->getNameAsString();
    }

    auto derivativeBaseName = m_DiffReq.BaseFunctionName;
    std::string gradientName = derivativeBaseName + funcPostfix();
    // To be consistent with older tests, nothing is appended to 'f_grad' if
    // we differentiate w.r.t. all the parameters at once.
    if (m_DiffReq.Mode == DiffMode::jacobian) {
      // If Jacobian is asked, the last parameter is the result parameter
      // and should be ignored
      if (args.size() != FD->getNumParams()-1){
        for (const auto* arg : args) {
          const auto* const it =
              std::find(FD->param_begin(), FD->param_end() - 1, arg);
          auto idx = std::distance(FD->param_begin(), it);
          gradientName += ('_' + std::to_string(idx));
        }
      }
    } else if (args.size() != FD->getNumParams()) {
      for (const auto* arg : args) {
        const auto* it = std::find(FD->param_begin(), FD->param_end(), arg);
        auto idx = std::distance(FD->param_begin(), it);
        gradientName += ('_' + std::to_string(idx));
      }
    }

    IdentifierInfo* II = &m_Context.Idents.get(gradientName);
    DeclarationNameInfo name(II, noLoc);

    // If we are in error estimation mode, we have an extra `double&`
    // parameter that stores the final error
    unsigned numExtraParam = 0;
    if (m_ExternalSource)
      m_ExternalSource->ActBeforeCreatingDerivedFnParamTypes(numExtraParam);

    auto paramTypes = ComputeParamTypes(args);

    if (m_ExternalSource)
      m_ExternalSource->ActAfterCreatingDerivedFnParamTypes(paramTypes);

    // If reverse mode differentiates only part of the arguments it needs to
    // generate an overload that can take in all the diff variables
    bool shouldCreateOverload = false;
    // FIXME: Gradient overload doesn't know how to handle additional parameters
    // added by the plugins yet.
    if (m_DiffReq.Mode != DiffMode::jacobian && numExtraParam == 0)
      shouldCreateOverload = true;
    if (!m_DiffReq.DeclarationOnly && !m_DiffReq.DerivedFDPrototypes.empty())
      // If the overload is already created, we don't need to create it again.
      shouldCreateOverload = false;

    const auto* originalFnType =
        dyn_cast<FunctionProtoType>(m_DiffReq->getType());
    // For a function f of type R(A1, A2, ..., An),
    // the type of the gradient function is void(A1, A2, ..., An, R*, R*, ...,
    // R*) . the type of the jacobian function is void(A1, A2, ..., An, R*, R*)
    // and for error estimation, the function type is
    // void(A1, A2, ..., An, R*, R*, ..., R*, double&)
    QualType gradientFunctionType = m_Context.getFunctionType(
        m_Context.VoidTy,
        llvm::ArrayRef<QualType>(paramTypes.data(), paramTypes.size()),
        // Cast to function pointer.
        originalFnType->getExtProtoInfo());

    // Check if the function is already declared as a custom derivative.
    // FIXME: We should not use const_cast to get the decl context here.
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    auto* DC = const_cast<DeclContext*>(m_DiffReq->getDeclContext());
    if (FunctionDecl* customDerivative = m_Builder.LookupCustomDerivativeDecl(
            gradientName, DC, gradientFunctionType)) {
      // Set m_Derivative for creating the overload.
      m_Derivative = customDerivative;
      FunctionDecl* gradientOverloadFD = nullptr;
      if (shouldCreateOverload)
        gradientOverloadFD = CreateGradientOverload();
      return DerivativeAndOverload{customDerivative, gradientOverloadFD};
    }

    // Create the gradient function declaration.
    llvm::SaveAndRestore<DeclContext*> SaveContext(m_Sema.CurContext);
    llvm::SaveAndRestore<Scope*> SaveScope(getCurrentScope(),
                                           getEnclosingNamespaceOrTUScope());
    m_Sema.CurContext = DC;
    DeclWithContext result = m_Builder.cloneFunction(
        m_DiffReq.Function, *this, DC, noLoc, name, gradientFunctionType);
    FunctionDecl* gradientFD = result.first;
    m_Derivative = gradientFD;

    if (m_ExternalSource)
      m_ExternalSource->ActBeforeCreatingDerivedFnScope();

    // Function declaration scope
    beginScope(Scope::FunctionPrototypeScope | Scope::FunctionDeclarationScope |
               Scope::DeclScope);
    m_Sema.PushFunctionScope();
    m_Sema.PushDeclContext(getCurrentScope(), m_Derivative);

    if (m_ExternalSource)
      m_ExternalSource->ActAfterCreatingDerivedFnScope();

    auto params = BuildParams(args);

    if (m_ExternalSource)
      m_ExternalSource->ActAfterCreatingDerivedFnParams(params);

    // if the function is a global kernel, all its parameters reside in the
    // global memory of the GPU
    if (m_DiffReq->hasAttr<clang::CUDAGlobalAttr>())
      for (auto* param : params)
        m_CUDAGlobalArgs.emplace(param);

    llvm::ArrayRef<ParmVarDecl*> paramsRef =
        clad_compat::makeArrayRef(params.data(), params.size());
    gradientFD->setParams(paramsRef);
    gradientFD->setBody(nullptr);

    if (!m_DiffReq.DeclarationOnly) {
      if (m_DiffReq.Mode == DiffMode::jacobian) {
        // Reference to the output parameter.
        m_Result = BuildDeclRef(params.back());
        numParams = args.size();

        // Creates the ArraySubscriptExprs for the independent variables
        size_t idx = 0;
        for (const auto* arg : args) {
          // FIXME: fix when adding array inputs, now we are just skipping all
          // array/pointer inputs (not treating them as independent variables).
          if (utils::isArrayOrPointerType(arg->getType())) {
            if (arg->getName() == "p")
              m_Variables[arg] = m_Result;
            idx += 1;
            continue;
          }
          auto size_type = m_Context.getSizeType();
          unsigned size_type_bits = m_Context.getIntWidth(size_type);
          // Create the idx literal.
          auto* i = IntegerLiteral::Create(
              m_Context, llvm::APInt(size_type_bits, idx), size_type, noLoc);
          // Create the jacobianMatrix[idx] expression.
          auto* result_at_i =
              m_Sema.CreateBuiltinArraySubscriptExpr(m_Result, noLoc, i, noLoc)
                  .get();
          m_Variables[arg] = result_at_i;
          idx += 1;
          m_IndependentVars.push_back(arg);
        }
      }

      if (m_ExternalSource)
        m_ExternalSource->ActBeforeCreatingDerivedFnBodyScope();

      // Function body scope.
      beginScope(Scope::FnScope | Scope::DeclScope);
      m_DerivativeFnScope = getCurrentScope();
      beginBlock();
      if (m_ExternalSource)
        m_ExternalSource->ActOnStartOfDerivedFnBody(m_DiffReq);

      Stmt* gradientBody = nullptr;

      if (!m_DiffReq.use_enzyme)
        DifferentiateWithClad();
      else
        DifferentiateWithEnzyme();

      gradientBody = endBlock();
      m_Derivative->setBody(gradientBody);
      endScope(); // Function body scope

      // Size >= current derivative order means that there exists a declaration
      // or prototype for the currently derived function.
      if (m_DiffReq.DerivedFDPrototypes.size() >=
          m_DiffReq.CurrentDerivativeOrder)
        m_Derivative->setPreviousDeclaration(
            m_DiffReq
                .DerivedFDPrototypes[m_DiffReq.CurrentDerivativeOrder - 1]);
    }
    m_Sema.PopFunctionScopeInfo();
    m_Sema.PopDeclContext();
    endScope(); // Function decl scope

    FunctionDecl* gradientOverloadFD = nullptr;
    if (shouldCreateOverload) {
      gradientOverloadFD =
          CreateGradientOverload();
    }

    return DerivativeAndOverload{result.first, gradientOverloadFD};
  }

  DerivativeAndOverload ReverseModeVisitor::DerivePullback() {
    const clang::FunctionDecl* FD = m_DiffReq.Function;
    // FIXME: Duplication of external source here is a workaround
    // for the two 'Derive's being different functions.
    if (m_ExternalSource)
      m_ExternalSource->ActOnStartOfDerive();
    assert(m_DiffReq.Mode == DiffMode::experimental_pullback);
    assert(m_DiffReq.Function && "Must not be null.");

    DiffParams args{};
    if (!m_DiffReq.DVI.empty())
      for (const auto& dParam : m_DiffReq.DVI)
        args.push_back(dParam.param);
    else
      std::copy(FD->param_begin(), FD->param_end(), std::back_inserter(args));
#ifndef NDEBUG
    bool isStaticMethod = utils::IsStaticMethod(FD);
    assert((!args.empty() || !isStaticMethod) &&
           "Cannot generate pullback function of a function "
           "with no differentiable arguments");
#endif

    if (m_ExternalSource)
      m_ExternalSource->ActAfterParsingDiffArgs(m_DiffReq, args);

    auto derivativeName =
        utils::ComputeEffectiveFnName(m_DiffReq.Function) + "_pullback";
    for (auto index : m_DiffReq.CUDAGlobalArgsIndexes)
      derivativeName += "_" + std::to_string(index);
    auto DNI = utils::BuildDeclarationNameInfo(m_Sema, derivativeName);

    auto paramTypes = ComputeParamTypes(args);
    const auto* originalFnType =
        dyn_cast<FunctionProtoType>(m_DiffReq->getType());

    if (m_ExternalSource)
      m_ExternalSource->ActAfterCreatingDerivedFnParamTypes(paramTypes);

    QualType pullbackFnType = m_Context.getFunctionType(
        m_Context.VoidTy, paramTypes, originalFnType->getExtProtoInfo());

    llvm::SaveAndRestore<DeclContext*> saveContext(m_Sema.CurContext);
    llvm::SaveAndRestore<Scope*> saveScope(getCurrentScope(),
                                           getEnclosingNamespaceOrTUScope());
    // FIXME: We should not use const_cast to get the decl context here.
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    m_Sema.CurContext = const_cast<DeclContext*>(m_DiffReq->getDeclContext());

    SourceLocation validLoc{m_DiffReq->getLocation()};
    DeclWithContext fnBuildRes =
        m_Builder.cloneFunction(m_DiffReq.Function, *this, m_Sema.CurContext,
                                validLoc, DNI, pullbackFnType);
    m_Derivative = fnBuildRes.first;

    if (m_ExternalSource)
      m_ExternalSource->ActBeforeCreatingDerivedFnScope();

    beginScope(Scope::FunctionPrototypeScope | Scope::FunctionDeclarationScope |
               Scope::DeclScope);
    m_Sema.PushFunctionScope();
    m_Sema.PushDeclContext(getCurrentScope(), m_Derivative);

    if (m_ExternalSource)
      m_ExternalSource->ActAfterCreatingDerivedFnScope();

    auto params = BuildParams(args);
    if (m_ExternalSource)
      m_ExternalSource->ActAfterCreatingDerivedFnParams(params);

    m_Derivative->setParams(params);
    // Match the global arguments of the call to the device function to the
    // pullback function's parameters.
    if (!m_DiffReq.CUDAGlobalArgsIndexes.empty())
      for (auto index : m_DiffReq.CUDAGlobalArgsIndexes)
        m_CUDAGlobalArgs.emplace(m_Derivative->getParamDecl(index));

    m_Derivative->setBody(nullptr);

    if (!m_DiffReq.DeclarationOnly) {
      if (m_ExternalSource)
        m_ExternalSource->ActBeforeCreatingDerivedFnBodyScope();

      beginScope(Scope::FnScope | Scope::DeclScope);
      m_DerivativeFnScope = getCurrentScope();

      beginBlock();
      if (m_ExternalSource)
        m_ExternalSource->ActOnStartOfDerivedFnBody(m_DiffReq);

      StmtDiff bodyDiff = Visit(m_DiffReq->getBody());
      Stmt* forward = bodyDiff.getStmt();
      Stmt* reverse = bodyDiff.getStmt_dx();

      // Create the body of the function.
      // Firstly, all "global" Stmts are put into fn's body.
      for (Stmt* S : m_Globals)
        addToCurrentBlock(S, direction::forward);
      // Forward pass.
      if (auto* CS = dyn_cast<CompoundStmt>(forward))
        for (Stmt* S : CS->body())
          addToCurrentBlock(S, direction::forward);

      // Reverse pass.
      if (auto* RCS = dyn_cast<CompoundStmt>(reverse))
        for (Stmt* S : RCS->body())
          addToCurrentBlock(S, direction::forward);

      if (m_ExternalSource)
        m_ExternalSource->ActOnEndOfDerivedFnBody();

      Stmt* fnBody = endBlock();
      m_Derivative->setBody(fnBody);
      endScope(); // Function body scope

      // Size >= current derivative order means that there exists a declaration
      // or prototype for the currently derived function.
      if (m_DiffReq.DerivedFDPrototypes.size() >=
          m_DiffReq.CurrentDerivativeOrder)
        m_Derivative->setPreviousDeclaration(
            m_DiffReq
                .DerivedFDPrototypes[m_DiffReq.CurrentDerivativeOrder - 1]);
    }
    m_Sema.PopFunctionScopeInfo();
    m_Sema.PopDeclContext();
    endScope(); // Function decl scope

    return DerivativeAndOverload{fnBuildRes.first, nullptr};
  }

  void ReverseModeVisitor::DifferentiateWithClad() {
    llvm::ArrayRef<ParmVarDecl*> paramsRef = m_Derivative->parameters();

    // create derived variables for parameters which are not part of
    // independent variables (args).
    for (std::size_t i = 0; i < m_DiffReq->getNumParams(); ++i) {
      ParmVarDecl* param = paramsRef[i];
      // derived variables are already created for independent variables.
      if (m_Variables.count(param))
        continue;
      // in vector mode last non diff parameter is output parameter.
      if (m_DiffReq.Mode == DiffMode::jacobian &&
          i == m_DiffReq->getNumParams() - 1)
        continue;
      auto VDDerivedType = param->getType();
      // We cannot initialize derived variable for pointer types because
      // we do not know the correct size.
      if (utils::isArrayOrPointerType(VDDerivedType))
        continue;
      auto* VDDerived =
          BuildGlobalVarDecl(VDDerivedType, "_d_" + param->getNameAsString(),
                             getZeroInit(VDDerivedType));
      m_Variables[param] = BuildDeclRef(VDDerived);
      addToBlock(BuildDeclStmt(VDDerived), m_Globals);
    }
    // Start the visitation process which outputs the statements in the
    // current block.
    StmtDiff BodyDiff = Visit(m_DiffReq->getBody());
    Stmt* Forward = BodyDiff.getStmt();
    Stmt* Reverse = BodyDiff.getStmt_dx();
    // Create the body of the function.
    // Firstly, all "global" Stmts are put into fn's body.
    for (Stmt* S : m_Globals)
      addToCurrentBlock(S, direction::forward);
    // Forward pass.
    if (auto* CS = dyn_cast<CompoundStmt>(Forward))
      for (Stmt* S : CS->body())
        addToCurrentBlock(S, direction::forward);
    else
      addToCurrentBlock(Forward, direction::forward);
    // Reverse pass.
    if (auto* RCS = dyn_cast<CompoundStmt>(Reverse))
      for (Stmt* S : RCS->body())
        addToCurrentBlock(S, direction::forward);
    else
      addToCurrentBlock(Reverse, direction::forward);
    // Add delete statements present in m_DeallocExprs to the current block.
    for (auto* S : m_DeallocExprs)
      if (auto* CS = dyn_cast<CompoundStmt>(S))
        for (Stmt* S : CS->body())
          addToCurrentBlock(S, direction::forward);
      else
        addToCurrentBlock(S, direction::forward);

    if (m_ExternalSource)
      m_ExternalSource->ActOnEndOfDerivedFnBody();
  }

  void ReverseModeVisitor::DifferentiateWithEnzyme() {
    unsigned numParams = m_DiffReq->getNumParams();
    auto origParams = m_DiffReq->parameters();
    llvm::ArrayRef<ParmVarDecl*> paramsRef = m_Derivative->parameters();
    const auto* originalFnType =
        dyn_cast<FunctionProtoType>(m_DiffReq->getType());

    // Prepare Arguments and Parameters to enzyme_autodiff
    llvm::SmallVector<Expr*, 16> enzymeArgs;
    llvm::SmallVector<ParmVarDecl*, 16> enzymeParams;
    llvm::SmallVector<ParmVarDecl*, 16> enzymeRealParams;
    llvm::SmallVector<ParmVarDecl*, 16> enzymeRealParamsDerived;

    // First add the function itself as a parameter/argument
    // FIXME: We should not use const_cast to get the decl context here.
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    enzymeArgs.push_back(
        BuildDeclRef(const_cast<FunctionDecl*>(m_DiffReq.Function)));
    // FIXME: We should not use const_cast to get the decl context here.
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    auto* fdDeclContext = const_cast<DeclContext*>(m_DiffReq->getDeclContext());
    enzymeParams.push_back(m_Sema.BuildParmVarDeclForTypedef(
        fdDeclContext, noLoc, m_DiffReq->getType()));

    // Add rest of the parameters/arguments
    for (unsigned i = 0; i < numParams; i++) {
      // First Add the original parameter
      enzymeArgs.push_back(BuildDeclRef(paramsRef[i]));
      enzymeParams.push_back(m_Sema.BuildParmVarDeclForTypedef(
          fdDeclContext, noLoc, paramsRef[i]->getType()));

      QualType paramType = origParams[i]->getOriginalType();
      // If original parameter is of a differentiable real type(but not
      // array/pointer), then add it to the list of params whose gradient must
      // be extracted later from the EnzymeGradient structure
      if (paramType->isRealFloatingType()) {
        enzymeRealParams.push_back(paramsRef[i]);
        enzymeRealParamsDerived.push_back(paramsRef[numParams + i]);
      } else if (utils::isArrayOrPointerType(paramType)) {
        // Add the corresponding array/pointer variable
        enzymeArgs.push_back(BuildDeclRef(paramsRef[numParams + i]));
        enzymeParams.push_back(m_Sema.BuildParmVarDeclForTypedef(
            fdDeclContext, noLoc, paramsRef[numParams + i]->getType()));
      }
    }

    llvm::SmallVector<QualType, 16> enzymeParamsType;
    for (auto* i : enzymeParams)
      enzymeParamsType.push_back(i->getType());

    QualType QT;
    if (!enzymeRealParams.empty()) {
      // Find the EnzymeGradient datastructure
      auto* gradDecl = LookupTemplateDeclInCladNamespace("EnzymeGradient");

      TemplateArgumentListInfo TLI{};
      llvm::APSInt argValue(std::to_string(enzymeRealParams.size()));
      TemplateArgument TA(m_Context, argValue, m_Context.UnsignedIntTy);
      TLI.addArgument(TemplateArgumentLoc(TA, TemplateArgumentLocInfo()));

      QT = InstantiateTemplate(gradDecl, TLI);
    } else {
      QT = m_Context.VoidTy;
    }

    // Prepare Function call
    std::string enzymeCallName =
        "__enzyme_autodiff_" + m_DiffReq->getNameAsString();
    IdentifierInfo* IIEnzyme = &m_Context.Idents.get(enzymeCallName);
    DeclarationName nameEnzyme(IIEnzyme);
    QualType enzymeFunctionType =
        m_Sema.BuildFunctionType(QT, enzymeParamsType, noLoc, nameEnzyme,
                                 originalFnType->getExtProtoInfo());
    FunctionDecl* enzymeCallFD = FunctionDecl::Create(
        m_Context, fdDeclContext, noLoc, noLoc, nameEnzyme, enzymeFunctionType,
        m_DiffReq->getTypeSourceInfo(), SC_Extern);
    enzymeCallFD->setParams(enzymeParams);
    Expr* enzymeCall = BuildCallExprToFunction(enzymeCallFD, enzymeArgs);

    // Prepare the statements that assign the gradients to
    // non array/pointer type parameters of the original function
    if (!enzymeRealParams.empty()) {
      auto* gradDeclStmt = BuildVarDecl(QT, "grad", enzymeCall, true);
      addToCurrentBlock(BuildDeclStmt(gradDeclStmt), direction::forward);

      for (unsigned i = 0; i < enzymeRealParams.size(); i++) {
        auto* LHSExpr =
            BuildOp(UO_Deref, BuildDeclRef(enzymeRealParamsDerived[i]));

        auto* ME = utils::BuildMemberExpr(m_Sema, getCurrentScope(),
                                          BuildDeclRef(gradDeclStmt), "d_arr");

        Expr* gradIndex = dyn_cast<Expr>(
            IntegerLiteral::Create(m_Context, llvm::APSInt(std::to_string(i)),
                                   m_Context.UnsignedIntTy, noLoc));
        Expr* RHSExpr =
            m_Sema.CreateBuiltinArraySubscriptExpr(ME, noLoc, gradIndex, noLoc)
                .get();

        auto* assignExpr = BuildOp(BO_Assign, LHSExpr, RHSExpr);
        addToCurrentBlock(assignExpr, direction::forward);
      }
    } else {
      // Add Function call to block
      Expr* enzymeCall = BuildCallExprToFunction(enzymeCallFD, enzymeArgs);
      addToCurrentBlock(enzymeCall);
    }
  }

  StmtDiff ReverseModeVisitor::VisitCXXStdInitializerListExpr(
      const clang::CXXStdInitializerListExpr* ILE) {
    return Visit(ILE->getSubExpr(), dfdx());
  }

  StmtDiff ReverseModeVisitor::VisitStmt(const Stmt* S) {
    diag(
        DiagnosticsEngine::Warning, S->getBeginLoc(),
        "attempted to differentiate unsupported statement, no changes applied");
    // Unknown stmt, just clone it.
    return StmtDiff(Clone(S));
  }

  StmtDiff ReverseModeVisitor::VisitCompoundStmt(const CompoundStmt* CS) {
    int scopeFlags = Scope::DeclScope;
    // If this is the outermost compound statement of the function,
    // propagate the function scope.
    if (getCurrentScope() == m_DerivativeFnScope)
      scopeFlags |= Scope::FnScope;
    beginScope(scopeFlags);
    beginBlock(direction::forward);
    beginBlock(direction::reverse);
    for (Stmt* S : CS->body()) {
      if (m_ExternalSource)
        m_ExternalSource->ActBeforeDifferentiatingStmtInVisitCompoundStmt();
      StmtDiff SDiff = DifferentiateSingleStmt(S);
      addToCurrentBlock(SDiff.getStmt(), direction::forward);
      addToCurrentBlock(SDiff.getStmt_dx(), direction::reverse);

      if (m_ExternalSource)
        m_ExternalSource->ActAfterProcessingStmtInVisitCompoundStmt();
    }
    CompoundStmt* Forward = endBlock(direction::forward);
    CompoundStmt* Reverse = endBlock(direction::reverse);
    endScope();
    return StmtDiff(Forward, Reverse);
  }

  StmtDiff ReverseModeVisitor::VisitIfStmt(const clang::IfStmt* If) {
    // Control scope of the IfStmt. E.g., in if (double x = ...) {...}, x goes
    // to this scope.
    beginScope(Scope::DeclScope | Scope::ControlScope);

    // Create a block "around" if statement, e.g:
    // {
    //   ...
    //  if (...) {...}
    // }
    beginBlock(direction::forward);
    beginBlock(direction::reverse);
    StmtDiff condDiff;
    // if the statement has an init, we process it
    if (If->hasInitStorage()) {
      StmtDiff initDiff = Visit(If->getInit());
      addToCurrentBlock(initDiff.getStmt(), direction::forward);
      addToCurrentBlock(initDiff.getStmt_dx(), direction::reverse);
    }
    // this ensures we can differentiate conditions that affect the derivatives
    // as well as declarations inside the condition:
    beginBlock(direction::reverse);
    if (const auto* condDeclStmt = If->getConditionVariableDeclStmt())
      condDiff = Visit(condDeclStmt);
    else
      condDiff = Visit(If->getCond());
    CompoundStmt* RCS = endBlock(direction::reverse);
    if (!RCS->body_empty()) {
      std::reverse(
          RCS->body_begin(),
          RCS->body_end()); // it is reversed in the endBlock() but we don't
                            // actually need this, so we reverse it once again
      addToCurrentBlock(RCS, direction::reverse);
    }

    // Condition has to be stored as a "global" variable, to take the correct
    // branch in the reverse pass.
    Expr* condDiffStored =
        GlobalStoreAndRef(condDiff.getExpr(), m_Context.BoolTy, "_cond");
    // Convert cond to boolean condition.
    if (condDiffStored)
      condDiffStored =
          m_Sema
              .ActOnCondition(getCurrentScope(), noLoc, condDiffStored,
                              Sema::ConditionKind::Boolean)
              .get()
              .second;

    auto VisitBranch = [&](const Stmt* Branch) -> StmtDiff {
      if (!Branch)
        return {};
      if (isa<CompoundStmt>(Branch)) {
        StmtDiff BranchDiff = Visit(Branch);
        return BranchDiff;
      }
      beginBlock(direction::forward);
      if (m_ExternalSource)
        m_ExternalSource
            ->ActBeforeDifferentiatingSingleStmtBranchInVisitIfStmt();
      StmtDiff BranchDiff = DifferentiateSingleStmt(Branch, /*dfdS=*/nullptr);
      addToCurrentBlock(BranchDiff.getStmt(), direction::forward);

      if (m_ExternalSource)
        m_ExternalSource
            ->ActBeforeFinalizingVisitBranchSingleStmtInIfVisitStmt();

      Stmt* Forward = utils::unwrapIfSingleStmt(endBlock(direction::forward));
      Stmt* Reverse = utils::unwrapIfSingleStmt(BranchDiff.getStmt_dx());
      return StmtDiff(Forward, Reverse);
    };

    StmtDiff thenDiff = VisitBranch(If->getThen());
    StmtDiff elseDiff = VisitBranch(If->getElse());
    Stmt* Forward = clad_compat::IfStmt_Create(
        m_Context, noLoc, If->isConstexpr(), /*Init=*/nullptr, /*Var=*/nullptr,
        condDiffStored, noLoc, noLoc, thenDiff.getStmt(), noLoc,
        elseDiff.getStmt());
    addToCurrentBlock(Forward, direction::forward);

    Stmt* Reverse = nullptr;
    // thenDiff.getStmt_dx() might be empty if TBR is on leadinf to a crash in
    // case of the braceless if.
    if (thenDiff.getStmt_dx())
      Reverse = clad_compat::IfStmt_Create(
          m_Context, noLoc, If->isConstexpr(), /*Init=*/nullptr,
          /*Var=*/nullptr, condDiffStored, noLoc, noLoc, thenDiff.getStmt_dx(),
          noLoc, elseDiff.getStmt_dx());
    else if (elseDiff.getStmt_dx())
      Reverse = clad_compat::IfStmt_Create(
          m_Context, noLoc, If->isConstexpr(), /*Init=*/nullptr,
          /*Var=*/nullptr,
          BuildOp(clang::UnaryOperatorKind::UO_LNot,
                  BuildParens(condDiffStored)),
          noLoc, noLoc, elseDiff.getStmt_dx(), noLoc, {});
    addToCurrentBlock(Reverse, direction::reverse);
    CompoundStmt* ForwardBlock = endBlock(direction::forward);
    CompoundStmt* ReverseBlock = endBlock(direction::reverse);
    endScope();
    return StmtDiff(utils::unwrapIfSingleStmt(ForwardBlock),
                    utils::unwrapIfSingleStmt(ReverseBlock),
                    /*forwSweepDiff=*/nullptr,
                    /*valueForRevSweep=*/condDiffStored);
  }

  StmtDiff ReverseModeVisitor::VisitConditionalOperator(
      const clang::ConditionalOperator* CO) {
    StmtDiff condDiff = Visit(CO->getCond());
    beginBlock(direction::reverse);
    addToCurrentBlock(condDiff.getStmt_dx(), direction::reverse);
    // Condition has to be stored as a "global" variable, to take the correct
    // branch in the reverse pass.
    Expr* condStored = GlobalStoreAndRef(condDiff.getExpr(), "_cond");
    // Convert cond to boolean condition.
    condStored = m_Sema
                     .ActOnCondition(getCurrentScope(), noLoc, condStored,
                                     Sema::ConditionKind::Boolean)
                     .get()
                     .second;

    auto* ifTrue = CO->getTrueExpr();
    auto* ifFalse = CO->getFalseExpr();

    auto VisitBranch = [&](const Expr* Branch,
                           Expr* dfdx) -> std::pair<StmtDiff, StmtDiff> {
      beginScope(Scope::DeclScope);
      auto Result = DifferentiateSingleExpr(Branch, dfdx);
      endScope();
      StmtDiff BranchDiff = Result.first;
      StmtDiff ExprDiff = Result.second;
      Stmt* Forward = utils::unwrapIfSingleStmt(BranchDiff.getStmt());
      Stmt* Reverse = utils::unwrapIfSingleStmt(BranchDiff.getStmt_dx());
      return {StmtDiff(Forward, Reverse), ExprDiff};
    };

    StmtDiff ifTrueDiff;
    StmtDiff ifTrueExprDiff;
    StmtDiff ifFalseDiff;
    StmtDiff ifFalseExprDiff;

    std::tie(ifTrueDiff, ifTrueExprDiff) = VisitBranch(ifTrue, dfdx());
    std::tie(ifFalseDiff, ifFalseExprDiff) = VisitBranch(ifFalse, dfdx());

    auto BuildIf = [&](Expr* Cond, Stmt* Then, Stmt* Else) -> Stmt* {
      if (!Then && !Else)
        return nullptr;
      if (!Then)
        Then = m_Sema.ActOnNullStmt(noLoc).get();
      return clad_compat::IfStmt_Create(m_Context,
                                        noLoc,
                                        false,
                                        nullptr,
                                        nullptr,
                                        Cond,
                                        noLoc,
                                        noLoc,
                                        Then,
                                        noLoc,
                                        Else);
    };

    Stmt* Forward =
        BuildIf(condStored, ifTrueDiff.getStmt(), ifFalseDiff.getStmt());
    Stmt* Reverse =
        BuildIf(condStored, ifTrueDiff.getStmt_dx(), ifFalseDiff.getStmt_dx());
    if (Forward)
      addToCurrentBlock(Forward, direction::forward);
    if (Reverse)
      addToCurrentBlock(Reverse, direction::reverse);

    Expr* condExpr = m_Sema
                         .ActOnConditionalOp(noLoc, noLoc, condStored,
                                             ifTrueExprDiff.getExpr(),
                                             ifFalseExprDiff.getExpr())
                         .get();
    // If result is a glvalue, we should keep it as it can potentially be
    // assigned as in (c ? a : b) = x;
    Expr* ResultRef = nullptr;
    if ((CO->isModifiableLvalue(m_Context) == Expr::MLV_Valid) &&
        ifTrueExprDiff.getExpr_dx() && ifFalseExprDiff.getExpr_dx()) {
      ResultRef = m_Sema
                      .ActOnConditionalOp(noLoc, noLoc, condStored,
                                          ifTrueExprDiff.getExpr_dx(),
                                          ifFalseExprDiff.getExpr_dx())
                      .get();
      if (ResultRef->isModifiableLvalue(m_Context) != Expr::MLV_Valid)
        ResultRef = nullptr;
    }
    Stmt* revBlock = utils::unwrapIfSingleStmt(endBlock(direction::reverse));
    addToCurrentBlock(revBlock, direction::reverse);
    return StmtDiff(condExpr, ResultRef);
  }

  StmtDiff
  ReverseModeVisitor::VisitCXXForRangeStmt(const CXXForRangeStmt* FRS) {
    const auto* RangeDecl = cast<VarDecl>(FRS->getRangeStmt()->getSingleDecl());
    const auto* BeginDecl = cast<VarDecl>(FRS->getBeginStmt()->getSingleDecl());
    DeclDiff<VarDecl> VisitRange =
        DifferentiateVarDecl(RangeDecl, /*keepLocal=*/true);
    DeclDiff<VarDecl> VisitBegin =
        DifferentiateVarDecl(BeginDecl, /*keepLocal=*/true);

    beginBlock(direction::reverse);
    LoopCounter loopCounter(*this);
    beginScope(Scope::DeclScope | Scope::ControlScope | Scope::BreakScope |
               Scope::ContinueScope);

    llvm::SaveAndRestore<Expr*> SaveCurrentBreakFlagExpr(
        m_CurrentBreakFlagExpr);
    m_CurrentBreakFlagExpr = nullptr;
    auto* activeBreakContHandler = PushBreakContStmtHandler();
    activeBreakContHandler->BeginCFSwitchStmtScope();
    const VarDecl* LoopVD = FRS->getLoopVariable();

    llvm::SaveAndRestore<bool> SaveIsInside(isInsideLoop,
                                            /*NewValue=*/false);

    beginBlock(direction::reverse);
    // Create all declarations needed.
    DeclRefExpr* beginDeclRef = BuildDeclRef(VisitBegin.getDecl());
    Expr* d_beginDeclRef = m_Variables[beginDeclRef->getDecl()];
    addToCurrentBlock(BuildDeclStmt(VisitRange.getDecl()));
    addToCurrentBlock(BuildDeclStmt(VisitRange.getDecl_dx()));
    addToCurrentBlock(BuildDeclStmt(VisitBegin.getDecl()));
    addToCurrentBlock(BuildDeclStmt(VisitBegin.getDecl_dx()));

    const auto* EndDecl = cast<VarDecl>(FRS->getEndStmt()->getSingleDecl());
    QualType endType = CloneType(EndDecl->getType());
    std::string endName = EndDecl->getNameAsString();
    Expr* endInit = Visit(EndDecl->getInit()).getExpr();
    VarDecl* endVarDecl =
        BuildGlobalVarDecl(endType, endName, endInit, /*DirectInit=*/false);
    addToCurrentBlock(BuildDeclStmt(endVarDecl));
    DeclRefExpr* endExpr = BuildDeclRef(endVarDecl);
    Expr* incBegin = BuildOp(UO_PreInc, beginDeclRef);

    beginBlock(direction::forward);
    DeclDiff<VarDecl> LoopVDDiff = DifferentiateVarDecl(LoopVD);
    Stmt* adjLoopVDAddAssign =
        utils::unwrapIfSingleStmt(endBlock(direction::forward));

    llvm::SaveAndRestore<bool> SaveIsInsideLoop(isInsideLoop,
                                                /*NewValue=*/true);

    Expr* d_incBegin = BuildOp(UO_PreInc, d_beginDeclRef);
    Expr* d_decBegin = BuildOp(UO_PostDec, d_beginDeclRef);
    Expr* forwardCond = BuildOp(BO_NE, beginDeclRef, endExpr);
    const Stmt* body = FRS->getBody();
    StmtDiff bodyDiff =
        DifferentiateLoopBody(body, loopCounter, nullptr, nullptr,
                              /*isForLoop=*/true);

    activeBreakContHandler->EndCFSwitchStmtScope();
    activeBreakContHandler->UpdateForwAndRevBlocks(bodyDiff);
    PopBreakContStmtHandler();

    StmtDiff storeLoop = StoreAndRestore(BuildDeclRef(LoopVDDiff.getDecl()));
    StmtDiff storeAdjLoop =
        StoreAndRestore(BuildDeclRef(LoopVDDiff.getDecl_dx()));
    addToCurrentBlock(BuildDeclStmt(LoopVDDiff.getDecl_dx()));

    Expr* loopInit = LoopVDDiff.getDecl()->getInit();
    LoopVDDiff.getDecl()->setInit(getZeroInit(LoopVDDiff.getDecl()->getType()));
    addToCurrentBlock(BuildDeclStmt(LoopVDDiff.getDecl()));
    Expr* assignLoop =
        BuildOp(BO_Assign, BuildDeclRef(LoopVDDiff.getDecl()), loopInit);

    if (!LoopVD->getType()->isReferenceType()) {
      Expr* d_LoopVD = BuildDeclRef(LoopVDDiff.getDecl_dx());
      adjLoopVDAddAssign =
          BuildOp(BO_Assign, d_LoopVD, BuildOp(UO_Deref, d_beginDeclRef));
    }

    beginBlock(direction::forward);
    addToCurrentBlock(adjLoopVDAddAssign);
    addToCurrentBlock(assignLoop);
    addToCurrentBlock(storeLoop.getStmt());
    addToCurrentBlock(storeAdjLoop.getStmt());
    CompoundStmt* LoopVDForwardDiff = endBlock(direction::forward);
    CompoundStmt* bodyForward = utils::PrependAndCreateCompoundStmt(
        m_Sema.getASTContext(), bodyDiff.getStmt(), LoopVDForwardDiff);

    beginBlock(direction::forward);
    addToCurrentBlock(d_decBegin);
    addToCurrentBlock(storeLoop.getStmt_dx());
    addToCurrentBlock(storeAdjLoop.getStmt_dx());
    CompoundStmt* LoopVDReverseDiff = endBlock(direction::forward);
    CompoundStmt* bodyReverse = utils::PrependAndCreateCompoundStmt(
        m_Sema.getASTContext(), bodyDiff.getStmt_dx(), LoopVDReverseDiff);

    Expr* inc = BuildOp(BO_Comma, incBegin, d_incBegin);
    Stmt* Forward = new (m_Context) ForStmt(
        m_Context, /*Init=*/nullptr, forwardCond, /*CondVar=*/nullptr, inc,
        bodyForward, FRS->getForLoc(), FRS->getBeginLoc(), FRS->getEndLoc());
    Expr* counterCondition =
        loopCounter.getCounterConditionResult().get().second;
    Expr* counterDecrement = loopCounter.getCounterDecrement();

    Stmt* Reverse = bodyReverse;
    addToCurrentBlock(Reverse, direction::reverse);
    Reverse = endBlock(direction::reverse);

    Reverse = new (m_Context)
        ForStmt(m_Context, /*Init=*/nullptr, counterCondition,
                /*CondVar=*/nullptr, counterDecrement, Reverse,
                FRS->getForLoc(), FRS->getBeginLoc(), FRS->getEndLoc());
    addToCurrentBlock(Reverse, direction::reverse);
    Reverse = endBlock(direction::reverse);
    endScope();

    return {utils::unwrapIfSingleStmt(Forward),
            utils::unwrapIfSingleStmt(Reverse)};
  }

  StmtDiff ReverseModeVisitor::VisitForStmt(const ForStmt* FS) {
    beginBlock(direction::reverse);
    LoopCounter loopCounter(*this);
    beginScope(Scope::DeclScope | Scope::ControlScope | Scope::BreakScope |
               Scope::ContinueScope);
    llvm::SaveAndRestore<Expr*> SaveCurrentBreakFlagExpr(
        m_CurrentBreakFlagExpr);
    m_CurrentBreakFlagExpr = nullptr;
    const Stmt* init = FS->getInit();
    if (m_ExternalSource)
      m_ExternalSource->ActBeforeDifferentiatingLoopInitStmt();
    StmtDiff initResult = init ? DifferentiateSingleStmt(init) : StmtDiff{};

    // Save the isInsideLoop value (we may be inside another loop).
    llvm::SaveAndRestore<bool> SaveIsInsideLoop(isInsideLoop);
    isInsideLoop = true;
    StmtDiff condVarRes;
    VarDecl* condVarClone = nullptr;
    if (FS->getConditionVariable()) {
      condVarRes = DifferentiateSingleStmt(FS->getConditionVariableDeclStmt());
      if (isa<DeclStmt>(condVarRes.getStmt())) {
        Decl* decl = cast<DeclStmt>(condVarRes.getStmt())->getSingleDecl();
        condVarClone = cast<VarDecl>(decl);
      }
    }

    // but it is not generally true, e.g. for (...; (x = y); ...)...
    StmtDiff condDiff;
    StmtDiff condExprDiff;
    if (FS->getCond())
      std::tie(condDiff, condExprDiff) = DifferentiateSingleExpr(FS->getCond());

    const auto* IDRE = dyn_cast<DeclRefExpr>(FS->getInc());
    const Expr* inc = IDRE ? Visit(FS->getInc()).getExpr() : FS->getInc();

    // Differentiate the increment expression of the for loop
    // incExprDiff.getExpr() is the reconstructed expression, incDiff.getStmt()
    // a block with all the intermediate statements used to reconstruct it on
    // the forward pass, incDiff.getStmt_dx() is the reverse pass block.
    StmtDiff incDiff;
    StmtDiff incExprDiff;
    if (inc)
      std::tie(incDiff, incExprDiff) = DifferentiateSingleExpr(inc);
    Expr* incResult = nullptr;
    // If any additional statements were created, enclose them into lambda.
    auto* Additional = cast<CompoundStmt>(incDiff.getStmt());
    bool anyNonExpr = std::any_of(Additional->body_begin(),
                                  Additional->body_end(),
                                  [](Stmt* S) { return !isa<Expr>(S); });
    if (anyNonExpr) {
      incResult = wrapInLambda(*this, m_Sema, inc, [&] {
        std::tie(incDiff, incExprDiff) = DifferentiateSingleExpr(inc);
        for (Stmt* S : cast<CompoundStmt>(incDiff.getStmt())->body())
          addToCurrentBlock(S);
        addToCurrentBlock(incDiff.getExpr());
      });
    }
    // Otherwise, join all exprs by comma operator.
    else if (incExprDiff.getExpr()) {
      auto CommaJoin = [this](Expr* Acc, Stmt* S) {
        Expr* E = cast<Expr>(S);
        return BuildOp(BO_Comma, E, BuildParens(Acc));
      };
      incResult = std::accumulate(Additional->body_rbegin(),
                                  Additional->body_rend(),
                                  incExprDiff.getExpr(),
                                  CommaJoin);
    }

    const Stmt* body = FS->getBody();
    StmtDiff BodyDiff = DifferentiateLoopBody(body, loopCounter,
                                              condVarRes.getStmt_dx(),
                                              incDiff.getStmt_dx(),
                                              /*isForLoop=*/true);

    /// FIXME: This part in necessary to replace local variables inside loops
    /// with function globals and replace initializations with assignments.
    /// This is a temporary measure to avoid the bug that arises from
    /// overwriting local variables on different loop passes.
    Expr* forwardCond = condExprDiff.getExpr();
    /// If there is a declaration in the condition, `cond` will be
    /// a DeclRefExpr of the declared variable. There is no point in
    /// inserting it since condVarRes.getExpr() represents an assignment with
    /// that variable on the LHS.
    /// e.g. for condition `int x = y`,
    /// condVarRes.getExpr() will represent `x = y`
    if (condVarRes.getExpr() != nullptr && isa<Expr>(condVarRes.getExpr()))
      forwardCond = cast<Expr>(condVarRes.getExpr());

    Stmt* breakStmt = m_Sema.ActOnBreakStmt(noLoc, getCurrentScope()).get();

    /// This part adds the forward pass of loop condition stmt in the body
    /// In this first loop condition diff stmts execute then loop condition
    /// is checked if and loop is terminated.
    beginBlock();
    if (utils::unwrapIfSingleStmt(condDiff.getStmt()))
      addToCurrentBlock(condDiff.getStmt());

    Stmt* IfStmt = clad_compat::IfStmt_Create(
        /*Ctx=*/m_Context, /*IL=*/noLoc, /*IsConstexpr=*/false,
        /*Init=*/nullptr, /*Var=*/nullptr,
        /*Cond=*/
        BuildOp(clang::UnaryOperatorKind::UO_LNot, BuildParens(forwardCond)),
        /*LPL=*/noLoc, /*RPL=*/noLoc,
        /*Then=*/breakStmt,
        /*EL=*/noLoc,
        /*Else=*/nullptr);
    addToCurrentBlock(IfStmt);

    Stmt* forwardCondStmts = endBlock();
    if (BodyDiff.getStmt()) {
      BodyDiff.updateStmt(utils::PrependAndCreateCompoundStmt(
          m_Context, BodyDiff.getStmt(), forwardCondStmts));
    } else {
      BodyDiff.updateStmt(utils::unwrapIfSingleStmt(forwardCondStmts));
    }

    Stmt* Forward = new (m_Context)
        ForStmt(m_Context, initResult.getStmt(), nullptr, condVarClone,
                incResult, BodyDiff.getStmt(), noLoc, noLoc, noLoc);

    // Create a condition testing counter for being zero, and its decrement.
    // To match the number of iterations in the forward pass, the reverse loop
    // will look like: for(; Counter; Counter--) ...
    Expr*
        CounterCondition = loopCounter.getCounterConditionResult().get().second;
    Expr* CounterDecrement = loopCounter.getCounterDecrement();

    /// This part adds the reverse pass of loop condition stmt in the body
    beginBlock(direction::reverse);
    Stmt* RevIfStmt = clad_compat::IfStmt_Create(
        /*Ctx=*/m_Context, /*IL=*/noLoc, /*IsConstexpr=*/false,
        /*Init=*/nullptr, /*Var=*/nullptr,
        /*Cond=*/BuildOp(clang::UnaryOperatorKind::UO_LNot, CounterCondition),
        /*LPL=*/noLoc, /*RPL=*/noLoc,
        /*Then=*/Clone(breakStmt),
        /*EL=*/noLoc,
        /*Else=*/nullptr);
    addToCurrentBlock(RevIfStmt, direction::reverse);

    if (condDiff.getStmt_dx()) {
      if (m_CurrentBreakFlagExpr) {
        Expr* loopBreakFlagCond =
            BuildOp(BinaryOperatorKind::BO_LOr,
                    BuildOp(UnaryOperatorKind::UO_LNot, CounterCondition),
                    BuildParens(m_CurrentBreakFlagExpr));
        auto* RevIfStmt = clad_compat::IfStmt_Create(
            m_Context, noLoc, false, nullptr, nullptr, loopBreakFlagCond, noLoc,
            noLoc, condDiff.getStmt_dx(), noLoc, nullptr);
        addToCurrentBlock(RevIfStmt, direction::reverse);
      } else {
        addToCurrentBlock(condDiff.getStmt_dx(), direction::reverse);
      }
    }

    Stmt* revPassCondStmts = endBlock(direction::reverse);
    if (BodyDiff.getStmt_dx()) {
      BodyDiff.updateStmtDx(utils::PrependAndCreateCompoundStmt(
          m_Context, BodyDiff.getStmt_dx(), revPassCondStmts));
    } else {
      BodyDiff.updateStmtDx(utils::unwrapIfSingleStmt(revPassCondStmts));
    }

    Stmt* Reverse = new (m_Context)
        ForStmt(m_Context, nullptr, nullptr, nullptr, CounterDecrement,
                BodyDiff.getStmt_dx(), noLoc, noLoc, noLoc);

    addToCurrentBlock(initResult.getStmt_dx(), direction::reverse);
    addToCurrentBlock(Reverse, direction::reverse);
    Reverse = endBlock(direction::reverse);
    endScope();

    return {utils::unwrapIfSingleStmt(Forward),
            utils::unwrapIfSingleStmt(Reverse)};
  }

  StmtDiff
  ReverseModeVisitor::VisitCXXDefaultArgExpr(const CXXDefaultArgExpr* DE) {
    return Visit(DE->getExpr(), dfdx());
  }

  StmtDiff
  ReverseModeVisitor::VisitCXXBoolLiteralExpr(const CXXBoolLiteralExpr* BL) {
    return Clone(BL);
  }

  StmtDiff
  ReverseModeVisitor::VisitCharacterLiteral(const CharacterLiteral* CL) {
    return Clone(CL);
  }

  StmtDiff ReverseModeVisitor::VisitStringLiteral(const StringLiteral* SL) {
    return StmtDiff(Clone(SL), StringLiteral::Create(
                                   m_Context, "", SL->getKind(), SL->isPascal(),
                                   SL->getType(), utils::GetValidSLoc(m_Sema)));
  }

  StmtDiff ReverseModeVisitor::VisitCXXNullPtrLiteralExpr(
      const CXXNullPtrLiteralExpr* NPE) {
    return StmtDiff(Clone(NPE), Clone(NPE));
  }

  StmtDiff ReverseModeVisitor::VisitReturnStmt(const ReturnStmt* RS) {
    // Initially, df/df = 1.
    const Expr* value = RS->getRetValue();
    QualType type = value->getType();
    auto* dfdf = m_Pullback;
    if (dfdf && (isa<FloatingLiteral>(dfdf) || isa<IntegerLiteral>(dfdf))) {
      ExprResult tmp = dfdf;
      dfdf = m_Sema
                 .ImpCastExprToType(tmp.get(), type,
                                    m_Sema.PrepareScalarCast(tmp, type))
                 .get();
    }
    auto ReturnResult = DifferentiateSingleExpr(value, dfdf);
    StmtDiff ReturnDiff = ReturnResult.first;
    StmtDiff ExprDiff = ReturnResult.second;
    Stmt* Reverse = ReturnDiff.getStmt_dx();
    for (Stmt* S : cast<CompoundStmt>(ReturnDiff.getStmt())->body())
      addToCurrentBlock(S, direction::forward);

    // FIXME: When the return type of a function is a class, ExprDiff.getExpr()
    // returns nullptr, which is a bug. For the time being, the only use case of
    // a return type being class is in pushforwards. Hence a special case has
    // been made to to not do the StoreAndRef operation when return type is
    // ValueAndPushforward.
    if (!utils::IsCladValueAndPushforwardType(type)) {
      if (m_ExternalSource)
        m_ExternalSource->ActBeforeFinalizingVisitReturnStmt(ExprDiff);
    }

    // If this return stmt is the last stmt in the function's body,
    // adding goto will only introduce
    // ```
    // goto _label0; // the forward sweep ends
    // _label0:  // the reverse sweep starts immediately
    // ```
    // Therefore, in this case, we can omit the goto.
    const Stmt* lastFuncStmt = m_DiffReq.Function->getBody();
    if (const auto* CS = dyn_cast<CompoundStmt>(lastFuncStmt))
      lastFuncStmt = *CS->body_rbegin();
    if (RS == lastFuncStmt)
      return {nullptr, Reverse};

    // If the original function returns at this point, some part of the reverse
    // pass (corresponding to other branches that do not return here) must be
    // skipped. We create a label in the reverse pass and jump to it via goto.
    LabelDecl* LD = LabelDecl::Create(m_Context, m_Sema.CurContext, noLoc,
                                      CreateUniqueIdentifier("_label"));
    m_Sema.PushOnScopeChains(LD, m_DerivativeFnScope, true);
    // Attach label to the last Stmt in the corresponding Reverse Stmt.
    if (!Reverse)
      Reverse = m_Sema.ActOnNullStmt(noLoc).get();
    Stmt* LS = m_Sema.ActOnLabelStmt(noLoc, LD, noLoc, Reverse).get();
    addToCurrentBlock(LS, direction::reverse);

    // Create goto to the label.
    return m_Sema.ActOnGotoStmt(noLoc, noLoc, LD).get();
  }

  StmtDiff ReverseModeVisitor::VisitParenExpr(const ParenExpr* PE) {
    StmtDiff subStmtDiff = Visit(PE->getSubExpr(), dfdx());
    return StmtDiff(BuildParens(subStmtDiff.getExpr()),
                    BuildParens(subStmtDiff.getExpr_dx()), nullptr,
                    BuildParens(subStmtDiff.getRevSweepAsExpr()));
  }

  StmtDiff ReverseModeVisitor::VisitInitListExpr(const InitListExpr* ILE) {
    QualType ILEType = ILE->getType();
    llvm::SmallVector<Expr*, 16> clonedExprs(ILE->getNumInits());
    if (isArrayOrPointerType(ILEType)) {
      for (unsigned i = 0, e = ILE->getNumInits(); i < e; i++) {
        Expr* I =
            ConstantFolder::synthesizeLiteral(m_Context.IntTy, m_Context, i);
        Expr* array_at_i = m_Sema
                               .ActOnArraySubscriptExpr(getCurrentScope(),
                                                        dfdx(), noLoc, I, noLoc)
                               .get();
        Expr* clonedEI = Visit(ILE->getInit(i), array_at_i).getExpr();
        clonedExprs[i] = clonedEI;
      }

      Expr* clonedILE = m_Sema.ActOnInitList(noLoc, clonedExprs, noLoc).get();
      return StmtDiff(clonedILE);
    }
    // Check if type is a CXXRecordDecl and a struct.
    if (!utils::IsCladValueAndPushforwardType(ILEType) &&
        ILEType->isRecordType() && ILEType->getAsCXXRecordDecl()->isStruct()) {
      for (unsigned i = 0, e = ILE->getNumInits(); i < e; i++) {
        // fetch ith field of the struct.
        auto field_iterator = ILEType->getAsCXXRecordDecl()->field_begin();
        std::advance(field_iterator, i);
        Expr* member_acess = utils::BuildMemberExpr(
            m_Sema, getCurrentScope(), dfdx(), (*field_iterator)->getName());
        clonedExprs[i] = Visit(ILE->getInit(i), member_acess).getExpr();
      }
      Expr* clonedILE = m_Sema.ActOnInitList(noLoc, clonedExprs, noLoc).get();
      return StmtDiff(clonedILE);
    }

    // FIXME: This is a makeshift arrangement to differentiate an InitListExpr
    // that represents a ValueAndPushforward type. Ideally this must be
    // differentiated at VisitCXXConstructExpr
#ifndef NDEBUG
    bool isValueAndPushforward = utils::IsCladValueAndPushforwardType(ILEType);
    assert(isValueAndPushforward &&
           "Only InitListExpr that represents arrays or ValueAndPushforward "
           "Object initialization is supported");
#endif

    // Here we assume that the adjoint expression of the first element in
    // InitList is dfdx().value and the adjoint for the second element is
    // dfdx().pushforward. At this point the top of the Tape must contain a
    // ValueAndPushforward object that represents derivative of the
    // ValueAndPushforward object returned by the function whose derivative is
    // requested.
    Expr* dValueExpr =
        utils::BuildMemberExpr(m_Sema, getCurrentScope(), dfdx(), "value");
    StmtDiff clonedValueEI = Visit(ILE->getInit(0), dValueExpr).getExpr();
    clonedExprs[0] = clonedValueEI.getExpr();

    Expr* dPushforwardExpr = utils::BuildMemberExpr(m_Sema, getCurrentScope(),
                                                    dfdx(), "pushforward");
    Expr* clonedPushforwardEI =
        Visit(ILE->getInit(1), dPushforwardExpr).getExpr();
    clonedExprs[1] = clonedPushforwardEI;

    Expr* clonedILE = m_Sema.ActOnInitList(noLoc, clonedExprs, noLoc).get();
    return StmtDiff(clonedILE);
  }

  StmtDiff
  ReverseModeVisitor::VisitArraySubscriptExpr(const ArraySubscriptExpr* ASE) {
    auto ASI = SplitArraySubscript(ASE);
    const Expr* Base = ASI.first;
    const auto& Indices = ASI.second;
    StmtDiff BaseDiff = Visit(Base);
    llvm::SmallVector<Expr*, 4> clonedIndices(Indices.size());
    llvm::SmallVector<Expr*, 4> reverseIndices(Indices.size());
    llvm::SmallVector<Expr*, 4> forwSweepDerivativeIndices(Indices.size());
    for (std::size_t i = 0; i < Indices.size(); i++) {
      // FIXME: Remove redundant indices vectors.
      StmtDiff IdxDiff = Visit(Indices[i]);
      clonedIndices[i] = Clone(IdxDiff.getExpr());
      reverseIndices[i] = Clone(IdxDiff.getExpr());
      forwSweepDerivativeIndices[i] = IdxDiff.getExpr();
    }
    auto* cloned = BuildArraySubscript(BaseDiff.getExpr(), clonedIndices);
    auto* valueForRevSweep =
        BuildArraySubscript(BaseDiff.getExpr(), reverseIndices);
    Expr* target = BaseDiff.getExpr_dx();
    if (!target)
      return cloned;
    Expr* result = nullptr;
    Expr* forwSweepDerivative = nullptr;
    // Create the target[idx] expression.
    result = BuildArraySubscript(target, reverseIndices);
    forwSweepDerivative =
        BuildArraySubscript(target, forwSweepDerivativeIndices);
    // Create the (target += dfdx) statement.
    if (dfdx()) {
      if (shouldUseCudaAtomicOps(target)) {
        Expr* atomicCall = BuildCallToCudaAtomicAdd(result, dfdx());
        // Add it to the body statements.
        addToCurrentBlock(atomicCall, direction::reverse);
      } else {
        auto* add_assign = BuildOp(BO_AddAssign, result, dfdx());
        // Add it to the body statements.
        addToCurrentBlock(add_assign, direction::reverse);
      }
    }
    if (m_ExternalSource)
      m_ExternalSource->ActAfterProcessingArraySubscriptExpr(valueForRevSweep);
    return StmtDiff(cloned, result, forwSweepDerivative, valueForRevSweep);
  }

  StmtDiff ReverseModeVisitor::VisitDeclRefExpr(const DeclRefExpr* DRE) {
    Expr* clonedDRE = Clone(DRE);
    // Check if referenced Decl was "replaced" with another identifier inside
    // the derivative
    if (auto* VD = dyn_cast<VarDecl>(cast<DeclRefExpr>(clonedDRE)->getDecl())) {
      // If current context is different than the context of the original
      // declaration (e.g. we are inside lambda), rebuild the DeclRefExpr
      // with Sema::BuildDeclRefExpr. This is required in some cases, e.g.
      // Sema::BuildDeclRefExpr is responsible for adding captured fields
      // to the underlying struct of a lambda.
      if (VD->getDeclContext() != m_Sema.CurContext) {
        auto* ccDRE = dyn_cast<DeclRefExpr>(clonedDRE);
        NestedNameSpecifier* NNS = DRE->getQualifier();
        auto* referencedDecl = cast<VarDecl>(ccDRE->getDecl());
        clonedDRE = BuildDeclRef(referencedDecl, NNS, DRE->getValueKind());
      }
      // This case happens when ref-type variables have to become function
      // global. Ref-type declarations cannot be moved to the function global
      // scope because they can't be separated from their inits.
      if (DRE->getDecl()->getType()->isReferenceType() &&
          VD->getType()->isPointerType())
        clonedDRE = BuildOp(UO_Deref, clonedDRE);
      if (m_DiffReq.Mode == DiffMode::jacobian) {
        if (m_VectorOutput.size() <= outputArrayCursor)
          return StmtDiff(clonedDRE);

        auto it = m_VectorOutput[outputArrayCursor].find(VD);
        if (it == std::end(m_VectorOutput[outputArrayCursor]))
          return StmtDiff(clonedDRE); // Not an independent variable, ignored.

        // Create the (jacobianMatrix[idx] += dfdx) statement.
        if (dfdx()) {
          auto* add_assign = BuildOp(BO_AddAssign, it->second, dfdx());
          // Add it to the body statements.
          addToCurrentBlock(add_assign, direction::reverse);
        }
        return StmtDiff(clonedDRE, it->second, it->second);
      } else {
        // Check DeclRefExpr is a reference to an independent variable.
        auto it = m_Variables.find(VD);
        if (it == std::end(m_Variables)) {
          // Is not an independent variable, ignored.
          return StmtDiff(clonedDRE);
        }
        // Create the (_d_param[idx] += dfdx) statement.
        if (dfdx()) {
          // FIXME: not sure if this is generic.
          // Don't update derivatives of record types.
          if (!VD->getType()->isRecordType()) {
            Expr* base = it->second;
            if (auto* UO = dyn_cast<UnaryOperator>(it->second))
              base = UO->getSubExpr()->IgnoreImpCasts();
            if (shouldUseCudaAtomicOps(base)) {
              Expr* atomicCall = BuildCallToCudaAtomicAdd(it->second, dfdx());
              // Add it to the body statements.
              addToCurrentBlock(atomicCall, direction::reverse);
            } else {
              auto* add_assign = BuildOp(BO_AddAssign, it->second, dfdx());
              addToCurrentBlock(add_assign, direction::reverse);
            }
          }
        }
        return StmtDiff(clonedDRE, it->second, it->second);
      }
    }

    return StmtDiff(clonedDRE);
  }

  StmtDiff ReverseModeVisitor::VisitIntegerLiteral(const IntegerLiteral* IL) {
    auto* Constant0 =
        ConstantFolder::synthesizeLiteral(m_Context.IntTy, m_Context, 0);
    return StmtDiff(Clone(IL), Constant0);
  }

  StmtDiff ReverseModeVisitor::VisitFloatingLiteral(const FloatingLiteral* FL) {
    return StmtDiff(Clone(FL));
  }

  StmtDiff ReverseModeVisitor::VisitCallExpr(const CallExpr* CE) {
    const FunctionDecl* FD = CE->getDirectCallee();
    if (!FD) {
      diag(DiagnosticsEngine::Warning,
           CE->getEndLoc(),
           "Differentiation of only direct calls is supported. Ignored");
      return StmtDiff(Clone(CE));
    }

    // If the function is non_differentiable, return zero derivative.
    if (clad::utils::hasNonDifferentiableAttribute(CE)) {
      // Calling the function without computing derivatives
      llvm::SmallVector<Expr*, 4> ClonedArgs;
      for (unsigned i = 0, e = CE->getNumArgs(); i < e; ++i)
        ClonedArgs.push_back(Clone(CE->getArg(i)));

      SourceLocation validLoc = clad::utils::GetValidSLoc(m_Sema);
      Expr* Call = m_Sema
                       .ActOnCallExpr(getCurrentScope(), Clone(CE->getCallee()),
                                      validLoc, ClonedArgs, validLoc)
                       .get();
      // Creating a zero derivative
      auto* zero = ConstantFolder::synthesizeLiteral(m_Context.IntTy, m_Context,
                                                     /*val=*/0);

      // Returning the function call and zero derivative
      return StmtDiff(Call, zero);
    }

    // begin and end are common enough to have a more efficient and nice-looking
    // special case. Instead of _forw and a useless _pullback functions, we can
    // express the result in terms of the same std::begin / std::end. Note:
    // since std::initializer_list is replaced with clad::array, this is the
    // simplest way to support begin/end functions of the former and not deal
    // with the type mismatch.
    std::string FDName = FD->getNameAsString();
    if (FDName == "begin" || FDName == "end") {
      const Expr* arg = nullptr;
      if (const auto* MCE = dyn_cast<CXXMemberCallExpr>(CE))
        arg = MCE->getImplicitObjectArgument();
      else
        arg = CE->getArg(0);
      if (const auto* CXXCE = dyn_cast<CXXConstructExpr>(arg))
        arg = CXXCE->getArg(0);
      StmtDiff argDiff = Visit(arg);
      llvm::SmallVector<Expr*, 1> params{argDiff.getExpr()};
      llvm::SmallVector<Expr*, 1> paramsDiff{argDiff.getExpr_dx()};
      Expr* call = GetFunctionCall(FDName, "std", params);
      Expr* callDiff = GetFunctionCall(FDName, "std", paramsDiff);
      return {call, callDiff};
    }

    auto NArgs = FD->getNumParams();
    // If the function has no args and is not a member function call then we
    // assume that it is not related to independent variables and does not
    // contribute to gradient.
    if ((NArgs == 0U) && !isa<CXXMemberCallExpr>(CE) &&
        !isa<CXXOperatorCallExpr>(CE))
      return StmtDiff(Clone(CE));

    // If all arguments are constant literals, then this does not contribute to
    // the gradient.
    // FIXME: revert this when this is integrated in the activity analysis pass.
    if (!isa<CXXMemberCallExpr>(CE) && !isa<CXXOperatorCallExpr>(CE)) {
      bool allArgsAreConstantLiterals = true;
      for (const Expr* arg : CE->arguments()) {
        // if it's of type MaterializeTemporaryExpr, then check its
        // subexpression.
        if (const auto* MTE = dyn_cast<MaterializeTemporaryExpr>(arg))
          arg = clad_compat::GetSubExpr(MTE)->IgnoreImpCasts();
        if (!arg->isEvaluatable(m_Context)) {
          allArgsAreConstantLiterals = false;
          break;
        }
      }
      if (allArgsAreConstantLiterals)
        return StmtDiff(Clone(CE), Clone(CE));
    }

    SourceLocation Loc = CE->getExprLoc();

    // Stores the call arguments for the function to be derived
    llvm::SmallVector<Expr*, 16> CallArgs{};
    // Stores the dx of the call arguments for the function to be derived
    llvm::SmallVector<Expr*, 16> CallArgDx{};
    // Stores the call arguments for the derived function
    llvm::SmallVector<Expr*, 16> DerivedCallArgs{};
    // Stores tape decl and pushes for multiarg numerically differentiated
    // calls.
    llvm::SmallVector<Stmt*, 16> PostCallStmts{};

    // For calls to C-style memory allocation functions, we do not need to
    // differentiate the call. We just need to visit the arguments to the
    // function.
    if (utils::IsMemoryFunction(FD)) {
      for (const Expr* Arg : CE->arguments()) {
        StmtDiff ArgDiff = Visit(Arg, dfdx());
        CallArgs.push_back(ArgDiff.getExpr());
        if (Arg->getType()->isPointerType())
          DerivedCallArgs.push_back(ArgDiff.getExpr_dx());
        else
          DerivedCallArgs.push_back(ArgDiff.getExpr());
      }
      Expr* call =
          m_Sema
              .ActOnCallExpr(getCurrentScope(), Clone(CE->getCallee()), Loc,
                             llvm::MutableArrayRef<Expr*>(CallArgs), Loc)
              .get();
      Expr* call_dx =
          m_Sema
              .ActOnCallExpr(getCurrentScope(), Clone(CE->getCallee()), Loc,
                             llvm::MutableArrayRef<Expr*>(DerivedCallArgs), Loc)
              .get();
      return StmtDiff(call, call_dx);
    }
    // For calls to C-style memory deallocation functions, we do not need to
    // differentiate the call. We just need to visit the arguments to the
    // function. Also, don't add any statements either in forward or reverse
    // pass. Instead, add it in m_DeallocExprs.
    if (utils::IsMemoryDeallocationFunction(FD)) {
      for (const Expr* Arg : CE->arguments()) {
        StmtDiff ArgDiff = Visit(Arg, dfdx());
        CallArgs.push_back(ArgDiff.getExpr());
        if (auto* DRE = dyn_cast<DeclRefExpr>(ArgDiff.getExpr())) {
          // If the arg is used for differentiation of the function, then we
          // cannot free it in the end as it's the result to be returned to the
          // user.
          if (m_ParamVarsWithDiff.find(DRE->getDecl()) ==
              m_ParamVarsWithDiff.end())
            DerivedCallArgs.push_back(ArgDiff.getExpr_dx());
        }
      }
      Expr* call =
          m_Sema
              .ActOnCallExpr(getCurrentScope(), Clone(CE->getCallee()), Loc,
                             llvm::MutableArrayRef<Expr*>(CallArgs), Loc)
              .get();
      m_DeallocExprs.push_back(call);

      if (!DerivedCallArgs.empty()) {
        Expr* call_dx =
            m_Sema
                .ActOnCallExpr(getCurrentScope(), Clone(CE->getCallee()), Loc,
                               llvm::MutableArrayRef<Expr*>(DerivedCallArgs),
                               Loc)
                .get();
        m_DeallocExprs.push_back(call_dx);
      }
      return StmtDiff();
    }

    // If the result does not depend on the result of the call, just clone
    // the call and visit arguments (since they may contain side-effects like
    // f(x = y))
    // If the callee function takes arguments by reference then it can affect
    // derivatives even if there is no `dfdx()` and thus we should call the
    // derived function. In the case of member functions, `implicit`
    // this object is always passed by reference.
    if (!dfdx() && !utils::HasAnyReferenceOrPointerArgument(FD) &&
        !isa<CXXMemberCallExpr>(CE) && !isa<CXXOperatorCallExpr>(CE)) {
      for (const Expr* Arg : CE->arguments()) {
        StmtDiff ArgDiff = Visit(Arg, dfdx());
        CallArgs.push_back(ArgDiff.getExpr());
      }
      Expr* call =
          m_Sema
              .ActOnCallExpr(getCurrentScope(), Clone(CE->getCallee()), Loc,
                             llvm::MutableArrayRef<Expr*>(CallArgs), Loc)
              .get();
      return call;
    }

    llvm::SmallVector<Stmt*, 16> PreCallStmts{};
    // Save current index in the current block, to potentially put some
    // statements there later.
    std::size_t insertionPoint = getCurrentBlock(direction::reverse).size();

    const auto* MD = dyn_cast<CXXMethodDecl>(FD);
    // Method operators have a base like methods do but it's included in the
    // call arguments so we have to shift the indexing of call arguments.
    bool isMethodOperatorCall = MD && isa<CXXOperatorCallExpr>(CE);

    for (std::size_t i = static_cast<std::size_t>(isMethodOperatorCall),
                     e = CE->getNumArgs();
         i != e; ++i) {
      const Expr* arg = CE->getArg(i);
      const auto* PVD = FD->getParamDecl(
          i - static_cast<unsigned long>(isMethodOperatorCall));
      StmtDiff argDiff{};
      // We do not need to create result arg for arguments passed by reference
      // because the derivatives of arguments passed by reference are directly
      // modified by the derived callee function.
      if (utils::IsReferenceOrPointerArg(arg) ||
          !m_DiffReq.shouldHaveAdjoint(PVD)) {
        argDiff = Visit(arg);
        CallArgDx.push_back(argDiff.getExpr_dx());
      } else {
        // Create temporary variables corresponding to derivative of each
        // argument, so that they can be referred to when arguments is visited.
        // Variables will be initialized later after arguments is visited. This
        // is done to reduce cloning complexity and only clone once. The type is
        // same as the call expression as it is the type used to declare the
        // _gradX array
        QualType dArgTy = getNonConstType(arg->getType(), m_Context, m_Sema);
        VarDecl* dArgDecl = BuildVarDecl(dArgTy, "_r", getZeroInit(dArgTy));
        PreCallStmts.push_back(BuildDeclStmt(dArgDecl));
        CallArgDx.push_back(BuildDeclRef(dArgDecl));
        // Visit using uninitialized reference.
        argDiff = Visit(arg, BuildDeclRef(dArgDecl));
      }

      // Save cloned arg in a "global" variable, so that it is accessible from
      // the reverse pass.
      // FIXME: At this point, we assume all the variables passed by reference
      // may be changed since we have no way to determine otherwise.
      // FIXME: We cannot use GlobalStoreAndRef to store a whole array so now
      // arrays are not stored.
      QualType paramTy = PVD->getType();
      bool passByRef = paramTy->isLValueReferenceType() &&
                       !paramTy.getNonReferenceType().isConstQualified();
      Expr* argDiffStore;
      if (passByRef && !argDiff.getExpr()->isEvaluatable(m_Context))
        argDiffStore =
            GlobalStoreAndRef(argDiff.getExpr(), "_t", /*force=*/true);
      else
        argDiffStore = argDiff.getExpr();

      // We need to pass the actual argument in the cloned call expression,
      // instead of a temporary, for arguments passed by reference. This is
      // because, callee function may modify the argument passed as reference
      // and if we use a temporary variable then the effect of the modification
      // will be lost.
      // For example:
      // ```
      // // original statements
      // modify(a); // a is passed by reference
      // modify(a); // a is passed by reference
      //
      // // forward pass
      // _t0 = a;
      // modify(_t0); // _t0 is modified instead of a
      // _t1 = a; // stale value of a is being used here
      // modify(_t1);
      //
      // // correct forward pass
      // _t0 = a;
      // modify(a);
      // _t1 = a;
      // modify(a);
      // ```
      // FIXME: We cannot use GlobalStoreAndRef to store a whole array so now
      // arrays are not stored.
      if (passByRef) {
        // Restore args
        Stmts& block = getCurrentBlock(direction::reverse);
        Expr* op = BuildOp(BinaryOperatorKind::BO_Assign, argDiff.getExpr(),
                           argDiffStore);
        block.insert(block.begin() + insertionPoint, op);
        // We added restoration of the original arg. Thus we need to
        // correspondingly adjust the insertion point.
        insertionPoint += 1;
      }
      CallArgs.push_back(argDiff.getExpr());
      DerivedCallArgs.push_back(argDiffStore);
    }

    Expr* OverloadedDerivedFn = nullptr;
    // If the function has a single arg and does not return a reference or take
    // arg by reference, we look for a derivative w.r.t. to this arg using the
    // forward mode(it is unlikely that we need gradient of a one-dimensional
    // function).
    bool asGrad = true;

    if (NArgs == 1 && !utils::HasAnyReferenceOrPointerArgument(FD) &&
        !isa<CXXMethodDecl>(FD)) {
      std::string customPushforward = FD->getNameAsString() + "_pushforward";
      auto pushforwardCallArgs = DerivedCallArgs;
      pushforwardCallArgs.push_back(ConstantFolder::synthesizeLiteral(
          DerivedCallArgs.front()->getType(), m_Context, 1));
      OverloadedDerivedFn =
          m_Builder.BuildCallToCustomDerivativeOrNumericalDiff(
              customPushforward, pushforwardCallArgs, getCurrentScope(),
              const_cast<DeclContext*>(FD->getDeclContext()));
      if (OverloadedDerivedFn)
        asGrad = false;
    }
    // Store all the derived call output args (if any)
    llvm::SmallVector<Expr*, 16> DerivedCallOutputArgs{};
    // It is required because call to numerical diff and reverse mode diff
    // requires (slightly) different arguments.
    llvm::SmallVector<Expr*, 16> pullbackCallArgs{};

    // Stores differentiation result of implicit `this` object, if any.
    StmtDiff baseDiff;
    Expr* baseExpr = nullptr;
    // If it has more args or f_darg0 was not found, we look for its pullback
    // function.
    std::vector<size_t> globalCallArgs;
    if (!OverloadedDerivedFn) {
      size_t idx = 0;

      /// Add base derivative expression in the derived call output args list if
      /// `CE` is a call to an instance member function.
      if (MD) {
        if (isLambdaCallOperator(MD)) {
          QualType ptrType = m_Context.getPointerType(m_Context.getRecordType(
              FD->getDeclContext()->getOuterLexicalRecordContext()));
          baseDiff =
              StmtDiff(Clone(dyn_cast<CXXOperatorCallExpr>(CE)->getArg(0)),
                       new (m_Context) CXXNullPtrLiteralExpr(ptrType, Loc));
        } else if (MD->isInstance()) {
          const Expr* baseOriginalE = nullptr;
          if (const auto* MCE = dyn_cast<CXXMemberCallExpr>(CE))
            baseOriginalE = MCE->getImplicitObjectArgument();
          else if (const auto* OCE = dyn_cast<CXXOperatorCallExpr>(CE))
            baseOriginalE = OCE->getArg(0);

          baseDiff = Visit(baseOriginalE);
          baseExpr = baseDiff.getExpr();
          Expr* baseDiffStore = GlobalStoreAndRef(baseDiff.getExpr());
          baseDiff.updateStmt(baseDiffStore);
          Expr* baseDerivative = baseDiff.getExpr_dx();
          if (!baseDerivative->getType()->isPointerType())
            baseDerivative =
                BuildOp(UnaryOperatorKind::UO_AddrOf, baseDerivative);
          DerivedCallOutputArgs.push_back(baseDerivative);
        }
      }

      for (auto* argDerivative : CallArgDx) {
        Expr* gradArgExpr = nullptr;
        QualType paramTy = FD->getParamDecl(idx)->getType();
        if (!argDerivative || utils::isArrayOrPointerType(paramTy) ||
            isCladArrayType(argDerivative->getType()))
          gradArgExpr = argDerivative;
        else
          gradArgExpr =
              BuildOp(UO_AddrOf, argDerivative, m_DiffReq->getLocation());
        DerivedCallOutputArgs.push_back(gradArgExpr);
        idx++;
      }
      Expr* pullback = dfdx();

      if ((pullback == nullptr) && FD->getReturnType()->isLValueReferenceType())
        pullback = getZeroInit(FD->getReturnType().getNonReferenceType());

      if (FD->getReturnType()->isVoidType()) {
        assert(pullback == nullptr && FD->getReturnType()->isVoidType() &&
               "Call to function returning void type should not have any "
               "corresponding dfdx().");
      }

      for (Expr* arg : DerivedCallOutputArgs)
        if (arg)
          DerivedCallArgs.push_back(arg);
      pullbackCallArgs = DerivedCallArgs;

      if (pullback)
        pullbackCallArgs.insert(pullbackCallArgs.begin() + CE->getNumArgs() -
                                    static_cast<int>(isMethodOperatorCall),
                                pullback);

      // Try to find it in builtin derivatives
      std::string customPullback =
          clad::utils::ComputeEffectiveFnName(FD) + "_pullback";
      // Add the indexes of the global args to the custom pullback name
      if (!m_CUDAGlobalArgs.empty())
        for (size_t i = 0; i < pullbackCallArgs.size(); i++)
          if (auto* DRE = dyn_cast<DeclRefExpr>(pullbackCallArgs[i]))
            if (auto* param = dyn_cast<ParmVarDecl>(DRE->getDecl()))
              if (m_CUDAGlobalArgs.find(param) != m_CUDAGlobalArgs.end()) {
                customPullback += "_" + std::to_string(i);
                globalCallArgs.emplace_back(i);
              }

      if (baseDiff.getExpr())
        pullbackCallArgs.insert(
            pullbackCallArgs.begin(),
            BuildOp(UnaryOperatorKind::UO_AddrOf, baseDiff.getExpr()));

      OverloadedDerivedFn =
          m_Builder.BuildCallToCustomDerivativeOrNumericalDiff(
              customPullback, pullbackCallArgs, getCurrentScope(),
              const_cast<DeclContext*>(FD->getDeclContext()));
      if (baseDiff.getExpr())
        pullbackCallArgs.erase(pullbackCallArgs.begin());
    }

    // Derivative was not found, check if it is a recursive call
    if (!OverloadedDerivedFn) {
      if (FD == m_DiffReq.Function &&
          m_DiffReq.Mode == DiffMode::experimental_pullback) {
        // Recursive call.
        Expr* selfRef =
            m_Sema
                .BuildDeclarationNameExpr(
                    CXXScopeSpec(), m_Derivative->getNameInfo(), m_Derivative)
                .get();

        OverloadedDerivedFn = m_Sema
                                  .ActOnCallExpr(getCurrentScope(), selfRef,
                                                 Loc, pullbackCallArgs, Loc)
                                  .get();
      } else {
        if (m_ExternalSource)
          m_ExternalSource->ActBeforeDifferentiatingCallExpr(
              pullbackCallArgs, PreCallStmts, dfdx());

        // Overloaded derivative was not found, request the CladPlugin to
        // derive the called function.
        DiffRequest pullbackRequest{};
        pullbackRequest.Function = FD;

        // Mark the indexes of the global args. Necessary if the argument of the
        // call has a different name than the function's signature parameter.
        pullbackRequest.CUDAGlobalArgsIndexes = globalCallArgs;

        pullbackRequest.BaseFunctionName =
            clad::utils::ComputeEffectiveFnName(FD);
        pullbackRequest.Mode = DiffMode::experimental_pullback;
        // Silence diag outputs in nested derivation process.
        pullbackRequest.VerboseDiags = false;
        pullbackRequest.EnableTBRAnalysis = m_DiffReq.EnableTBRAnalysis;
        pullbackRequest.EnableVariedAnalysis = m_DiffReq.EnableVariedAnalysis;
        bool isaMethod = isa<CXXMethodDecl>(FD);
        for (size_t i = 0, e = FD->getNumParams(); i < e; ++i)
          if (MD && isLambdaCallOperator(MD)) {
            if (const auto* paramDecl = FD->getParamDecl(i))
              pullbackRequest.DVI.push_back(paramDecl);
          } else if (DerivedCallOutputArgs[i + isaMethod])
            pullbackRequest.DVI.push_back(FD->getParamDecl(i));

        FunctionDecl* pullbackFD = nullptr;
        if (m_ExternalSource)
          // FIXME: Error estimation currently uses singleton objects -
          // m_ErrorEstHandler and m_EstModel, which is cleared after each
          // error_estimate request. This requires the pullback to be derived
          // at the same time to access the singleton objects.
          pullbackFD =
              plugin::ProcessDiffRequest(m_CladPlugin, pullbackRequest);
        else
          pullbackFD = m_Builder.HandleNestedDiffRequest(pullbackRequest);

        // Clad failed to derive it.
        // FIXME: Add support for reference arguments to the numerical diff. If
        // it already correctly support reference arguments then confirm the
        // support and add tests for the same.
        if (!pullbackFD && !utils::HasAnyReferenceOrPointerArgument(FD) &&
            !isa<CXXMethodDecl>(FD)) {
          // Try numerically deriving it.
          if (NArgs == 1) {
            OverloadedDerivedFn = GetSingleArgCentralDiffCall(
                Clone(CE->getCallee()), DerivedCallArgs[0],
                /*targetPos=*/0,
                /*numArgs=*/1, DerivedCallArgs);
            asGrad = !OverloadedDerivedFn;
          } else {
            auto CEType = getNonConstType(CE->getType(), m_Context, m_Sema);
            OverloadedDerivedFn = GetMultiArgCentralDiffCall(
                Clone(CE->getCallee()), CEType.getCanonicalType(),
                CE->getNumArgs(), dfdx(), PreCallStmts, PostCallStmts,
                DerivedCallArgs, CallArgDx);
          }
          CallExprDiffDiagnostics(FD, CE->getBeginLoc());
          if (!OverloadedDerivedFn) {
            Stmts& block = getCurrentBlock(direction::reverse);
            block.insert(block.begin(), PreCallStmts.begin(),
                         PreCallStmts.end());
            return StmtDiff(Clone(CE));
          }
        } else if (pullbackFD) {
          if (baseDiff.getExpr()) {
            Expr* baseE = baseDiff.getExpr();
            OverloadedDerivedFn = BuildCallExprToMemFn(
                baseE, pullbackFD->getName(), pullbackCallArgs, Loc);
          } else {
            OverloadedDerivedFn =
                m_Sema
                    .ActOnCallExpr(getCurrentScope(), BuildDeclRef(pullbackFD),
                                   Loc, pullbackCallArgs, Loc)
                    .get();
          }
        }
      }
    }

    if (OverloadedDerivedFn) {
      // Derivative was found.
      FunctionDecl* fnDecl = dyn_cast<CallExpr>(OverloadedDerivedFn)
                                 ->getDirectCallee();
      // Put Result array declaration in the function body.
      // Call the gradient, passing Result as the last Arg.
      Stmts& block = getCurrentBlock(direction::reverse);
      Stmts::iterator it = std::begin(block) + insertionPoint;
      // Insert PreCallStmts
      it = block.insert(it, PreCallStmts.begin(), PreCallStmts.end());
      it += PreCallStmts.size();
      if (!asGrad) {
        if (utils::IsCladValueAndPushforwardType(fnDecl->getReturnType()))
          OverloadedDerivedFn = utils::BuildMemberExpr(
              m_Sema, getCurrentScope(), OverloadedDerivedFn, "pushforward");
        // If the derivative is called through _darg0 instead of _grad.
        Expr* d = BuildOp(BO_Mul, dfdx(), OverloadedDerivedFn);
        Expr* addGrad = BuildOp(BO_AddAssign, Clone(CallArgDx[0]), d);
        it = block.insert(it, addGrad);
        it++;
      } else {
        // Insert the CallExpr to the derived function
        it = block.insert(it, OverloadedDerivedFn);
        it++;
      }
      // Insert PostCallStmts
      block.insert(it, PostCallStmts.begin(), PostCallStmts.end());
    }
    if (m_ExternalSource)
      m_ExternalSource->ActBeforeFinalizingVisitCallExpr(
          CE, OverloadedDerivedFn, DerivedCallArgs, CallArgDx, asGrad);

    Expr* call = nullptr;

    QualType returnType = FD->getReturnType();
    if (baseDiff.getExpr_dx() &&
        !baseDiff.getExpr_dx()->getType()->isPointerType())
      CallArgDx.insert(CallArgDx.begin(), BuildOp(UnaryOperatorKind::UO_AddrOf,
                                                  baseDiff.getExpr_dx(), Loc));

    if (Expr* customForwardPassCE =
            BuildCallToCustomForwPassFn(FD, CallArgs, CallArgDx, baseExpr)) {
      if (!utils::isNonConstReferenceType(returnType) &&
          !returnType->isPointerType())
        return StmtDiff{customForwardPassCE};
      auto* callRes = StoreAndRef(customForwardPassCE);
      auto* resValue =
          utils::BuildMemberExpr(m_Sema, getCurrentScope(), callRes, "value");
      auto* resAdjoint =
          utils::BuildMemberExpr(m_Sema, getCurrentScope(), callRes, "adjoint");
      return StmtDiff(resValue, resAdjoint, resAdjoint);
    }
    if (utils::isNonConstReferenceType(returnType) ||
        returnType->isPointerType()) {
      DiffRequest calleeFnForwPassReq;
      calleeFnForwPassReq.Function = FD;
      calleeFnForwPassReq.Mode = DiffMode::reverse_mode_forward_pass;
      calleeFnForwPassReq.BaseFunctionName =
          clad::utils::ComputeEffectiveFnName(FD);
      calleeFnForwPassReq.VerboseDiags = true;

      FunctionDecl* calleeFnForwPassFD =
          m_Builder.HandleNestedDiffRequest(calleeFnForwPassReq);

      assert(calleeFnForwPassFD &&
             "Clad failed to generate callee function forward pass function");

      // FIXME: We are using the derivatives in forward pass here
      // If `expr_dx()` is only meant to be used in reverse pass,
      // (for example, `clad::pop(...)` expression and a corresponding
      // `clad::push(...)` in the forward pass), then this can result in
      // incorrect derivative or crash at runtime. Ideally, we should have
      // a separate routine to use derivative in the forward pass.

      // We cannot reuse the derivatives previously computed because
      // they might contain 'clad::pop(..)` expression.
      if (baseDiff.getExpr_dx()) {
        Expr* derivedBase = baseDiff.getExpr_dx();
        // FIXME: We may need this if-block once we support pointers, and
        // passing pointers-by-reference if
        // (isCladArrayType(derivedBase->getType()))
        //   CallArgs.push_back(derivedBase);
        // else
        // Currently derivedBase `*d_this` can never be CladArrayType
        CallArgs.push_back(
            BuildOp(UnaryOperatorKind::UO_AddrOf, derivedBase, Loc));
      }

      for (std::size_t i = static_cast<std::size_t>(isMethodOperatorCall),
                       e = CE->getNumArgs();
           i != e; ++i) {
        const Expr* arg = CE->getArg(i);
        StmtDiff argDiff = Visit(arg);
        // Has to be removed once nondifferentiable arguments are handeled
        if (argDiff.getStmt_dx())
          CallArgs.push_back(argDiff.getExpr_dx());
        else
          CallArgs.push_back(getZeroInit(arg->getType()));
      }
      if (Expr* baseE = baseDiff.getExpr()) {
        call = BuildCallExprToMemFn(baseE, calleeFnForwPassFD->getName(),
                                    CallArgs, Loc);
      } else {
        call = m_Sema
                   .ActOnCallExpr(getCurrentScope(),
                                  BuildDeclRef(calleeFnForwPassFD), Loc,
                                  CallArgs, Loc)
                   .get();
      }
      auto* callRes = StoreAndRef(call);
      auto* resValue =
          utils::BuildMemberExpr(m_Sema, getCurrentScope(), callRes, "value");
      auto* resAdjoint =
          utils::BuildMemberExpr(m_Sema, getCurrentScope(), callRes, "adjoint");
      return StmtDiff(resValue, resAdjoint, resAdjoint);
    } // Recreate the original call expression.

    if (isMethodOperatorCall) {
      const auto* OCE = cast<CXXOperatorCallExpr>(CE);
      auto* FD = const_cast<CXXMethodDecl*>(
          dyn_cast<CXXMethodDecl>(OCE->getCalleeDecl()));

      NestedNameSpecifierLoc NNS(FD->getQualifier(),
                                 /*Data=*/nullptr);
      auto DAP = DeclAccessPair::make(FD, FD->getAccess());
      auto* memberExpr = MemberExpr::Create(
          m_Context, Clone(OCE->getArg(0)), /*isArrow=*/false, Loc, NNS, noLoc,
          FD, DAP, FD->getNameInfo(),
          /*TemplateArgs=*/nullptr, m_Context.BoundMemberTy,
          CLAD_COMPAT_ExprValueKind_R_or_PR_Value,
          ExprObjectKind::OK_Ordinary CLAD_COMPAT_CLANG9_MemberExpr_ExtraParams(
              NOUR_None));
      call = m_Sema
                 .BuildCallToMemberFunction(getCurrentScope(), memberExpr, Loc,
                                            CallArgs, Loc)
                 .get();
      return StmtDiff(call);
    }

    call = m_Sema
               .ActOnCallExpr(getCurrentScope(), Clone(CE->getCallee()), Loc,
                              CallArgs, Loc)
               .get();
    return StmtDiff(call);
  }

  Expr* ReverseModeVisitor::GetMultiArgCentralDiffCall(
      Expr* targetFuncCall, QualType retType, unsigned numArgs, Expr* dfdx,
      llvm::SmallVectorImpl<Stmt*>& PreCallStmts,
      llvm::SmallVectorImpl<Stmt*>& PostCallStmts,
      llvm::SmallVectorImpl<Expr*>& args,
      llvm::SmallVectorImpl<Expr*>& outputArgs) {
    int printErrorInf = m_Builder.shouldPrintNumDiffErrs();
    llvm::SmallVector<Expr*, 16U> NumDiffArgs = {};
    NumDiffArgs.push_back(targetFuncCall);
    // build the output array declaration.
    Expr* size =
        ConstantFolder::synthesizeLiteral(m_Context.IntTy, m_Context, numArgs);
    QualType GradType = clad_compat::getConstantArrayType(
        m_Context, retType,
        llvm::APInt(m_Context.getTargetInfo().getIntWidth(), numArgs),
        /*SizeExpr=*/size,
        /*ASM=*/clad_compat::ArraySizeModifier_Normal,
        /*IndexTypeQuals*/ 0);
    Expr* zero =
        ConstantFolder::synthesizeLiteral(m_Context.IntTy, m_Context, 0);
    Expr* init = m_Sema.ActOnInitList(noLoc, {zero}, noLoc).get();
    auto* VD = BuildVarDecl(GradType, "_grad", init);

    PreCallStmts.push_back(BuildDeclStmt(VD));
    NumDiffArgs.push_back(BuildDeclRef(VD));
    NumDiffArgs.push_back(ConstantFolder::synthesizeLiteral(
        m_Context.IntTy, m_Context, printErrorInf));

    // Build the tape push expressions.
    VD->setLocation(m_DiffReq->getLocation());
    for (unsigned i = 0, e = numArgs; i < e; i++) {
      Expr* gradRef = BuildDeclRef(VD);
      Expr* idx =
          ConstantFolder::synthesizeLiteral(m_Context.IntTy, m_Context, i);
      Expr* gradElem = BuildArraySubscript(gradRef, {idx});
      Expr* gradExpr = BuildOp(BO_Mul, dfdx, gradElem);
      // Inputs were not pointers, so the output args are not in global GPU
      // memory. Hence, no need to use atomic ops.
      PostCallStmts.push_back(BuildOp(BO_AddAssign, outputArgs[i], gradExpr));
      NumDiffArgs.push_back(args[i]);
    }
    std::string Name = "central_difference";
    return m_Builder.BuildCallToCustomDerivativeOrNumericalDiff(
        Name, NumDiffArgs, getCurrentScope(),
        /*OriginalFnDC=*/nullptr,
        /*forCustomDerv=*/false,
        /*namespaceShouldExist=*/false);
  }

  StmtDiff ReverseModeVisitor::VisitUnaryOperator(const UnaryOperator* UnOp) {
    auto opCode = UnOp->getOpcode();
    Expr* valueForRevPass = nullptr;
    StmtDiff diff{};
    Expr* E = UnOp->getSubExpr();
    // If it is a post-increment/decrement operator, its result is a reference
    // and we should return it.
    Expr* ResultRef = nullptr;

    // For increment/decrement of pointer, perform the same on the
    // derivative pointer also.
    bool isPointerOp = E->getType()->isPointerType();

    if (opCode == UO_Plus)
      // xi = +xj
      // dxi/dxj = +1.0
      // df/dxj += df/dxi * dxi/dxj = df/dxi
      diff = Visit(E, dfdx());
    else if (opCode == UO_Minus) {
      // xi = -xj
      // dxi/dxj = -1.0
      // df/dxj += df/dxi * dxi/dxj = -df/dxi
      auto* d = BuildOp(UO_Minus, dfdx());
      diff = Visit(E, d);
    } else if (opCode == UO_PostInc || opCode == UO_PostDec) {
      diff = Visit(E, dfdx());
      Expr* diff_dx = diff.getExpr_dx();
      if (isPointerOp)
        addToCurrentBlock(BuildOp(opCode, diff_dx), direction::forward);
      if (m_DiffReq.shouldBeRecorded(E)) {
        auto op = opCode == UO_PostInc ? UO_PostDec : UO_PostInc;
        addToCurrentBlock(BuildOp(op, Clone(diff.getRevSweepAsExpr())),
                          direction::reverse);
        if (isPointerOp)
          addToCurrentBlock(BuildOp(op, diff_dx), direction::reverse);
      }

      ResultRef = diff_dx;
      valueForRevPass = diff.getRevSweepAsExpr();
      if (m_ExternalSource)
        m_ExternalSource->ActBeforeFinalizingPostIncDecOp(diff);
    } else if (opCode == UO_PreInc || opCode == UO_PreDec) {
      diff = Visit(E, dfdx());
      Expr* diff_dx = diff.getExpr_dx();
      if (isPointerOp)
        addToCurrentBlock(BuildOp(opCode, diff_dx), direction::forward);
      if (m_DiffReq.shouldBeRecorded(E)) {
        auto op = opCode == UO_PreInc ? UO_PreDec : UO_PreInc;
        addToCurrentBlock(BuildOp(op, Clone(diff.getRevSweepAsExpr())),
                          direction::reverse);
        if (isPointerOp)
          addToCurrentBlock(BuildOp(op, diff_dx), direction::reverse);
      }
      auto op = opCode == UO_PreInc ? BinaryOperatorKind::BO_Add
                                    : BinaryOperatorKind::BO_Sub;
      auto* sum = BuildOp(
          op, diff.getRevSweepAsExpr(),
          ConstantFolder::synthesizeLiteral(m_Context.IntTy, m_Context, 1));
      valueForRevPass = utils::BuildParenExpr(m_Sema, sum);
    } else if (opCode == UnaryOperatorKind::UO_Real ||
               opCode == UnaryOperatorKind::UO_Imag) {
      diff = VisitWithExplicitNoDfDx(E);
      ResultRef = BuildOp(opCode, diff.getExpr_dx());
      /// Create and add `__real r += dfdx()` expression.
      if (dfdx()) {
        Expr* add_assign = BuildOp(BO_AddAssign, ResultRef, dfdx());
        // Add it to the body statements.
        addToCurrentBlock(add_assign, direction::reverse);
      }
    } else if (opCode == UnaryOperatorKind::UO_AddrOf) {
      diff = Visit(E);
      Expr* cloneE = BuildOp(UnaryOperatorKind::UO_AddrOf, diff.getExpr());
      Expr* derivedE = BuildOp(UnaryOperatorKind::UO_AddrOf, diff.getExpr_dx());
      return {cloneE, derivedE};
    } else if (opCode == UnaryOperatorKind::UO_Deref) {
      diff = Visit(E);
      Expr* cloneE = BuildOp(UnaryOperatorKind::UO_Deref, diff.getExpr());

      // If we have a pointer to a member expression, which is
      // non-differentiable, we just return a clone of the original expression.
      if (auto* ME = dyn_cast<MemberExpr>(diff.getExpr()))
        if (clad::utils::hasNonDifferentiableAttribute(ME->getMemberDecl()))
          return {cloneE};

      Expr* diff_dx = diff.getExpr_dx();
      bool specialDThisCase = false;
      Expr* derivedE = nullptr;
      if (const auto* MD = dyn_cast<CXXMethodDecl>(m_DiffReq.Function)) {
        if (MD->isInstance() && !diff_dx->getType()->isPointerType())
          specialDThisCase = true; // _d_this is already dereferenced.
      }
      if (specialDThisCase)
        derivedE = diff_dx;
      else {
        derivedE = BuildOp(UnaryOperatorKind::UO_Deref, diff_dx);
        // Create the (target += dfdx) statement.
        if (dfdx() && derivedE) {
          if (shouldUseCudaAtomicOps(diff_dx)) {
            Expr* atomicCall = BuildCallToCudaAtomicAdd(diff_dx, dfdx());
            // Add it to the body statements.
            addToCurrentBlock(atomicCall, direction::reverse);
          } else {
            auto* add_assign = BuildOp(BO_AddAssign, derivedE, dfdx());
            // Add it to the body statements.
            addToCurrentBlock(add_assign, direction::reverse);
          }
        }
      }
      return {cloneE, derivedE, derivedE};
    } else {
      if (opCode != UO_LNot)
        // We should only output warnings on visiting boolean conditions
        // when it is related to some indepdendent variable and causes
        // discontinuity in the function space.
        // FIXME: We should support boolean differentiation or ignore it
        // completely
        unsupportedOpWarn(UnOp->getEndLoc());
      diff = Visit(E);
      ResultRef = diff.getExpr_dx();
    }
    Expr* op = BuildOp(opCode, diff.getExpr());
    return StmtDiff(op, ResultRef, nullptr, valueForRevPass);
  }

  StmtDiff
  ReverseModeVisitor::VisitBinaryOperator(const BinaryOperator* BinOp) {
    auto opCode = BinOp->getOpcode();
    StmtDiff Ldiff{};
    StmtDiff Rdiff{};
    StmtDiff Lstored{};
    Expr* valueForRevPass = nullptr;
    auto* L = BinOp->getLHS();
    auto* R = BinOp->getRHS();
    // If it is an assignment operator, its result is a reference to LHS and
    // we should return it.
    Expr* ResultRef = nullptr;

    bool isPointerOp =
        L->getType()->isPointerType() || R->getType()->isPointerType();

    if (opCode == BO_Add) {
      // xi = xl + xr
      // dxi/xl = 1.0
      // df/dxl += df/dxi * dxi/xl = df/dxi
      Ldiff = Visit(L, dfdx());
      // dxi/xr = 1.0
      // df/dxr += df/dxi * dxi/xr = df/dxi
      Rdiff = Visit(R, dfdx());
    } else if (opCode == BO_Sub) {
      // xi = xl - xr
      // dxi/xl = 1.0
      // df/dxl += df/dxi * dxi/xl = df/dxi
      Ldiff = Visit(L, dfdx());
      // dxi/xr = -1.0
      // df/dxl += df/dxi * dxi/xr = -df/dxi
      auto* dr = BuildOp(UO_Minus, dfdx());
      Rdiff = Visit(R, dr);
    } else if (opCode == BO_Mul) {
      // xi = xl * xr
      // dxi/xl = xr
      // df/dxl += df/dxi * dxi/xl = df/dxi * xr
      // Create uninitialized "global" variable for the right multiplier.
      // It will be assigned later after R is visited and cloned. This allows
      // to reduce cloning complexity and only clones once. Storing it in a
      // global variable allows to save current result and make it accessible
      // in the reverse pass.
      DelayedStoreResult RDelayed = DelayedGlobalStoreAndRef(R);
      StmtDiff& RResult = RDelayed.Result;

      Expr* dl = nullptr;
      if (dfdx())
        dl = BuildOp(BO_Mul, dfdx(), RResult.getRevSweepAsExpr());
      Ldiff = Visit(L, dl);
      // dxi/xr = xl
      // df/dxr += df/dxi * dxi/xr = df/dxi * xl
      // Store left multiplier and assign it with L.
      StmtDiff LStored = Ldiff;
      // Catch the pop statement and emit it after
      // the LStored value is used.
      // This workaround is necessary because GlobalStoreAndRef
      // is designed to work with the reversed order of statements
      // in the reverse sweep and in RMV::VisitBinaryOperator
      // the order is not reversed.
      beginBlock(direction::reverse);
      if (!ShouldRecompute(LStored.getExpr()))
        LStored = GlobalStoreAndRef(LStored.getExpr(), /*prefix=*/"_t",
                                    /*force=*/true);
      Stmt* LPop = endBlock(direction::reverse);
      Expr* dr = nullptr;
      if (dfdx())
        dr = BuildOp(BO_Mul, LStored.getRevSweepAsExpr(), dfdx());
      Rdiff = Visit(R, dr);
      // Assign right multiplier's variable with R.
      RDelayed.Finalize(Rdiff.getExpr());
      addToCurrentBlock(utils::unwrapIfSingleStmt(LPop), direction::reverse);
      std::tie(Ldiff, Rdiff) = std::make_pair(LStored, RResult);
    } else if (opCode == BO_Div) {
      // xi = xl / xr
      // dxi/xl = 1 / xr
      // df/dxl += df/dxi * dxi/xl = df/dxi * (1/xr)
      auto RDelayed = DelayedGlobalStoreAndRef(R, /*prefix=*/"_t",
                                               /*forceStore=*/true);
      StmtDiff& RResult = RDelayed.Result;
      Expr* dl = nullptr;
      if (dfdx())
        dl = BuildOp(BO_Div, dfdx(), RResult.getExpr());
      Ldiff = Visit(L, dl);
      StmtDiff LStored = Ldiff;
      // Catch the pop statement and emit it after
      // the LStored value is used.
      // This workaround is necessary because GlobalStoreAndRef
      // is designed to work with the reversed order of statements
      // in the reverse sweep and in RMV::VisitBinaryOperator
      // the order is not reversed.
      beginBlock(direction::reverse);
      if (!ShouldRecompute(LStored.getExpr()))
        LStored = GlobalStoreAndRef(LStored.getExpr(), /*prefix=*/"_t",
                                    /*force=*/true);
      Stmt* LPop = endBlock(direction::reverse);
      Expr::EvalResult dummy;
      if (!clad_compat::Expr_EvaluateAsConstantExpr(R, dummy, m_Context) ||
          RDelayed.needsUpdate) {
        // dxi/xr = -xl / (xr * xr)
        // df/dxl += df/dxi * dxi/xr = df/dxi * (-xl /(xr * xr))
        // Wrap R * R in parentheses: (R * R). otherwise code like 1 / R * R is
        // produced instead of 1 / (R * R).
        Expr* dr = nullptr;
        if (dfdx()) {
          Expr* RxR = BuildParens(
              BuildOp(BO_Mul, RResult.getExpr(), RResult.getExpr()));
          dr = BuildOp(BO_Mul, dfdx(),
                       BuildOp(UO_Minus,
                               BuildParens(BuildOp(
                                   BO_Div, LStored.getRevSweepAsExpr(), RxR))));
          dr = StoreAndRef(dr, direction::reverse);
        }
        Rdiff = Visit(R, dr);
        RDelayed.Finalize(Rdiff.getExpr());
      }
      addToCurrentBlock(utils::unwrapIfSingleStmt(LPop), direction::reverse);
      std::tie(Ldiff, Rdiff) = std::make_pair(LStored, RResult);
    } else if (BinOp->isAssignmentOp()) {
      if (L->isModifiableLvalue(m_Context) != Expr::MLV_Valid) {
        diag(DiagnosticsEngine::Warning,
             BinOp->getEndLoc(),
             "derivative of an assignment attempts to assign to unassignable "
             "expr, assignment ignored");
        auto* LDRE = dyn_cast<DeclRefExpr>(L);
        auto* RDRE = dyn_cast<DeclRefExpr>(R);

        if (!LDRE && !RDRE)
          return Clone(BinOp);
        Expr* LExpr = LDRE ? Visit(L).getRevSweepAsExpr() : L;
        Expr* RExpr = RDRE ? Visit(R).getRevSweepAsExpr() : R;

        return BuildOp(opCode, LExpr, RExpr);
      }

      // FIXME: Put this code into a separate subroutine and break out early
      // using return if the diff mode is not jacobian and we are not dealing
      // with the `outputArray`.
      if (auto* ASE = dyn_cast<ArraySubscriptExpr>(L)) {
        if (auto* DRE =
                dyn_cast<DeclRefExpr>(ASE->getBase()->IgnoreImplicit())) {
          auto type = QualType(DRE->getType()->getPointeeOrArrayElementType(),
                               /*Quals=*/0);
          std::string DRE_str = DRE->getDecl()->getNameAsString();

          llvm::APSInt intIdx;
          Expr::EvalResult res;
          Expr::SideEffectsKind AllowSideEffects =
              Expr::SideEffectsKind::SE_NoSideEffects;
          auto isIdxValid =
              ASE->getIdx()->EvaluateAsInt(res, m_Context, AllowSideEffects);

          if (DRE_str == outputArrayStr && isIdxValid) {
            intIdx = res.Val.getInt();
            if (m_DiffReq.Mode == DiffMode::jacobian) {
              outputArrayCursor = intIdx.getExtValue();

              std::unordered_map<const clang::ValueDecl*, clang::Expr*>
                  temp_m_Variables;
              for (unsigned i = 0; i < numParams; i++) {
                auto size_type = m_Context.getSizeType();
                unsigned size_type_bits = m_Context.getIntWidth(size_type);
                llvm::APInt idxValue(size_type_bits,
                                     i + (outputArrayCursor * numParams));
                auto* idx = IntegerLiteral::Create(m_Context, idxValue,
                                                   size_type, noLoc);
                // Create the jacobianMatrix[idx] expression.
                auto* result_at_i = m_Sema
                                        .CreateBuiltinArraySubscriptExpr(
                                            m_Result, noLoc, idx, noLoc)
                                        .get();
                temp_m_Variables[m_IndependentVars[i]] = result_at_i;
              }
              if (m_VectorOutput.size() <= outputArrayCursor)
                m_VectorOutput.resize(outputArrayCursor + 1);
              m_VectorOutput[outputArrayCursor] = std::move(temp_m_Variables);
            }

            auto* dfdf = ConstantFolder::synthesizeLiteral(m_Context.IntTy,
                                                           m_Context, 1);
            ExprResult tmp = dfdf;
            dfdf = m_Sema
                       .ImpCastExprToType(tmp.get(), type,
                                          m_Sema.PrepareScalarCast(tmp, type))
                       .get();
            auto ReturnResult = DifferentiateSingleExpr(R, dfdf);
            StmtDiff ReturnDiff = ReturnResult.first;
            Stmt* Reverse = ReturnDiff.getStmt_dx();
            addToCurrentBlock(Reverse, direction::reverse);
            for (Stmt* S : cast<CompoundStmt>(ReturnDiff.getStmt())->body())
              addToCurrentBlock(S, direction::forward);
          }
        }
      }

      // Visit LHS, but delay emission of its derivative statements, save them
      // in Lblock
      beginBlock(direction::reverse);
      Ldiff = Visit(L, dfdx());

      if (L->HasSideEffects(m_Context)) {
        Expr* E = Ldiff.getExpr();
        llvm::SmallVector<Expr*, 4> returnExprs;
        utils::GetInnermostReturnExpr(E, returnExprs);
        if (returnExprs.size() == 1) {
          addToCurrentBlock(E, direction::forward);
          Ldiff.updateStmt(returnExprs[0]);
        } else {
          auto* storeE = GlobalStoreAndRef(BuildOp(UO_AddrOf, E));
          Ldiff.updateStmt(BuildOp(UO_Deref, storeE));
        }
      }

      Stmts Lblock = EndBlockWithoutCreatingCS(direction::reverse);

      Expr* LCloned = Ldiff.getExpr();
      // For x, ResultRef is _d_x, for x[i] its _d_x[i], for reference exprs
      // like (x = y) it propagates recursively, so _d_x is also returned.
      ResultRef = Ldiff.getExpr_dx();
      // If assigned expr is dependent, first update its derivative;
      if (dfdx() && !Lblock.empty()) {
        addToCurrentBlock(*Lblock.begin(), direction::reverse);
        Lblock.erase(Lblock.begin());
      }

      // Store the value of the LHS of the assignment in the forward pass
      // and restore it in the reverse pass
      if (m_DiffReq.shouldBeRecorded(L)) {
        StmtDiff pushPop = StoreAndRestore(LCloned);
        addToCurrentBlock(pushPop.getStmt(), direction::forward);
        addToCurrentBlock(pushPop.getStmt_dx(), direction::reverse);
      }

      if (!ResultRef)
        return Clone(BinOp);
      // We need to store values of derivative pointer variables in forward pass
      // and restore them in reverse pass.
      if (isPointerOp) {
        StmtDiff pushPop = StoreAndRestore(Ldiff.getExpr_dx());
        addToCurrentBlock(pushPop.getStmt(), direction::forward);
        addToCurrentBlock(pushPop.getStmt_dx(), direction::reverse);
      }

      if (m_ExternalSource)
        m_ExternalSource->ActAfterCloningLHSOfAssignOp(LCloned, R, opCode);

      // Save old value for the derivative of LHS, to avoid problems with cases
      // like x = x.
      clang::Expr* oldValue = nullptr;

      // For pointer types, no need to store old derivatives.
      if (!isPointerOp)
        oldValue = StoreAndRef(ResultRef, direction::reverse, "_r_d",
                               /*forceDeclCreation=*/true);
      if (opCode == BO_Assign) {
        if (!isPointerOp) {
          // Add the statement `dl = 0;`
          Expr* zero = getZeroInit(ResultRef->getType());
          addToCurrentBlock(BuildOp(BO_Assign, ResultRef, zero),
                            direction::reverse);
        }
        Rdiff = Visit(R, oldValue);
        valueForRevPass = Rdiff.getRevSweepAsExpr();
      } else if (opCode == BO_AddAssign) {
        Rdiff = Visit(R, oldValue);
        if (!isPointerOp)
          valueForRevPass = BuildOp(BO_Add, Rdiff.getRevSweepAsExpr(),
                                    Ldiff.getRevSweepAsExpr());
      } else if (opCode == BO_SubAssign) {
        Rdiff = Visit(R, BuildOp(UO_Minus, oldValue));
        if (!isPointerOp)
          valueForRevPass = BuildOp(BO_Sub, Rdiff.getRevSweepAsExpr(),
                                    Ldiff.getRevSweepAsExpr());
      } else if (opCode == BO_MulAssign) {
        // Create a reference variable to keep the result of LHS, since it
        // must be used on 2 places: when storing to a global variable
        // accessible from the reverse pass, and when rebuilding the original
        // expression for the forward pass. This allows to avoid executing
        // same expression with side effects twice. E.g., on
        //   double r = (x *= y) *= z;
        // instead of:
        //   _t0 = (x *= y);
        //   double r = (x *= y) *= z;
        // which modifies x twice, we get:
        //   double & _ref0 = (x *= y);
        //   _t0 = _ref0;
        //   double r = _ref0 *= z;
        if (isInsideLoop)
          addToCurrentBlock(LCloned, direction::forward);
        // Add the statement `dl = 0;`
        Expr* zero = getZeroInit(ResultRef->getType());
        addToCurrentBlock(BuildOp(BO_Assign, ResultRef, zero),
                          direction::reverse);
        /// Capture all the emitted statements while visiting R
        /// and insert them after `dl += dl * R`
        beginBlock(direction::reverse);
        Expr* dr = BuildOp(BO_Mul, LCloned, oldValue);
        Rdiff = Visit(R, dr);
        Stmts RBlock = EndBlockWithoutCreatingCS(direction::reverse);
        addToCurrentBlock(
            BuildOp(BO_AddAssign, ResultRef,
                    BuildOp(BO_Mul, oldValue, Rdiff.getRevSweepAsExpr())),
            direction::reverse);
        for (auto& S : RBlock)
          addToCurrentBlock(S, direction::reverse);
        valueForRevPass = BuildOp(BO_Mul, Rdiff.getRevSweepAsExpr(),
                                  Ldiff.getRevSweepAsExpr());
        std::tie(Ldiff, Rdiff) = std::make_pair(LCloned, Rdiff.getExpr());
      } else if (opCode == BO_DivAssign) {
        // Add the statement `dl = 0;`
        Expr* zero = getZeroInit(ResultRef->getType());
        addToCurrentBlock(BuildOp(BO_Assign, ResultRef, zero),
                          direction::reverse);
        auto RDelayed = DelayedGlobalStoreAndRef(R, /*prefix=*/"_t",
                                                 /*forceStore=*/true);
        StmtDiff& RResult = RDelayed.Result;
        Expr* RStored =
            StoreAndRef(RResult.getRevSweepAsExpr(), direction::reverse);
        addToCurrentBlock(BuildOp(BO_AddAssign, ResultRef,
                                  BuildOp(BO_Div, oldValue, RStored)),
                          direction::reverse);
        if (isInsideLoop)
          addToCurrentBlock(LCloned, direction::forward);
        Expr* RxR = BuildParens(BuildOp(BO_Mul, RStored, RStored));
        Expr* dr = BuildOp(BO_Mul, oldValue,
                           BuildOp(UO_Minus, BuildOp(BO_Div, LCloned, RxR)));
        dr = StoreAndRef(dr, direction::reverse);
        Rdiff = Visit(R, dr);
        RDelayed.Finalize(Rdiff.getExpr());
        valueForRevPass = BuildOp(BO_Div, Rdiff.getRevSweepAsExpr(),
                                  Ldiff.getRevSweepAsExpr());
        std::tie(Ldiff, Rdiff) = std::make_pair(LCloned, RResult);
      } else
        llvm_unreachable("unknown assignment opCode");
      if (m_ExternalSource)
        m_ExternalSource->ActBeforeFinalizingAssignOp(LCloned, ResultRef, R,
                                                      opCode);

      // Output statements from Visit(L).
      for (Stmt* S : Lblock)
        addToCurrentBlock(S, direction::reverse);
    } else if (opCode == BO_Comma) {
      auto* zero =
          ConstantFolder::synthesizeLiteral(m_Context.IntTy, m_Context, 0);
      Rdiff = Visit(R, dfdx());
      Ldiff = Visit(L, zero);
      valueForRevPass = Ldiff.getRevSweepAsExpr();
      ResultRef = Ldiff.getExpr();
    } else if (opCode == BO_LAnd) {
      VarDecl* condVar = GlobalStoreImpl(m_Context.BoolTy, "_cond");
      VarDecl* derivedCondVar = GlobalStoreImpl(
          m_Context.DoubleTy, "_d" + condVar->getNameAsString());
      addToBlock(BuildOp(BO_Assign, BuildDeclRef(derivedCondVar),
                         ConstantFolder::synthesizeLiteral(
                             m_Context.DoubleTy, m_Context, /*val=*/0)),
                 m_Globals);
      Expr* condVarRef = BuildDeclRef(condVar);
      Expr* assignExpr = BuildOp(BO_Assign, condVarRef, Clone(R));
      m_Variables.emplace(condVar, BuildDeclRef(derivedCondVar));
      auto* IfStmt = clad_compat::IfStmt_Create(
          /*Ctx=*/m_Context, /*IL=*/noLoc, /*IsConstexpr=*/false,
          /*Init=*/nullptr, /*Var=*/nullptr,
          /*Cond=*/L, /*LPL=*/noLoc, /*RPL=*/noLoc, /*Then=*/assignExpr,
          /*EL=*/noLoc,
          /*Else=*/nullptr);

      StmtDiff IfStmtDiff = VisitIfStmt(IfStmt);
      addToCurrentBlock(utils::unwrapIfSingleStmt(IfStmtDiff.getStmt()));
      addToCurrentBlock(utils::unwrapIfSingleStmt(IfStmtDiff.getStmt_dx()),
                        direction::reverse);
      auto* condDiffStored = IfStmtDiff.getRevSweepAsExpr();
      return BuildOp(BO_LAnd, condDiffStored, condVarRef);
    } else {
      // We should not output any warning on visiting boolean conditions
      // FIXME: We should support boolean differentiation or ignore it
      // completely
      if (!BinOp->isComparisonOp() && !BinOp->isLogicalOp())
        unsupportedOpWarn(BinOp->getEndLoc());

      return BuildOp(opCode, Visit(L).getExpr(), Visit(R).getExpr());
    }
    Expr* op = BuildOp(opCode, Ldiff.getExpr(), Rdiff.getExpr());

    // For pointer types.
    if (isPointerOp) {
      if (opCode == BO_Add || opCode == BO_Sub) {
        Expr* derivedL = nullptr;
        Expr* derivedR = nullptr;
        ComputeEffectiveDOperands(Ldiff, Rdiff, derivedL, derivedR);
        if (opCode == BO_Sub)
          derivedR = BuildParens(derivedR);
        return StmtDiff(op, BuildOp(opCode, derivedL, derivedR), nullptr,
                        valueForRevPass);
      }
      if (opCode == BO_Assign || opCode == BO_AddAssign ||
          opCode == BO_SubAssign) {
        Expr* derivedL = nullptr;
        Expr* derivedR = nullptr;
        ComputeEffectiveDOperands(Ldiff, Rdiff, derivedL, derivedR);
        addToCurrentBlock(BuildOp(opCode, derivedL, derivedR),
                          direction::forward);
      }
    }
    return StmtDiff(op, ResultRef, nullptr, valueForRevPass);
  }

  DeclDiff<VarDecl> ReverseModeVisitor::DifferentiateVarDecl(const VarDecl* VD,
                                                             bool keepLocal) {
    StmtDiff initDiff;
    Expr* VDDerivedInit = nullptr;

    // Local declarations are promoted to the function global scope. This
    // procedure is done to make declarations visible in the reverse sweep.
    // The reverse_mode_forward_pass mode does not have a reverse pass so
    // declarations don't have to be moved to the function global scope.
    bool promoteToFnScope =
        !getCurrentScope()->isFunctionScope() &&
        m_DiffReq.Mode != DiffMode::reverse_mode_forward_pass && !keepLocal;
    QualType VDCloneType;
    QualType VDDerivedType;
    QualType VDType = VD->getType();
    // If the cloned declaration is moved to the function global scope,
    // change its type for the corresponding adjoint type.
    if (promoteToFnScope) {
      VDDerivedType = ComputeAdjointType(CloneType(VDType));
      VDCloneType = VDDerivedType;
      if (isa<ArrayType>(VDCloneType) && !isa<IncompleteArrayType>(VDCloneType))
        VDCloneType =
            GetCladArrayOfType(m_Context.getBaseElementType(VDCloneType));
    } else {
      VDCloneType = CloneType(VDType);
      VDDerivedType = getNonConstType(VDCloneType, m_Context, m_Sema);
    }

    bool isRefType = VDType->isLValueReferenceType();
    VarDecl* VDDerived = nullptr;
    bool isPointerType = VDType->isPointerType();
    bool isInitializedByNewExpr = false;
    bool initializeDerivedVar = true;

    // We need to replace std::initializer_list with clad::array because the
    // former is temporary by design and it's not possible to create modifiable
    // adjoints.
    if (m_Sema.isStdInitializerList(utils::GetValueType(VDType),
                                    /*Element=*/nullptr)) {
      if (const Expr* init = VD->getInit()) {
        if (const auto* CXXILE =
                dyn_cast<CXXStdInitializerListExpr>(init->IgnoreImplicit())) {
          if (const auto* ILE = dyn_cast<InitListExpr>(
                  CXXILE->getSubExpr()->IgnoreImplicit())) {
            VDDerivedType =
                GetCladArrayOfType(ILE->getInit(/*Init=*/0)->getType());
            unsigned numInits = ILE->getNumInits();
            VDDerivedInit = ConstantFolder::synthesizeLiteral(
                m_Context.getSizeType(), m_Context, numInits);
            VDCloneType = VDDerivedType;
          }
        } else if (isRefType) {
          initDiff = Visit(init);
          if (promoteToFnScope) {
            VDDerivedInit = BuildOp(UO_AddrOf, initDiff.getExpr_dx());
            VDDerivedType = VDDerivedInit->getType();
          } else {
            VDDerivedInit = initDiff.getExpr_dx();
            VDDerivedType =
                m_Context.getLValueReferenceType(VDDerivedInit->getType());
          }
          VDCloneType = VDDerivedType;
        }
      }
    }

    // Check if the variable is pointer type and initialized by new expression
    if (isPointerType && VD->getInit() && isa<CXXNewExpr>(VD->getInit()))
      isInitializedByNewExpr = true;

    ConstructorPullbackCallInfo constructorPullbackInfo;

    // VDDerivedInit now serves two purposes -- as the initial derivative value
    // or the size of the derivative array -- depending on the primal type.
    if (const auto* AT = dyn_cast<ArrayType>(VDType)) {
      if (!isa<VariableArrayType>(AT)) {
        Expr* zero =
            ConstantFolder::synthesizeLiteral(m_Context.IntTy, m_Context, 0);
        VDDerivedInit = m_Sema.ActOnInitList(noLoc, {zero}, noLoc).get();
      }
      if (promoteToFnScope) {
        // If an array-type declaration is promoted to function global,
        // its type is changed for clad::array. In that case we should
        // initialize it with its size.
        initDiff = getArraySizeExpr(AT, m_Context, *this);
      }
      VDDerived = BuildGlobalVarDecl(
          VDDerivedType, "_d_" + VD->getNameAsString(), VDDerivedInit, false,
          nullptr, VarDecl::InitializationStyle::CInit);
    } else {
      // If VD is a reference to a local variable, then the initial value is set
      // to the derived variable of the corresponding local variable.
      // If VD is a reference to a non-local variable (global variable, struct
      // member etc), then no derived variable is available, thus `VDDerived`
      // does not need to reference any variable, consequentially the
      // `VDDerivedType` is the corresponding non-reference type and the initial
      // value is set to 0.
      // Otherwise, for non-reference types, the initial value is set to 0.
      if (!VDDerivedInit)
        VDDerivedInit = getZeroInit(VDType);

      // `specialThisDiffCase` is only required for correctly differentiating
      // the following code:
      // ```
      // Class _d_this_obj;
      // Class* _d_this = &_d_this_obj;
      // ```
      // Computation of hessian requires this code to be correctly
      // differentiated.
      bool specialThisDiffCase = false;
      if (const auto* MD = dyn_cast<CXXMethodDecl>(m_DiffReq.Function)) {
        if (VDDerivedType->isPointerType() && MD->isInstance()) {
          specialThisDiffCase = true;
        }
      }

      if (isRefType) {
        initDiff = Visit(VD->getInit());
        if (!initDiff.getForwSweepExpr_dx()) {
          VDDerivedType = ComputeAdjointType(VDType.getNonReferenceType());
          isRefType = false;
        }
        if (promoteToFnScope || !isRefType)
          VDDerivedInit = getZeroInit(VDDerivedType);
        else
          VDDerivedInit = initDiff.getForwSweepExpr_dx();
      }

      if (VDType->isStructureOrClassType()) {
        m_TrackConstructorPullbackInfo = true;
        initDiff = Visit(VD->getInit());
        m_TrackConstructorPullbackInfo = false;
        constructorPullbackInfo = getConstructorPullbackCallInfo();
        resetConstructorPullbackCallInfo();
        if (initDiff.getForwSweepExpr_dx())
          VDDerivedInit = initDiff.getForwSweepExpr_dx();
      }

      // FIXME: Remove the special cases introduced by `specialThisDiffCase`
      // once reverse mode supports pointers. `specialThisDiffCase` is only
      // required for correctly differentiating the following code:
      // ```
      // Class _d_this_obj;
      // Class* _d_this = &_d_this_obj;
      // ```
      // Computation of hessian requires this code to be correctly
      // differentiated.
      if (specialThisDiffCase && VD->getNameAsString() == "_d_this") {
        VDDerivedType = getNonConstType(VDDerivedType, m_Context, m_Sema);
        initDiff = Visit(VD->getInit());
        if (initDiff.getExpr_dx())
          VDDerivedInit = initDiff.getExpr_dx();
      }
      // if VD is a pointer type, then the initial value is set to the derived
      // expression of the corresponding pointer type.
      else if (isPointerType) {
        if (!isInitializedByNewExpr)
          initDiff = Visit(VD->getInit());

        // If the pointer is const and derived expression is not available, then
        // we should not create a derived variable for it. This will be useful
        // for reducing number of differentiation variables in pullbacks.
        bool constPointer = VDType->getPointeeType().isConstQualified();
        if (constPointer && !isInitializedByNewExpr && !initDiff.getExpr_dx())
          initializeDerivedVar = false;
        else {
          VDDerivedType = getNonConstType(VDDerivedType, m_Context, m_Sema);
          // If it's a pointer to a constant type, then remove the constness.
          if (constPointer) {
            // first extract the pointee type
            auto pointeeType = VDType->getPointeeType();
            // then remove the constness
            pointeeType.removeLocalConst();
            // then create a new pointer type with the new pointee type
            VDDerivedType = m_Context.getPointerType(pointeeType);
          }
          VDDerivedInit = getZeroInit(VDDerivedType);
        }
      }
      if (initializeDerivedVar)
        VDDerived = BuildGlobalVarDecl(
            VDDerivedType, "_d_" + VD->getNameAsString(), VDDerivedInit, false,
            nullptr, VD->getInitStyle());
    }

    if (!m_DiffReq.shouldHaveAdjoint((VD)))
      VDDerived = nullptr;

    // If `VD` is a reference to a local variable, then it is already
    // differentiated and should not be differentiated again.
    // If `VD` is a reference to a non-local variable then also there's no
    // need to call `Visit` since non-local variables are not differentiated.
    if (!isRefType && (!isPointerType || isInitializedByNewExpr)) {
      Expr* derivedE = nullptr;

      if (VDDerived && !clad::utils::hasNonDifferentiableAttribute(VD)) {
        derivedE = BuildDeclRef(VDDerived);
        if (isInitializedByNewExpr)
          derivedE = BuildOp(UnaryOperatorKind::UO_Deref, derivedE);
      }

      if (VD->getInit()) {
        if (VDType->isStructureOrClassType()) {
          if (!initDiff.getExpr())
            initDiff = Visit(VD->getInit());
        } else
          initDiff = Visit(VD->getInit(), derivedE);
      }

      // If we are differentiating `VarDecl` corresponding to a local variable
      // inside a loop, then we need to reset it to 0 at each iteration.
      //
      // for example, if defined inside a loop,
      // ```
      // double localVar = i;
      // ```
      // this statement should get differentiated to,
      // ```
      // {
      //   *_d_i += _d_localVar;
      //   _d_localVar = 0;
      // }
      if (VDDerived && isInsideLoop) {
        Stmt* assignToZero = nullptr;
        Expr* declRef = BuildDeclRef(VDDerived);
        if (!isa<ArrayType>(VDDerivedType))
          assignToZero = BuildOp(BinaryOperatorKind::BO_Assign, declRef,
                                 getZeroInit(VDDerivedType));
        else
          assignToZero = GetCladZeroInit(declRef);
        if (!keepLocal)
          addToCurrentBlock(assignToZero, direction::reverse);
      }
    }

    VarDecl* VDClone = nullptr;
    Expr* derivedVDE = nullptr;
    if (VDDerived && m_DiffReq.shouldHaveAdjoint(const_cast<VarDecl*>(VD)))
      derivedVDE = BuildDeclRef(VDDerived);
    // FIXME: Add extra parantheses if derived variable pointer is pointing to a
    // class type object.
    if (isRefType && promoteToFnScope) {
      Expr* assignDerivativeE =
          BuildOp(BinaryOperatorKind::BO_Assign, derivedVDE,
                  BuildOp(UnaryOperatorKind::UO_AddrOf,
                          initDiff.getForwSweepExpr_dx()));
      addToCurrentBlock(assignDerivativeE);
      if (isInsideLoop) {
        StmtDiff pushPop = StoreAndRestore(derivedVDE);
        if (!keepLocal)
          addToCurrentBlock(pushPop.getStmt(), direction::forward);
        m_LoopBlock.back().push_back(pushPop.getStmt_dx());
      }
      derivedVDE = BuildOp(UnaryOperatorKind::UO_Deref, derivedVDE);
    }

    // If a ref-type declaration is promoted to function global scope,
    // it's replaced with a pointer and should be initialized with the
    // address of the cloned init. e.g.
    // double& ref = x;
    // ->
    // double* ref;
    // ref = &x;
    if (isRefType && promoteToFnScope)
      VDClone = BuildGlobalVarDecl(
          VDCloneType, VD->getNameAsString(),
          BuildOp(UnaryOperatorKind::UO_AddrOf, initDiff.getExpr()),
          VD->isDirectInit());
    else
      VDClone = BuildGlobalVarDecl(VDCloneType, VD->getNameAsString(),
                                   initDiff.getExpr(), VD->isDirectInit(),
                                   nullptr, VD->getInitStyle());
    if (isPointerType && derivedVDE) {
      if (promoteToFnScope) {
        Expr* assignDerivativeE = BuildOp(BinaryOperatorKind::BO_Assign,
                                          derivedVDE, initDiff.getExpr_dx());
        addToCurrentBlock(assignDerivativeE, direction::forward);
        if (isInsideLoop) {
          auto tape = MakeCladTapeFor(derivedVDE);
          if (!keepLocal)
            addToCurrentBlock(tape.Push);
          auto* reverseSweepDerivativePointerE =
              BuildVarDecl(derivedVDE->getType(), "_t", tape.Pop);
          m_LoopBlock.back().push_back(
              BuildDeclStmt(reverseSweepDerivativePointerE));
          derivedVDE = BuildDeclRef(reverseSweepDerivativePointerE);
        }
      } else {
        m_Sema.AddInitializerToDecl(VDDerived, initDiff.getExpr_dx(), true);
        VDDerived->setInitStyle(VarDecl::InitializationStyle::CInit);
      }
    }

    if (derivedVDE)
      m_Variables.emplace(VDClone, derivedVDE);

    // Check if decl's name is the same as before. The name may be changed
    // if decl name collides with something in the derivative body.
    // This can happen in rare cases, e.g. when the original function
    // has both y and _d_y (here _d_y collides with the name produced by
    // the derivation process), e.g.
    // double f(double x) {
    //   double y = x;
    //   double _d_y = x;
    // }
    // ->
    // double f_darg0(double x) {
    //   double _d_x = 1;
    //   double _d_y = _d_x; // produced as a derivative for y
    //   double y = x;
    //   double _d__d_y = _d_x;
    //   double _d_y = x; // copied from original function, collides with
    //   _d_y
    // }
    if ((VD->getDeclName() != VDClone->getDeclName() ||
         VDType != VDClone->getType()))
      m_DeclReplacements[VD] = VDClone;

    if (!constructorPullbackInfo.empty()) {
      Expr* thisE =
          BuildOp(UnaryOperatorKind::UO_AddrOf, BuildDeclRef(VDClone));
      Expr* dThisE =
          BuildOp(UnaryOperatorKind::UO_AddrOf, BuildDeclRef(VDDerived));
      constructorPullbackInfo.updateThisParmArgs(thisE, dThisE);
    }
    return DeclDiff<VarDecl>(VDClone, VDDerived);
  }

  // TODO: 'shouldEmit' parameter should be removed after converting
  // Error estimation framework to callback style. Some more research
  // need to be done to
  StmtDiff ReverseModeVisitor::DifferentiateSingleStmt(const Stmt* S,
                                                       Expr* dfdS) {
    if (m_ExternalSource)
      m_ExternalSource->ActOnStartOfDifferentiateSingleStmt();
    beginBlock(direction::reverse);
    StmtDiff SDiff = Visit(S, dfdS);

    if (m_ExternalSource)
      m_ExternalSource->ActBeforeFinalizingDifferentiateSingleStmt(direction::reverse);

    // If the statement is a standalone call to a memory function, we want to
    // add its derived statement in the same block as the original statement.
    // For ex: memset(x, 0, 10) -> memset(_d_x, 0, 10)
    Stmt* stmtDx = SDiff.getStmt_dx();
    bool dxInForward = false;
    if (auto* callExpr = dyn_cast_or_null<CallExpr>(stmtDx))
      if (auto* FD = dyn_cast<FunctionDecl>(callExpr->getCalleeDecl()))
        if (utils::IsMemoryFunction(FD))
          dxInForward = true;
    if (stmtDx) {
      if (dxInForward)
        addToCurrentBlock(stmtDx, direction::forward);
      else
        addToCurrentBlock(stmtDx, direction::reverse);
    }
    CompoundStmt* RCS = endBlock(direction::reverse);
    std::reverse(RCS->body_begin(), RCS->body_end());
    Stmt* ReverseResult = utils::unwrapIfSingleStmt(RCS);

    return StmtDiff(SDiff.getStmt(), ReverseResult);
  }

  std::pair<StmtDiff, StmtDiff>
  ReverseModeVisitor::DifferentiateSingleExpr(const Expr* E, Expr* dfdE) {
    beginBlock(direction::forward);
    beginBlock(direction::reverse);
    StmtDiff EDiff = Visit(E, dfdE);
    if (m_ExternalSource)
      m_ExternalSource->ActBeforeFinalizingDifferentiateSingleExpr(direction::reverse);
    CompoundStmt* RCS = endBlock(direction::reverse);
    Stmt* ForwardResult = endBlock(direction::forward);
    std::reverse(RCS->body_begin(), RCS->body_end());
    Stmt* ReverseResult = utils::unwrapIfSingleStmt(RCS);
    return {StmtDiff(ForwardResult, ReverseResult), EDiff};
  }

  StmtDiff ReverseModeVisitor::VisitDeclStmt(const DeclStmt* DS) {
    llvm::SmallVector<Stmt*, 16> inits;
    llvm::SmallVector<Decl*, 4> decls;
    llvm::SmallVector<Decl*, 4> declsDiff;
    // Need to put array decls inlined.
    llvm::SmallVector<Decl*, 4> localDeclsDiff;
    // reverse_mode_forward_pass does not have a reverse pass so declarations
    // don't have to be moved to the function global scope.
    bool promoteToFnScope =
        !getCurrentScope()->isFunctionScope() &&
        m_DiffReq.Mode != DiffMode::reverse_mode_forward_pass;

    // If the DeclStmt is not empty, check the first declaration in case it is a
    // lambda function. This case it is treated separately for now and we don't
    // create a variable for its derivative.
    bool isLambda = false;
    const auto* declsBegin = DS->decls().begin();
    if (declsBegin != DS->decls().end() && isa<VarDecl>(*declsBegin)) {
      auto* VD = dyn_cast<VarDecl>(*declsBegin);
      QualType QT = VD->getType();
      if (QT->isPointerType())
        QT = QT->getPointeeType();

      auto* typeDecl = QT->getAsCXXRecordDecl();
      // We should also simply copy the original lambda. The differentiation
      // of lambdas is happening in the `VisitCallExpr`. For now, only the
      // declarations with lambda expressions without captures are supported.
      isLambda = typeDecl && typeDecl->isLambda();
      if (isLambda ||
          (typeDecl && clad::utils::hasNonDifferentiableAttribute(typeDecl))) {
        for (auto* D : DS->decls())
          if (auto* VD = dyn_cast<VarDecl>(D))
            decls.push_back(VD);
        Stmt* DSClone = BuildDeclStmt(decls);
        return StmtDiff(DSClone, nullptr);
      }
    }

    // For each variable declaration v, create another declaration _d_v to
    // store derivatives for potential reassignments. E.g.
    // double y = x;
    // ->
    // double _d_y = _d_x; double y = x;
    for (auto* D : DS->decls()) {
      if (auto* VD = dyn_cast<VarDecl>(D)) {
        DeclDiff<VarDecl> VDDiff;

        if (!isLambda)
          VDDiff = DifferentiateVarDecl(VD);

        // Here, we move the declaration to the function global scope.
        // Initialization is replaced with an assignment operation at the same
        // place as the original declaration. This procedure is done to make the
        // declaration visible in the reverse sweep. The variable is stored
        // before the assignment in case its value is overwritten in a loop.
        // e.g.
        // while (cond) {
        //   double x = k * n;
        // ...
        // ->
        // double x;
        // clad::tape<double> _t0 = {};
        // while (cond) {
        //   clad::push(_t0, x), x = k * n;
        // ...
        if (promoteToFnScope) {
          auto* decl = VDDiff.getDecl();
          if (VD->getInit()) {
            auto* declRef = BuildDeclRef(decl);
            auto* assignment = BuildOp(BO_Assign, declRef, decl->getInit());
            if (isInsideLoop) {
              auto pushPop = StoreAndRestore(declRef);
              if (pushPop.getExpr() != declRef)
                addToCurrentBlock(pushPop.getExpr_dx(), direction::reverse);
              assignment = BuildOp(BO_Comma, pushPop.getExpr(), assignment);
            }
            inits.push_back(assignment);
            if (const auto* AT = dyn_cast<ArrayType>(VD->getType())) {
              m_Sema.AddInitializerToDecl(
                  decl, Clone(getArraySizeExpr(AT, m_Context, *this)), true);
              decl->setInitStyle(VarDecl::InitializationStyle::CallInit);
            } else {
              m_Sema.AddInitializerToDecl(decl, getZeroInit(VD->getType()),
                                          /*DirectInit=*/true);
              decl->setInitStyle(VarDecl::InitializationStyle::CInit);
            }
          }
        }

        decls.push_back(VDDiff.getDecl());
        if (VDDiff.getDecl_dx()) {
          if (isa<VariableArrayType>(VD->getType()))
            localDeclsDiff.push_back(VDDiff.getDecl_dx());
          else
            declsDiff.push_back(VDDiff.getDecl_dx());
        }
      } else if (auto* SAD = dyn_cast<StaticAssertDecl>(D)) {
        DeclDiff<StaticAssertDecl> SADDiff = DifferentiateStaticAssertDecl(SAD);
        if (SADDiff.getDecl())
          decls.push_back(SADDiff.getDecl());
        if (SADDiff.getDecl_dx())
          declsDiff.push_back(SADDiff.getDecl_dx());
      } else {
        diag(DiagnosticsEngine::Warning,
             D->getEndLoc(),
             "Unsupported declaration");
      }
    }

    Stmt* DSClone = nullptr;
    if (!decls.empty())
      DSClone = BuildDeclStmt(decls);

    if (!localDeclsDiff.empty()) {
      Stmt* localDSDIff = BuildDeclStmt(localDeclsDiff);
      addToCurrentBlock(
          localDSDIff,
          clad::rmv::forward); // Doesnt work for arrays decl'd in loops.
      for (Decl* decl : localDeclsDiff)
        if (const auto* VAT =
                dyn_cast<VariableArrayType>(cast<VarDecl>(decl)->getType())) {
          std::array<Expr*, 2> args{};
          args[0] = BuildDeclRef(cast<VarDecl>(decl));
          args[1] = Clone(VAT->getSizeExpr());
          Stmt* initCall = GetCladZeroInit(args);
          addToCurrentBlock(initCall, direction::forward);
        }
    }
    if (!declsDiff.empty()) {
      Stmt* DSDiff = BuildDeclStmt(declsDiff);
      Stmts& block =
          promoteToFnScope ? m_Globals : getCurrentBlock(direction::forward);
      addToBlock(DSDiff, block);
    }

    if (m_ExternalSource) {
      declsDiff.append(localDeclsDiff.begin(), localDeclsDiff.end());
      m_ExternalSource->ActBeforeFinalizingVisitDeclStmt(decls, declsDiff);
    }

    // This part in necessary to replace local variables inside loops
    // with function globals and replace initializations with assignments.
    if (promoteToFnScope) {
      // FIXME: We only need to produce separate decl stmts
      // because arrays promoted to the function scope are
      // turned into clad::array. This is done because of
      // mixed declarations.
      // e.g.
      // double a, b[5];
      // ->
      // double a, b(5UL);
      // when it should be
      // double a;
      // clad::array<double> b(5UL);
      // If we remove the need for clad::array here,
      // just add DSClone to the block.
      for (Decl* decl : decls)
        addToBlock(BuildDeclStmt(decl), m_Globals);
      Stmt* initAssignments = MakeCompoundStmt(inits);
      initAssignments = utils::unwrapIfSingleStmt(initAssignments);
      return StmtDiff(initAssignments);
    }

    return StmtDiff(DSClone);
  }

  StmtDiff
  ReverseModeVisitor::VisitImplicitCastExpr(const ImplicitCastExpr* ICE) {
    // Casts should be handled automatically when the result is used by
    // Sema::ActOn.../Build...
    return Visit(ICE->getSubExpr(), dfdx());
  }

  StmtDiff ReverseModeVisitor::VisitImplicitValueInitExpr(
      const ImplicitValueInitExpr* IVIE) {
    return {Clone(IVIE), Clone(IVIE)};
  }

  StmtDiff ReverseModeVisitor::VisitCStyleCastExpr(const CStyleCastExpr* CSCE) {
    StmtDiff subExprDiff = Visit(CSCE->getSubExpr(), dfdx());
    Expr* castExpr = m_Sema
                         .BuildCStyleCastExpr(
                             CSCE->getLParenLoc(), CSCE->getTypeInfoAsWritten(),
                             CSCE->getRParenLoc(), subExprDiff.getExpr())
                         .get();
    Expr* castExprDiff = subExprDiff.getExpr_dx();
    if (castExprDiff != nullptr)
      castExprDiff = m_Sema
                         .BuildCStyleCastExpr(
                             CSCE->getLParenLoc(), CSCE->getTypeInfoAsWritten(),
                             CSCE->getRParenLoc(), subExprDiff.getExpr_dx())
                         .get();
    return {castExpr, castExprDiff};
  }

  StmtDiff
  ReverseModeVisitor::VisitPseudoObjectExpr(const PseudoObjectExpr* POE) {
    // Used for CUDA Builtins
    return {Clone(POE), Clone(POE)};
  }

  StmtDiff ReverseModeVisitor::VisitMemberExpr(const MemberExpr* ME) {
    auto baseDiff = VisitWithExplicitNoDfDx(ME->getBase());
    auto* field = ME->getMemberDecl();
    assert(!isa<CXXMethodDecl>(field) &&
           "CXXMethodDecl nodes not supported yet!");
    MemberExpr* clonedME = utils::BuildMemberExpr(
        m_Sema, getCurrentScope(), baseDiff.getExpr(), field->getName());
    if (clad::utils::hasNonDifferentiableAttribute(ME)) {
      auto* zero = ConstantFolder::synthesizeLiteral(m_Context.IntTy, m_Context,
                                                     /*val=*/0);
      return {clonedME, zero};
    }
    if (!baseDiff.getExpr_dx())
      return {clonedME, nullptr};
    MemberExpr* derivedME = utils::BuildMemberExpr(
        m_Sema, getCurrentScope(), baseDiff.getExpr_dx(), field->getName());
    if (dfdx()) {
      Expr* addAssign =
          BuildOp(BinaryOperatorKind::BO_AddAssign, derivedME, dfdx());
      addToCurrentBlock(addAssign, direction::reverse);
    }
    return {clonedME, derivedME, derivedME};
  }

  StmtDiff
  ReverseModeVisitor::VisitExprWithCleanups(const ExprWithCleanups* EWC) {
    StmtDiff subExprDiff = Visit(EWC->getSubExpr(), dfdx());
    // FIXME: We are unable to create cleanup objects currently, this can be
    // potentially problematic
    return StmtDiff(subExprDiff.getExpr(), subExprDiff.getExpr_dx());
  }

  bool ReverseModeVisitor::ShouldRecompute(const Expr* E) {
    return !(utils::ContainsFunctionCalls(E) || E->HasSideEffects(m_Context));
  }

  bool ReverseModeVisitor::UsefulToStoreGlobal(Expr* E) {
    if (!E)
      return false;
    // Use stricter policy when inside loops: IsEvaluatable is also true for
    // arithmetical expressions consisting of constants, e.g. (1 + 2)*3. This
    // chech is more expensive, but it doesn't make sense to push such constants
    // into stack.
    if (isInsideLoop && E->isEvaluatable(m_Context, Expr::SE_NoSideEffects))
      return false;
    Expr* B = E->IgnoreParenImpCasts();
    // FIXME: find a more general way to determine that or add more options.
    if (isa<FloatingLiteral>(B) || isa<IntegerLiteral>(B))
      return false;
    if (isa<UnaryOperator>(B)) {
      auto* UO = cast<UnaryOperator>(B);
      auto OpKind = UO->getOpcode();
      if (OpKind == UO_Plus || OpKind == UO_Minus)
        return UsefulToStoreGlobal(UO->getSubExpr());
      return true;
    }

    // FIXME: Attach checkpointing.
    if (isa<CallExpr>(B))
      return false;

    // Assume E is useful to store.
    return true;
  }

  VarDecl* ReverseModeVisitor::GlobalStoreImpl(QualType Type,
                                               llvm::StringRef prefix,
                                               Expr* init) {
    // Create identifier before going to topmost scope
    // to let Sema::LookupName see the whole scope.
    auto* identifier = CreateUniqueIdentifier(prefix);
    // Save current scope and temporarily go to topmost function scope.
    llvm::SaveAndRestore<Scope*> SaveScope(getCurrentScope());
    assert(m_DerivativeFnScope && "must be set");
    setCurrentScope(m_DerivativeFnScope);

    VarDecl* Var = nullptr;
    if (isa<ArrayType>(Type)) {
      Type = GetCladArrayOfType(m_Context.getBaseElementType(Type));
      Var = BuildVarDecl(Type, identifier, init, false, nullptr,
                         clang::VarDecl::InitializationStyle::CallInit);
    } else {
      Var = BuildVarDecl(Type, identifier, init, false, nullptr,
                         VarDecl::InitializationStyle::CInit);
    }

    // Add the declaration to the body of the gradient function.
    addToBlock(BuildDeclStmt(Var), m_Globals);
    return Var;
  }

  Expr* ReverseModeVisitor::GlobalStoreAndRef(Expr* E, QualType Type,
                                              llvm::StringRef prefix,
                                              bool force) {
    assert(E && "must be provided, otherwise use DelayedGlobalStoreAndRef");
    assert(!isa<ArrayType>(Type) && "Array types cannot be stored.");
    if (!force && !UsefulToStoreGlobal(E))
      return E;

    if (isInsideLoop) {
      CladTapeResult CladTape = MakeCladTapeFor(E, prefix);
      addToCurrentBlock(CladTape.Push, direction::forward);
      addToCurrentBlock(CladTape.Pop, direction::reverse);

      return CladTape.Last();
    }

    VarDecl* VD = BuildGlobalVarDecl(Type, prefix);
    DeclStmt* decl = BuildDeclStmt(VD);
    Expr* Ref = BuildDeclRef(VD);
    bool isFnScope = getCurrentScope()->isFunctionScope() ||
                     m_DiffReq.Mode == DiffMode::reverse_mode_forward_pass;
    if (isFnScope) {
      addToCurrentBlock(decl, direction::forward);
      m_Sema.AddInitializerToDecl(VD, E, /*DirectInit=*/true);
      VD->setInitStyle(VarDecl::InitializationStyle::CInit);
    } else {
      addToBlock(decl, m_Globals);
      Expr* Set = BuildOp(BO_Assign, Ref, E);
      addToCurrentBlock(Set, direction::forward);
    }

    return Ref;
  }

  Expr* ReverseModeVisitor::GlobalStoreAndRef(Expr* E, llvm::StringRef prefix,
                                              bool force) {
    assert(E && "cannot infer type");
    return GlobalStoreAndRef(
        E, getNonConstType(E->getType(), m_Context, m_Sema), prefix, force);
  }

  StmtDiff ReverseModeVisitor::StoreAndRestore(clang::Expr* E,
                                               llvm::StringRef prefix) {
    assert(E && "must be provided");
    auto Type = getNonConstType(E->getType(), m_Context, m_Sema);

    if (isInsideLoop) {
      auto CladTape = MakeCladTapeFor(Clone(E), prefix);
      Expr* Push = CladTape.Push;
      Expr* Pop = CladTape.Pop;
      auto* popAssign = BuildOp(BinaryOperatorKind::BO_Assign, Clone(E), Pop);
      return {Push, popAssign};
    }

    Expr* init = nullptr;
    if (const auto* AT = dyn_cast<ArrayType>(Type))
      init = getArraySizeExpr(AT, m_Context, *this);

    VarDecl* VD = BuildGlobalVarDecl(Type, prefix, init);
    DeclStmt* decl = BuildDeclStmt(VD);
    Expr* Ref = BuildDeclRef(VD);
    Stmt* Store = nullptr;
    bool isFnScope = getCurrentScope()->isFunctionScope() ||
                     m_DiffReq.Mode == DiffMode::reverse_mode_forward_pass;
    if (isFnScope) {
      Store = decl;
      m_Sema.AddInitializerToDecl(VD, E, /*DirectInit=*/true);
      VD->setInitStyle(VarDecl::InitializationStyle::CInit);
    } else {
      addToBlock(decl, m_Globals);
      Store = BuildOp(BO_Assign, Ref, Clone(E));
    }

    Stmt* Restore = nullptr;
    if (E->isModifiableLvalue(m_Context) == Expr::MLV_Valid)
      Restore = BuildOp(BO_Assign, Clone(E), Ref);

    return {Store, Restore};
  }

  void ReverseModeVisitor::DelayedStoreResult::Finalize(Expr* New) {
    // Placeholders are used when we have to use an expr before we have that.
    // For instance, this is necessary for multiplication and division when the
    // RHS and LHS need the derivatives of each other to be differentiated. We
    // need placeholders to break this loop.
    class PlaceholderReplacer
        : public RecursiveASTVisitor<PlaceholderReplacer> {
    public:
      const Expr* placeholder;
      Sema& m_Sema;
      ASTContext& m_Context;
      Expr* newExpr{nullptr};
      PlaceholderReplacer(const Expr* Placeholder, Sema& S)
          : placeholder(Placeholder), m_Sema(S), m_Context(S.getASTContext()) {}

      void Replace(ReverseModeVisitor& RMV, Expr* New, StmtDiff& Result) {
        newExpr = New;
        for (Stmt* S : RMV.getCurrentBlock(direction::forward))
          TraverseStmt(S);
        for (Stmt* S : RMV.getCurrentBlock(direction::reverse))
          TraverseStmt(S);
        Result = New;
      }

      // We chose iteration rather than visiting because we only do this for
      // simple Expression subtrees and it is not worth it to implement an
      // entire visitor infrastructure for simple replacements.
      bool VisitExpr(Expr* E) const {
        for (Stmt*& S : E->children())
          if (S == placeholder) {
            // Since we are manually replacing the statement, implicit casts are
            // not generated automatically.
            ExprResult newExprRes{newExpr};
            QualType targetTy = cast<Expr>(S)->getType();
            CastKind kind = m_Sema.PrepareScalarCast(newExprRes, targetTy);
            // CK_NoOp casts trigger an assertion on debug Clang
            if (kind == CK_NoOp)
              S = newExpr;
            else
              S = m_Sema.ImpCastExprToType(newExpr, targetTy, kind).get();
          }
        return true;
      }
      PlaceholderReplacer(const PlaceholderReplacer&) = delete;
      PlaceholderReplacer(PlaceholderReplacer&&) = delete;
    };

    if (!needsUpdate)
      return;

    if (Placeholder) {
      PlaceholderReplacer repl(Placeholder, V.m_Sema);
      repl.Replace(V, New, Result);
      return;
    }

    if (isInsideLoop) {
      auto* Push = cast<CallExpr>(Result.getExpr());
      unsigned lastArg = Push->getNumArgs() - 1;
      Push->setArg(lastArg, V.m_Sema.DefaultLvalueConversion(New).get());
    } else if (isFnScope) {
      V.m_Sema.AddInitializerToDecl(Declaration, New, true);
      Declaration->setInitStyle(VarDecl::InitializationStyle::CInit);
      V.addToCurrentBlock(V.BuildDeclStmt(Declaration), direction::forward);
    } else {
      V.addToCurrentBlock(V.BuildOp(BO_Assign, Result.getExpr(), New),
                          direction::forward);
    }
  }

  ReverseModeVisitor::DelayedStoreResult
  ReverseModeVisitor::DelayedGlobalStoreAndRef(Expr* E, llvm::StringRef prefix,
                                               bool forceStore) {
    assert(E && "must be provided");
    if (!UsefulToStore(E)) {
      StmtDiff Ediff = Visit(E);
      Expr::EvalResult evalRes;
      return DelayedStoreResult{*this, Ediff,
                                /*Declaration=*/nullptr,
                                /*isInsideLoop=*/false,
                                /*isFnScope=*/false};
    }
    if (!forceStore && ShouldRecompute(E)) {
      // The value of the literal has no. It's given a very particular value for
      // easier debugging.
      Expr* PH = ConstantFolder::synthesizeLiteral(E->getType(), m_Context,
                                                   /*val=*/~0U);
      return DelayedStoreResult{
          *this,
          StmtDiff{PH, /*diff=*/nullptr, /*forwSweepDiff=*/nullptr, PH},
          /*Declaration=*/nullptr,
          /*isInsideLoop=*/false,
          /*isFnScope=*/false,
          /*pNeedsUpdate=*/true,
          /*pPlaceholder=*/PH};
    }
    if (isInsideLoop) {
      Expr* dummy = E;
      auto CladTape = MakeCladTapeFor(dummy);
      Expr* Push = CladTape.Push;
      Expr* Pop = CladTape.Pop;
      return DelayedStoreResult{*this,
                                StmtDiff{Push, nullptr, nullptr, Pop},
                                /*Declaration=*/nullptr,
                                /*isInsideLoop=*/true,
                                /*isFnScope=*/false,
                                /*pNeedsUpdate=*/true};
    }
    bool isFnScope = getCurrentScope()->isFunctionScope() ||
                     m_DiffReq.Mode == DiffMode::reverse_mode_forward_pass;
    VarDecl* VD = BuildGlobalVarDecl(
        getNonConstType(E->getType(), m_Context, m_Sema), prefix);
    Expr* Ref = BuildDeclRef(VD);
    if (!isFnScope)
      addToBlock(BuildDeclStmt(VD), m_Globals);
    // Return reference to the declaration instead of original expression.
    return DelayedStoreResult{*this,
                              StmtDiff{Ref, nullptr, nullptr, Ref},
                              /*Declaration=*/VD,
                              /*isInsideLoop=*/false,
                              /*isFnScope=*/isFnScope,
                              /*pNeedsUpdate=*/true};
  }

  ReverseModeVisitor::LoopCounter::LoopCounter(ReverseModeVisitor& RMV)
      : m_RMV(RMV) {
    ASTContext& C = m_RMV.m_Context;
    Expr* zero = ConstantFolder::synthesizeLiteral(C.getSizeType(), C,
                                                   /*val=*/0);
    m_Ref = m_RMV.GlobalStoreAndRef(zero, C.getSizeType(), "_t",
                                    /*force=*/true);
  }

  StmtDiff ReverseModeVisitor::VisitWhileStmt(const WhileStmt* WS) {
    beginBlock(direction::reverse);
    LoopCounter loopCounter(*this);

    // begin scope for while statement
    beginScope(Scope::DeclScope | Scope::ControlScope | Scope::BreakScope |
               Scope::ContinueScope);

    llvm::SaveAndRestore<bool> SaveIsInsideLoop(isInsideLoop);
    isInsideLoop = true;

    Expr* condClone = (WS->getCond() ? Clone(WS->getCond()) : nullptr);
    const VarDecl* condVarDecl = WS->getConditionVariable();
    StmtDiff condVarRes;
    if (condVarDecl)
      condVarRes = DifferentiateSingleStmt(WS->getConditionVariableDeclStmt());

    // compute condition result object for the forward pass `while`
    // statement.
    Sema::ConditionResult condResult;
    if (condVarDecl) {
      if (condVarRes.getStmt()) {
        if (isa<DeclStmt>(condVarRes.getStmt())) {
          Decl* condVarClone =
              cast<DeclStmt>(condVarRes.getStmt())->getSingleDecl();
          condResult = m_Sema.ActOnConditionVariable(
              condVarClone, noLoc, Sema::ConditionKind::Boolean);
        } else {
          condResult = m_Sema.ActOnCondition(getCurrentScope(), noLoc,
                                             cast<Expr>(condVarRes.getStmt()),
                                             Sema::ConditionKind::Boolean);
        }
      }
    } else {
      condResult = m_Sema.ActOnCondition(getCurrentScope(), noLoc, condClone,
                                         Sema::ConditionKind::Boolean);
    }

    const Stmt* body = WS->getBody();
    StmtDiff bodyDiff = DifferentiateLoopBody(body, loopCounter,
                                              condVarRes.getStmt_dx());
    // Create forward-pass `while` loop.
    Stmt* forwardWS = clad_compat::Sema_ActOnWhileStmt(m_Sema, condResult,
                                                       bodyDiff.getStmt())
                          .get();

    // Create reverse-pass `while` loop.
    Sema::ConditionResult CounterCondition = loopCounter
                                                 .getCounterConditionResult();
    Stmt* reverseWS = clad_compat::Sema_ActOnWhileStmt(m_Sema, CounterCondition,
                                                       bodyDiff.getStmt_dx())
                          .get();
    // for while statement
    endScope();
    addToCurrentBlock(reverseWS, direction::reverse);
    reverseWS = utils::unwrapIfSingleStmt(endBlock(direction::reverse));
    return {forwardWS, reverseWS};
  }

  StmtDiff ReverseModeVisitor::VisitDoStmt(const DoStmt* DS) {
    beginBlock(direction::reverse);
    LoopCounter loopCounter(*this);

    // begin scope for do statement
    beginScope(Scope::ContinueScope | Scope::BreakScope);

    llvm::SaveAndRestore<bool> SaveIsInsideLoop(isInsideLoop);
    isInsideLoop = true;

    Expr* clonedCond = (DS->getCond() ? Clone(DS->getCond()) : nullptr);

    const Stmt* body = DS->getBody();
    StmtDiff bodyDiff = DifferentiateLoopBody(body, loopCounter);

    // Create forward-pass `do-while` statement.
    Stmt* forwardDS = m_Sema
                          .ActOnDoStmt(/*DoLoc=*/noLoc, bodyDiff.getStmt(),
                                       /*WhileLoc=*/noLoc,
                                       /*CondLParen=*/noLoc, clonedCond,
                                       /*CondRParen=*/noLoc)
                          .get();

    // create reverse-pass `do-while` statement.
    Expr*
        counterCondition = loopCounter.getCounterConditionResult().get().second;
    Stmt* reverseDS = m_Sema
                          .ActOnDoStmt(/*DoLoc=*/noLoc, bodyDiff.getStmt_dx(),
                                       /*WhileLoc=*/noLoc,
                                       /*CondLParen=*/noLoc, counterCondition,
                                       /*CondRParen=*/noLoc)
                          .get();
    // for do-while statement
    endScope();
    addToCurrentBlock(reverseDS, direction::reverse);
    reverseDS = utils::unwrapIfSingleStmt(endBlock(direction::reverse));
    return {forwardDS, reverseDS};
  }

  // Basic idea used for differentiating switch statement is that in the reverse
  // pass, processing of the differentiated statments of the switch statement
  // body should start either from a `break` statement or from the last
  // statement of the switch statement body and always end at a switch
  // case/default statement.
  //
  // Therefore, here we keep track of which `break` was hit in the forward pass,
  // or if we no `break` statement was hit at all in a variable or clad tape.
  // This information is further used by an auxilliary switch statement in the
  // reverse pass to jump the execution to the correct point (that is,
  // differentiated statement of the statement just before the `break` statement
  // that was hit in the forward pass)
  StmtDiff ReverseModeVisitor::VisitSwitchStmt(const SwitchStmt* SS) {
    // Scope and blocks for the compound statement that encloses the switch
    // statement in both the forward and the reverse pass. Block is required
    // for handling condition variable and switch-init statement.
    beginScope(Scope::DeclScope);
    beginBlock(direction::forward);
    beginBlock(direction::reverse);

    // Handles switch init statement
    if (SS->getInit()) {
      StmtDiff switchInitDiff = DifferentiateSingleStmt(SS->getInit());
      addToCurrentBlock(switchInitDiff.getStmt(), direction::forward);
      addToCurrentBlock(switchInitDiff.getStmt_dx(), direction::reverse);
    }

    // Handles condition variable
    if (SS->getConditionVariable()) {
      StmtDiff condVarDiff =
          DifferentiateSingleStmt(SS->getConditionVariableDeclStmt());
      addToCurrentBlock(condVarDiff.getStmt(), direction::forward);
      addToCurrentBlock(condVarDiff.getStmt_dx(), direction::reverse);
    }

    StmtDiff condDiff = DifferentiateSingleStmt(SS->getCond());
    addToCurrentBlock(condDiff.getStmt(), direction::forward);
    addToCurrentBlock(condDiff.getStmt_dx(), direction::reverse);
    Expr* condExpr = GlobalStoreAndRef(condDiff.getExpr(), "_cond");

    auto* activeBreakContHandler = PushBreakContStmtHandler(
        /*forSwitchStmt=*/true);
    activeBreakContHandler->BeginCFSwitchStmtScope();
    auto* SSData = PushSwitchStmtInfo();

    SSData->switchStmtCond = condExpr;

    // scope for the switch statement body.
    beginScope(Scope::DeclScope);

    const Stmt* body = SS->getBody();
    StmtDiff bodyDiff = nullptr;
    if (isa<CompoundStmt>(body))
      bodyDiff = Visit(body);
    else
      bodyDiff = DifferentiateSingleStmt(body);

    // Each switch case statement of the original function gets transformed to
    // an if condition in the reverse pass. The if condition decides at runtime
    // whether the processing of the differentiated statements of the switch
    // statement body should stop or continue. This is based on the fact that
    // processing of statements of switch statement body always starts at a case
    // statement. For example,
    // ```
    // case 3:
    // ```
    // gets transformed to,
    //
    // ```
    // if (3 == _cond)
    //   break;
    // ```
    //
    // This kind of if expression cannot by easily formed for the default
    // statement, thus, we instead compare value of the switch condition with
    // the values of all the case statements to determine if the default
    // statement was selected in the forward pass.
    // Therefore,
    //
    // ```
    // default:
    // ```
    //
    // will get transformed to something like,
    //
    // ```
    // if (_cond != 1 && _cond != 2 && _cond != 3)
    //   break;
    // ```
    if (SSData->defaultIfBreakExpr) {
      Expr* breakCond = nullptr;
      for (auto* SC : SSData->cases) {
        if (auto* CS = dyn_cast<CaseStmt>(SC)) {
          if (breakCond) {
            breakCond = BuildOp(BinaryOperatorKind::BO_LAnd, breakCond,
                                BuildOp(BinaryOperatorKind::BO_NE,
                                        SSData->switchStmtCond, CS->getLHS()));
          } else {
            breakCond = BuildOp(BinaryOperatorKind::BO_NE,
                                SSData->switchStmtCond, CS->getLHS());
          }
        }
      }
      if (!breakCond)
        breakCond = m_Sema.ActOnCXXBoolLiteral(noLoc, tok::kw_true).get();
      SSData->defaultIfBreakExpr->setCond(breakCond);
    }

    activeBreakContHandler->EndCFSwitchStmtScope();

    // If switch statement contains no cases, then, no statement of the switch
    // statement body will be processed in both the forward and the reverse
    // pass. Thus, we do not need to add them in the differentiated function.
    if (!(SSData->cases.empty())) {
      Sema::ConditionResult condRes = m_Sema.ActOnCondition(
          getCurrentScope(), noLoc, condExpr, Sema::ConditionKind::Switch);
      SwitchStmt* forwardSS =
          clad_compat::Sema_ActOnStartOfSwitchStmt(m_Sema, nullptr, condRes)
              .getAs<SwitchStmt>();
      activeBreakContHandler->UpdateForwAndRevBlocks(bodyDiff);

      // Registers all the cases to the switch statement.
      for (auto* SC : SSData->cases)
        forwardSS->addSwitchCase(SC);

      forwardSS =
          m_Sema.ActOnFinishSwitchStmt(noLoc, forwardSS, bodyDiff.getStmt())
              .getAs<SwitchStmt>();

      addToCurrentBlock(forwardSS, direction::forward);
      addToCurrentBlock(bodyDiff.getStmt_dx(), direction::reverse);
    }

    PopBreakContStmtHandler();
    PopSwitchStmtInfo();
    return {endBlock(direction::forward), endBlock(direction::reverse)};
  }

  StmtDiff ReverseModeVisitor::VisitCaseStmt(const CaseStmt* CS) {
    beginBlock(direction::forward);
    beginBlock(direction::reverse);
    SwitchStmtInfo* SSData = GetActiveSwitchStmtInfo();

    Expr* lhsClone = (CS->getLHS() ? Clone(CS->getLHS()) : nullptr);
    Expr* rhsClone = (CS->getRHS() ? Clone(CS->getRHS()) : nullptr);

    auto* newSC = CaseStmt::Create(m_Sema.getASTContext(), lhsClone, rhsClone,
                                   noLoc, noLoc, noLoc);

    Expr* ifCond = BuildOp(BinaryOperatorKind::BO_EQ, newSC->getLHS(),
                           SSData->switchStmtCond);
    Stmt* ifThen = m_Sema.ActOnBreakStmt(noLoc, getCurrentScope()).get();
    Stmt* ifBreakExpr = clad_compat::IfStmt_Create(
        m_Context, noLoc, false, nullptr, nullptr, ifCond, noLoc, noLoc, ifThen,
        noLoc, nullptr);
    SSData->cases.push_back(newSC);
    addToCurrentBlock(ifBreakExpr, direction::reverse);
    addToCurrentBlock(newSC, direction::forward);
    auto diff = DifferentiateSingleStmt(CS->getSubStmt());
    utils::SetSwitchCaseSubStmt(newSC, diff.getStmt());
    addToCurrentBlock(diff.getStmt_dx(), direction::reverse);
    return {endBlock(direction::forward), endBlock(direction::reverse)};
  }

  StmtDiff ReverseModeVisitor::VisitDefaultStmt(const DefaultStmt* DS) {
    beginBlock(direction::reverse);
    beginBlock(direction::forward);
    auto* SSData = GetActiveSwitchStmtInfo();
    auto* newDefaultStmt =
        new (m_Sema.getASTContext()) DefaultStmt(noLoc, noLoc, nullptr);
    Stmt* ifThen = m_Sema.ActOnBreakStmt(noLoc, getCurrentScope()).get();
    Stmt* ifBreakExpr = clad_compat::IfStmt_Create(
        m_Context, noLoc, false, nullptr, nullptr, nullptr, noLoc, noLoc,
        ifThen, noLoc, nullptr);
    SSData->cases.push_back(newDefaultStmt);
    SSData->defaultIfBreakExpr = cast<IfStmt>(ifBreakExpr);
    addToCurrentBlock(ifBreakExpr, direction::reverse);
    addToCurrentBlock(newDefaultStmt, direction::forward);
    auto diff = DifferentiateSingleStmt(DS->getSubStmt());
    utils::SetSwitchCaseSubStmt(newDefaultStmt, diff.getStmt());
    addToCurrentBlock(diff.getStmt_dx(), direction::reverse);
    return {endBlock(direction::forward), endBlock(direction::reverse)};
  }

  StmtDiff ReverseModeVisitor::DifferentiateLoopBody(const Stmt* body,
                                                     LoopCounter& loopCounter,
                                                     Stmt* condVarDiff,
                                                     Stmt* forLoopIncDiff,
                                                     bool isForLoop) {
    Expr* counterIncrement = loopCounter.getCounterIncrement();
    auto* activeBreakContHandler = PushBreakContStmtHandler();
    activeBreakContHandler->BeginCFSwitchStmtScope();
    m_LoopBlock.emplace_back();
    // differentiate loop body and add loop increment expression
    // in the forward block.
    StmtDiff bodyDiff = nullptr;
    if (isa<CompoundStmt>(body)) {
      bodyDiff = Visit(body);
      beginBlock(direction::forward);
      addToCurrentBlock(counterIncrement);
      for (Stmt* S : cast<CompoundStmt>(bodyDiff.getStmt())->body())
        addToCurrentBlock(S);
      bodyDiff = {endBlock(direction::forward), bodyDiff.getStmt_dx()};
    } else {
      // for forward-pass loop statement body
      beginScope(Scope::DeclScope);
      beginBlock(direction::forward);
      addToCurrentBlock(counterIncrement);
      if (m_ExternalSource)
        m_ExternalSource->ActBeforeDifferentiatingSingleStmtLoopBody();
      bodyDiff = DifferentiateSingleStmt(body, /*dfdS=*/nullptr);
      addToCurrentBlock(bodyDiff.getStmt());
      if (m_ExternalSource)
        m_ExternalSource->ActAfterProcessingSingleStmtBodyInVisitForLoop();

      Stmt* reverseBlock = utils::unwrapIfSingleStmt(bodyDiff.getStmt_dx());
      bodyDiff = {endBlock(direction::forward), reverseBlock};
      // for forward-pass loop statement body
      endScope();
    }
    Stmts revLoopBlock = m_LoopBlock.back();
    utils::AppendIndividualStmts(revLoopBlock, bodyDiff.getStmt_dx());
    if (!revLoopBlock.empty())
      bodyDiff.updateStmtDx(MakeCompoundStmt(revLoopBlock));
    m_LoopBlock.pop_back();

    // Increment statement in the for-loop is only executed if the iteration
    // did not end with a break/continue statement. Therefore, forLoopIncDiff
    // should be inside the last switch case in the reverse pass.
    if (forLoopIncDiff) {
      if (bodyDiff.getStmt_dx()) {
        bodyDiff.updateStmtDx(utils::PrependAndCreateCompoundStmt(
            m_Context, bodyDiff.getStmt_dx(), forLoopIncDiff));
      } else {
        bodyDiff.updateStmtDx(forLoopIncDiff);
      }
    }

    activeBreakContHandler->EndCFSwitchStmtScope();
    activeBreakContHandler->UpdateForwAndRevBlocks(bodyDiff);
    PopBreakContStmtHandler();

    Expr* counterDecrement = loopCounter.getCounterDecrement();

    // Create reverse pass loop body statements by arranging various
    // differentiated statements in the correct order.
    // Order used:
    //
    // 1) `for` loop increment differentiation statements
    // 2) loop body differentiation statements
    // 3) condition variable differentiation statements
    // 4) counter decrement expression
    beginBlock(direction::reverse);
    // `for` loops have counter decrement expression in the
    // loop iteration-expression.
    if (!isForLoop)
      addToCurrentBlock(counterDecrement, direction::reverse);
    addToCurrentBlock(condVarDiff, direction::reverse);
    addToCurrentBlock(bodyDiff.getStmt_dx(), direction::reverse);
    bodyDiff = {bodyDiff.getStmt(),
                utils::unwrapIfSingleStmt(endBlock(direction::reverse))};
    return bodyDiff;
  }

  StmtDiff ReverseModeVisitor::VisitContinueStmt(const ContinueStmt* CS) {
    beginBlock(direction::forward);
    Stmt* newCS = m_Sema.ActOnContinueStmt(noLoc, getCurrentScope()).get();
    auto* activeBreakContHandler = GetActiveBreakContStmtHandler();
    Stmt* CFCaseStmt = activeBreakContHandler->GetNextCFCaseStmt();
    Stmt* pushExprToCurrentCase = activeBreakContHandler
                                      ->CreateCFTapePushExprToCurrentCase();
    addToCurrentBlock(pushExprToCurrentCase);
    addToCurrentBlock(newCS);
    return {endBlock(direction::forward), CFCaseStmt};
  }

  StmtDiff ReverseModeVisitor::VisitBreakStmt(const BreakStmt* BS) {
    beginBlock(direction::forward);
    Stmt* newBS = m_Sema.ActOnBreakStmt(noLoc, getCurrentScope()).get();
    auto* activeBreakContHandler = GetActiveBreakContStmtHandler();
    Stmt* CFCaseStmt = activeBreakContHandler->GetNextCFCaseStmt();
    Stmt* pushExprToCurrentCase = activeBreakContHandler
                                      ->CreateCFTapePushExprToCurrentCase();
    if (isInsideLoop && !activeBreakContHandler->m_IsInvokedBySwitchStmt) {
      Expr* tapeBackExprForCurrentCase =
          activeBreakContHandler->CreateCFTapeBackExprForCurrentCase();
      if (m_CurrentBreakFlagExpr) {
        m_CurrentBreakFlagExpr =
            BuildOp(BinaryOperatorKind::BO_LAnd, m_CurrentBreakFlagExpr,
                    tapeBackExprForCurrentCase);

      } else {
        m_CurrentBreakFlagExpr = tapeBackExprForCurrentCase;
      }
    }
    addToCurrentBlock(pushExprToCurrentCase);
    addToCurrentBlock(newBS);
    return {endBlock(direction::forward), CFCaseStmt};
  }

  Expr* ReverseModeVisitor::BreakContStmtHandler::CreateSizeTLiteralExpr(
      std::size_t value) {
    ASTContext& C = m_RMV.m_Context;
    auto* literalExpr =
        ConstantFolder::synthesizeLiteral(C.getSizeType(), C, value);
    return literalExpr;
  }

  void ReverseModeVisitor::BreakContStmtHandler::InitializeCFTape() {
    assert(!m_ControlFlowTape && "InitializeCFTape() should not be called if "
                                 "m_ControlFlowTape is already initialized");

    auto* zeroLiteral = CreateSizeTLiteralExpr(0);
    m_ControlFlowTape.reset(
        new CladTapeResult(m_RMV.MakeCladTapeFor(zeroLiteral)));
  }

  Expr* ReverseModeVisitor::BreakContStmtHandler::CreateCFTapePushExpr(
      std::size_t value) {
    Expr* pushDRE = m_RMV.GetCladTapePushDRE();
    Expr* callArgs[] = {m_ControlFlowTape->Ref, CreateSizeTLiteralExpr(value)};
    Expr* pushExpr = m_RMV.m_Sema
                         .ActOnCallExpr(m_RMV.getCurrentScope(), pushDRE, noLoc,
                                        callArgs, noLoc)
                         .get();
    return pushExpr;
  }

  void
  ReverseModeVisitor::BreakContStmtHandler::BeginCFSwitchStmtScope() const {
    m_RMV.beginScope(Scope::SwitchScope | Scope::ControlScope |
                     Scope::BreakScope | Scope::DeclScope);
  }

  void ReverseModeVisitor::BreakContStmtHandler::EndCFSwitchStmtScope() const {
    m_RMV.endScope();
  }

  CaseStmt* ReverseModeVisitor::BreakContStmtHandler::GetNextCFCaseStmt() {
    ++m_CaseCounter;
    auto* counterLiteral = CreateSizeTLiteralExpr(m_CaseCounter);
    CaseStmt* CS = CaseStmt::Create(m_RMV.m_Context, counterLiteral, nullptr,
                                    noLoc, noLoc, noLoc);

    // Initialise switch case statements with null statement because it is
    // necessary for switch case statements to have a substatement but it
    // is possible that there are no statements after the corresponding
    // break/continue statement. It's also easier to just set null statement
    // as substatement instead of keeping track of switch cases and
    // corresponding next statements.
    CS->setSubStmt(m_RMV.m_Sema.ActOnNullStmt(noLoc).get());

    m_SwitchCases.push_back(CS);
    return CS;
  }

  Expr* ReverseModeVisitor::BreakContStmtHandler::
      CreateCFTapeBackExprForCurrentCase() {
    return m_RMV.BuildOp(
        BinaryOperatorKind::BO_NE, m_ControlFlowTape->Last(),
        ConstantFolder::synthesizeLiteral(m_RMV.m_Context.IntTy,
                                          m_RMV.m_Context, m_CaseCounter));
  }

  Stmt* ReverseModeVisitor::BreakContStmtHandler::
      CreateCFTapePushExprToCurrentCase() {
    if (!m_ControlFlowTape)
      InitializeCFTape();
    return CreateCFTapePushExpr(m_CaseCounter);
  }

  void ReverseModeVisitor::BreakContStmtHandler::UpdateForwAndRevBlocks(
      StmtDiff& bodyDiff) {
    if (m_SwitchCases.empty() && !m_IsInvokedBySwitchStmt)
      return;

    // Add case statement in the beginning of the reverse block
    // and corresponding push expression for this case statement
    // at the end of the forward block to cover the case when no
    // `break`/`continue` statements are hit.
    auto* lastSC = GetNextCFCaseStmt();
    auto* pushExprToCurrentCase = CreateCFTapePushExprToCurrentCase();

    Stmt* forwBlock = nullptr;
    Stmt* revBlock = nullptr;

    forwBlock = utils::AppendAndCreateCompoundStmt(m_RMV.m_Context,
                                                   bodyDiff.getStmt(),
                                                   pushExprToCurrentCase);
    revBlock = utils::PrependAndCreateCompoundStmt(m_RMV.m_Context,
                                                   bodyDiff.getStmt_dx(),
                                                   lastSC);

    bodyDiff = {forwBlock, revBlock};

    auto condResult = m_RMV.m_Sema.ActOnCondition(m_RMV.getCurrentScope(),
                                                  noLoc, m_ControlFlowTape->Pop,
                                                  Sema::ConditionKind::Switch);
    auto* CFSS = clad_compat::Sema_ActOnStartOfSwitchStmt(m_RMV.m_Sema, nullptr,
                                                          condResult)
                     .getAs<SwitchStmt>();
    // Registers all the switch cases
    for (auto* SC : m_SwitchCases)
      CFSS->addSwitchCase(SC);
    m_RMV.m_Sema.ActOnFinishSwitchStmt(noLoc, CFSS, bodyDiff.getStmt_dx());

    bodyDiff = {bodyDiff.getStmt(), CFSS};
  }

  void ReverseModeVisitor::AddExternalSource(ExternalRMVSource& source) {
    if (!m_ExternalSource)
      m_ExternalSource = new MultiplexExternalRMVSource();
    source.InitialiseRMV(*this);
    m_ExternalSource->AddSource(source);
  }

  StmtDiff ReverseModeVisitor::VisitCXXThisExpr(const CXXThisExpr* CTE) {
    Expr* clonedCTE = Clone(CTE);
    return {clonedCTE, m_ThisExprDerivative};
  }

  StmtDiff ReverseModeVisitor::VisitCXXNewExpr(const clang::CXXNewExpr* CNE) {
    StmtDiff initializerDiff;
    if (CNE->hasInitializer())
      initializerDiff = Visit(CNE->getInitializer(), dfdx());

    Expr* clonedArraySizeE = nullptr;
    Expr* derivedArraySizeE = nullptr;
    if (CNE->getArraySize()) {
      clonedArraySizeE =
          Visit(clad_compat::ArraySize_GetValue(CNE->getArraySize())).getExpr();
      // Array size is a non-differentiable expression, thus the original value
      // should be used in both the cloned and the derived statements.
      derivedArraySizeE = Clone(clonedArraySizeE);
    }
    Expr* clonedNewE = utils::BuildCXXNewExpr(
        m_Sema, CNE->getAllocatedType(), clonedArraySizeE,
        initializerDiff.getExpr(), CNE->getAllocatedTypeSourceInfo());
    Expr* diffInit = initializerDiff.getExpr_dx();
    if (!diffInit) {
      // we should initialize it implicitly using ParenListExpr.
      diffInit = m_Sema.ActOnParenListExpr(noLoc, noLoc, {}).get();
    }
    Expr* derivedNewE = utils::BuildCXXNewExpr(
        m_Sema, CNE->getAllocatedType(), derivedArraySizeE, diffInit,
        CNE->getAllocatedTypeSourceInfo());
    return {clonedNewE, derivedNewE};
  }

  StmtDiff
  ReverseModeVisitor::VisitCXXDeleteExpr(const clang::CXXDeleteExpr* CDE) {
    StmtDiff argDiff = Visit(CDE->getArgument());
    Expr* clonedDeleteE =
        m_Sema
            .ActOnCXXDelete(noLoc, CDE->isGlobalDelete(), CDE->isArrayForm(),
                            argDiff.getExpr())
            .get();
    Expr* derivedDeleteE =
        m_Sema
            .ActOnCXXDelete(noLoc, CDE->isGlobalDelete(), CDE->isArrayForm(),
                            argDiff.getExpr_dx())
            .get();
    // create a compound statement containing both the cloned and the derived
    // delete expressions.
    CompoundStmt* CS = MakeCompoundStmt({clonedDeleteE, derivedDeleteE});
    m_DeallocExprs.push_back(CS);
    return {nullptr, nullptr};
  }

  StmtDiff
  ReverseModeVisitor::VisitCXXConstructExpr(const CXXConstructExpr* CE) {

    llvm::SmallVector<Expr*, 4> primalArgs;
    llvm::SmallVector<Expr*, 4> adjointArgs;
    llvm::SmallVector<Expr*, 4> reverseForwAdjointArgs;
    // It is used to store '_r0' temporary gradient variables that are used for
    // differentiating non-reference args.
    llvm::SmallVector<Stmt*, 4> prePullbackCallStmts;

    // Insertion point is required because we need to insert pullback call
    // before the statements inserted by 'Visit(arg, ...)' calls for arguments.
    std::size_t insertionPoint = getCurrentBlock(direction::reverse).size();

    // FIXME: Restore arguments passed as non-const reference.
    for (const auto* arg : CE->arguments()) {
      QualType ArgTy = arg->getType();
      StmtDiff argDiff{};
      Expr* adjointArg = nullptr;
      if (utils::IsReferenceOrPointerArg(arg->IgnoreParenImpCasts())) {
        argDiff = Visit(arg);
        adjointArg = argDiff.getExpr_dx();
      } else {
        // non-reference arguments are differentiated as follows:
        //
        // primal code:
        // ```
        // SomeClass c(u, ...);
        // ```
        //
        // Derivative code:
        // ```
        // // forward pass
        // ...
        // // reverse pass
        // double _r0 = 0;
        // SomeClass_pullback(c, u, ..., &_d_c, &_r0, ...);
        // _d_u += _r0;
        QualType dArgTy = getNonConstType(ArgTy, m_Context, m_Sema);
        VarDecl* dArgDecl = BuildVarDecl(dArgTy, "_r", getZeroInit(dArgTy));
        prePullbackCallStmts.push_back(BuildDeclStmt(dArgDecl));
        adjointArg = BuildDeclRef(dArgDecl);
        argDiff = Visit(arg, BuildDeclRef(dArgDecl));
      }

      if (utils::isArrayOrPointerType(ArgTy)) {
        reverseForwAdjointArgs.push_back(adjointArg);
        adjointArgs.push_back(adjointArg);
      } else {
        if (utils::IsReferenceOrPointerArg(arg->IgnoreParenImpCasts()))
          reverseForwAdjointArgs.push_back(adjointArg);
        else
          reverseForwAdjointArgs.push_back(getZeroInit(ArgTy));
        adjointArgs.push_back(BuildOp(UnaryOperatorKind::UO_AddrOf, adjointArg,
                                      m_DiffReq->getLocation()));
      }
      primalArgs.push_back(argDiff.getExpr());
    }

    // Try to create a pullback constructor call
    llvm::SmallVector<Expr*, 4> pullbackArgs;
    QualType recordType =
        m_Context.getRecordType(CE->getConstructor()->getParent());
    QualType recordPointerType = m_Context.getPointerType(recordType);
    // thisE = object being created by this constructor call.
    // dThisE = adjoint of the object being created by this constructor call.
    //
    // We cannot fill these args yet because these objects have not yet been
    // created. The caller which triggers 'VisitCXXConstructExpr' is
    // responsible for updating these args.
    Expr* thisE = getZeroInit(recordPointerType);
    Expr* dThisE = getZeroInit(recordPointerType);

    pullbackArgs.push_back(thisE);
    pullbackArgs.append(primalArgs.begin(), primalArgs.end());
    pullbackArgs.push_back(dThisE);
    pullbackArgs.append(adjointArgs.begin(), adjointArgs.end());

    Stmts& curRevBlock = getCurrentBlock(direction::reverse);
    Stmts::iterator it = std::begin(curRevBlock) + insertionPoint;
    curRevBlock.insert(it, prePullbackCallStmts.begin(),
                       prePullbackCallStmts.end());
    it += prePullbackCallStmts.size();
    std::string customPullbackName = "constructor_pullback";
    if (Expr* customPullbackCall =
            m_Builder.BuildCallToCustomDerivativeOrNumericalDiff(
                customPullbackName, pullbackArgs, getCurrentScope(),
                const_cast<DeclContext*>(
                    CE->getConstructor()->getDeclContext()))) {
      curRevBlock.insert(it, customPullbackCall);
      if (m_TrackConstructorPullbackInfo) {
        setConstructorPullbackCallInfo(llvm::cast<CallExpr>(customPullbackCall),
                                       primalArgs.size() + 1);
        m_TrackConstructorPullbackInfo = false;
      }
    }
    // FIXME: If no compatible custom constructor pullback is found then try
    // to automatically differentiate the constructor.

    // Create the constructor call in the forward-pass, or creates
    // 'constructor_forw' call if possible.

    // This works as follows:
    //
    // primal code:
    // ```
    // SomeClass c(u, v);
    // ```
    //
    // adjoint code:
    // ```
    // // forward-pass
    // clad::ValueAndAdjoint<SomeClass, SomeClass> _t0 =
    //   constructor_forw(clad::ConstructorReverseForwTag<SomeClass>{}, u, v,
    //     _d_u, _d_v);
    // SomeClass _d_c = _t0.adjoint;
    // SomeClass c = _t0.value;
    // ```
    if (Expr* customReverseForwFnCall = BuildCallToCustomForwPassFn(
            CE->getConstructor(), primalArgs, reverseForwAdjointArgs,
            /*baseExpr=*/nullptr)) {
      Expr* callRes = StoreAndRef(customReverseForwFnCall);
      Expr* val =
          utils::BuildMemberExpr(m_Sema, getCurrentScope(), callRes, "value");
      Expr* adjoint =
          utils::BuildMemberExpr(m_Sema, getCurrentScope(), callRes, "adjoint");
      return {val, nullptr, adjoint};
    }

    Expr* clonedArgsE = nullptr;

    if (CE->getNumArgs() != 1) {
      if (CE->isListInitialization()) {
        clonedArgsE = m_Sema.ActOnInitList(noLoc, primalArgs, noLoc).get();
      } else {
        if (CE->getNumArgs() == 0) {
          // ParenList is empty -- default initialisation.
          // Passing empty parenList here will silently cause 'most vexing
          // parse' issue.
          return StmtDiff();
        }
        clonedArgsE = m_Sema.ActOnParenListExpr(noLoc, noLoc, primalArgs).get();
      }
    } else {
      clonedArgsE = primalArgs[0];
    }
    // `CXXConstructExpr` node will be created automatically by passing these
    // initialiser to higher level `ActOn`/`Build` Sema functions.
    return {clonedArgsE};
  }

  StmtDiff ReverseModeVisitor::VisitMaterializeTemporaryExpr(
      const clang::MaterializeTemporaryExpr* MTE) {
    // `MaterializeTemporaryExpr` node will be created automatically if it is
    // required by `ActOn`/`Build` Sema functions.
    StmtDiff MTEDiff = Visit(clad_compat::GetSubExpr(MTE), dfdx());
    return MTEDiff;
  }

  StmtDiff ReverseModeVisitor::VisitSubstNonTypeTemplateParmExpr(
      const clang::SubstNonTypeTemplateParmExpr* NTTP) {
    return Visit(NTTP->getReplacement());
  }

  StmtDiff ReverseModeVisitor::VisitUnaryExprOrTypeTraitExpr(
      const clang::UnaryExprOrTypeTraitExpr* UE) {
    return {Clone(UE), Clone(UE)};
  }

  DeclDiff<StaticAssertDecl> ReverseModeVisitor::DifferentiateStaticAssertDecl(
      const clang::StaticAssertDecl* SAD) {
    return DeclDiff<StaticAssertDecl>(nullptr, nullptr);
  }

  QualType ReverseModeVisitor::GetParameterDerivativeType(QualType yType,
                                                          QualType xType) {

    assert((m_DiffReq.Mode != DiffMode::reverse || yType->isRealType()) &&
           "yType should be a non-reference builtin-numerical scalar type!!");
    QualType xValueType = utils::GetValueType(xType);
    // derivative variables should always be of non-const type.
    xValueType.removeLocalConst();
    QualType nonRefXValueType = xValueType.getNonReferenceType();
    return m_Context.getPointerType(nonRefXValueType);
  }

  StmtDiff ReverseModeVisitor::VisitCXXStaticCastExpr(
      const clang::CXXStaticCastExpr* SCE) {
    StmtDiff subExprDiff = Visit(SCE->getSubExpr(), dfdx());
    return subExprDiff;
  }

  clang::QualType ReverseModeVisitor::ComputeAdjointType(clang::QualType T) {
    if (T->isReferenceType()) {
      QualType TValueType = utils::GetValueType(T);
      TValueType.removeLocalConst();
      return m_Context.getPointerType(TValueType);
    }
    T.removeLocalConst();
    return T;
  }

  clang::QualType ReverseModeVisitor::ComputeParamType(clang::QualType T) {
    QualType TValueType = utils::GetValueType(T);
    return m_Context.getPointerType(TValueType);
  }

  llvm::SmallVector<clang::QualType, 8>
  ReverseModeVisitor::ComputeParamTypes(const DiffParams& diffParams) {
    llvm::SmallVector<clang::QualType, 8> paramTypes;
    paramTypes.reserve(m_DiffReq->getNumParams() * 2);
    for (auto* PVD : m_DiffReq->parameters())
      paramTypes.push_back(PVD->getType());
    // TODO: Add DiffMode::experimental_pullback support here as well.
    if (m_DiffReq.Mode == DiffMode::reverse ||
        m_DiffReq.Mode == DiffMode::experimental_pullback) {
      QualType effectiveReturnType =
          m_DiffReq->getReturnType().getNonReferenceType();
      if (m_DiffReq.Mode == DiffMode::experimental_pullback) {
        // FIXME: Generally, we use the function's return type as the argument's
        // derivative type. We cannot follow this strategy for `void` function
        // return type. Thus, temporarily use `double` type as the placeholder
        // type for argument derivatives. We should think of a more uniform and
        // consistent solution to this problem. One effective strategy that may
        // hold well: If we are differentiating a variable of type Y with
        // respect to variable of type X, then the derivative should be of type
        // X. Check this related issue for more details:
        // https://github.com/vgvassilev/clad/issues/385
        if (effectiveReturnType->isVoidType() ||
            effectiveReturnType->isPointerType())
          effectiveReturnType = m_Context.DoubleTy;
        else
          paramTypes.push_back(effectiveReturnType);
      }

      if (const auto* MD = dyn_cast<CXXMethodDecl>(m_DiffReq.Function)) {
        const CXXRecordDecl* RD = MD->getParent();
        if (MD->isInstance() && !RD->isLambda()) {
          QualType thisType = MD->getThisType();
          paramTypes.push_back(
              GetParameterDerivativeType(effectiveReturnType, thisType));
        }
      }

      for (auto* PVD : m_DiffReq->parameters()) {
        const auto* it =
            std::find(std::begin(diffParams), std::end(diffParams), PVD);
        if (it != std::end(diffParams))
          paramTypes.push_back(ComputeParamType(PVD->getType()));
      }
    } else if (m_DiffReq.Mode == DiffMode::jacobian) {
      std::size_t lastArgIdx = m_DiffReq->getNumParams() - 1;
      QualType derivativeParamType =
          m_DiffReq->getParamDecl(lastArgIdx)->getType();
      paramTypes.push_back(derivativeParamType);
    }
    return paramTypes;
  }

  llvm::SmallVector<clang::ParmVarDecl*, 8>
  ReverseModeVisitor::BuildParams(DiffParams& diffParams) {
    llvm::SmallVector<clang::ParmVarDecl*, 8> params;
    llvm::SmallVector<clang::ParmVarDecl*, 8> paramDerivatives;
    params.reserve(m_DiffReq->getNumParams() + diffParams.size());
    const auto* derivativeFnType =
        cast<FunctionProtoType>(m_Derivative->getType());
    std::size_t dParamTypesIdx = m_DiffReq->getNumParams();

    if (m_DiffReq.Mode == DiffMode::experimental_pullback &&
        !m_DiffReq->getReturnType()->isVoidType() &&
        !m_DiffReq->getReturnType()->isPointerType()) {
      ++dParamTypesIdx;
    }

    if (const auto* MD = dyn_cast<CXXMethodDecl>(m_DiffReq.Function)) {
      const CXXRecordDecl* RD = MD->getParent();
      if (m_DiffReq.Mode != DiffMode::jacobian && MD->isInstance() &&
          !RD->isLambda()) {
        auto* thisDerivativePVD = utils::BuildParmVarDecl(
            m_Sema, m_Derivative, CreateUniqueIdentifier("_d_this"),
            derivativeFnType->getParamType(dParamTypesIdx));
        paramDerivatives.push_back(thisDerivativePVD);

        if (thisDerivativePVD->getIdentifier())
          m_Sema.PushOnScopeChains(thisDerivativePVD, getCurrentScope(),
                                   /*AddToContext=*/false);

        // This can instantiate an array_ref and needs a fake source location.
        SourceLocation fakeLoc = utils::GetValidSLoc(m_Sema);
        Expr* deref = BuildOp(UnaryOperatorKind::UO_Deref,
                              BuildDeclRef(thisDerivativePVD), fakeLoc);
        m_ThisExprDerivative = utils::BuildParenExpr(m_Sema, deref);
        ++dParamTypesIdx;
      }
    }

    for (auto* PVD : m_DiffReq->parameters()) {
      auto* newPVD = utils::BuildParmVarDecl(
          m_Sema, m_Derivative, PVD->getIdentifier(), PVD->getType(),
          PVD->getStorageClass(), /*DefArg=*/nullptr, PVD->getTypeSourceInfo());
      params.push_back(newPVD);

      if (newPVD->getIdentifier())
        m_Sema.PushOnScopeChains(newPVD, getCurrentScope(),
                                 /*AddToContext=*/false);

      auto* it = std::find(std::begin(diffParams), std::end(diffParams), PVD);
      if (it != std::end(diffParams)) {
        *it = newPVD;
        if (m_DiffReq.Mode == DiffMode::reverse ||
            m_DiffReq.Mode == DiffMode::experimental_pullback) {
          QualType dType = derivativeFnType->getParamType(dParamTypesIdx);
          IdentifierInfo* dII =
              CreateUniqueIdentifier("_d_" + PVD->getNameAsString());
          auto* dPVD = utils::BuildParmVarDecl(m_Sema, m_Derivative, dII, dType,
                                               PVD->getStorageClass());
          paramDerivatives.push_back(dPVD);
          ++dParamTypesIdx;

          if (dPVD->getIdentifier())
            m_Sema.PushOnScopeChains(dPVD, getCurrentScope(),
                                     /*AddToContext=*/false);

          if (utils::isArrayOrPointerType(PVD->getType())) {
            m_Variables[*it] = (Expr*)BuildDeclRef(dPVD);
          } else {
            QualType valueType = dPVD->getType()->getPointeeType();
            m_Variables[*it] =
                BuildOp(UO_Deref, BuildDeclRef(dPVD), m_DiffReq->getLocation());
            // Add additional paranthesis if derivative is of record type
            // because `*derivative.someField` will be incorrectly evaluated if
            // the derived function is compiled standalone.
            if (valueType->isRecordType())
              m_Variables[*it] =
                  utils::BuildParenExpr(m_Sema, m_Variables[*it]);
          }
          m_ParamVarsWithDiff.emplace(*it);
        }
      }
    }

    if (m_DiffReq.Mode == DiffMode::experimental_pullback &&
        !m_DiffReq->getReturnType()->isVoidType() &&
        !m_DiffReq->getReturnType()->isPointerType()) {
      IdentifierInfo* pullbackParamII = CreateUniqueIdentifier("_d_y");
      QualType pullbackType =
          derivativeFnType->getParamType(m_DiffReq->getNumParams());
      ParmVarDecl* pullbackPVD = utils::BuildParmVarDecl(
          m_Sema, m_Derivative, pullbackParamII, pullbackType);
      paramDerivatives.insert(paramDerivatives.begin(), pullbackPVD);

      if (pullbackPVD->getIdentifier())
        m_Sema.PushOnScopeChains(pullbackPVD, getCurrentScope(),
                                 /*AddToContext=*/false);

      m_Pullback = BuildDeclRef(pullbackPVD);
      ++dParamTypesIdx;
    }

    if (m_DiffReq.Mode == DiffMode::jacobian) {
      IdentifierInfo* II = CreateUniqueIdentifier("jacobianMatrix");
      // FIXME: Why are we taking storageClass of `params.front()`?
      auto* dPVD = utils::BuildParmVarDecl(
          m_Sema, m_Derivative, II,
          derivativeFnType->getParamType(dParamTypesIdx),
          params.front()->getStorageClass());
      paramDerivatives.push_back(dPVD);
      if (dPVD->getIdentifier())
        m_Sema.PushOnScopeChains(dPVD, getCurrentScope(),
                                 /*AddToContext=*/false);
    }
    params.insert(params.end(), paramDerivatives.begin(),
                  paramDerivatives.end());
    // FIXME: If we do not consider diffParams as an independent argument for
    // jacobian mode, then we should keep diffParams list empty for jacobian
    // mode and thus remove the if condition.
    if (m_DiffReq.Mode == DiffMode::reverse ||
        m_DiffReq.Mode == DiffMode::experimental_pullback)
      m_IndependentVars.insert(m_IndependentVars.end(), diffParams.begin(),
                               diffParams.end());
    return params;
  }

  Expr* ReverseModeVisitor::BuildCallToCustomForwPassFn(
      const FunctionDecl* FD, llvm::ArrayRef<Expr*> primalArgs,
      llvm::ArrayRef<clang::Expr*> derivedArgs, Expr* baseExpr) {
    std::string forwPassFnName =
        clad::utils::ComputeEffectiveFnName(FD) + "_reverse_forw";
    llvm::SmallVector<Expr*, 4> args;
    if (baseExpr) {
      baseExpr = BuildOp(UnaryOperatorKind::UO_AddrOf, baseExpr,
                         m_DiffReq->getLocation());
      args.push_back(baseExpr);
    }
    if (auto CD = llvm::dyn_cast<CXXConstructorDecl>(FD)) {
      const RecordDecl* RD = CD->getParent();
      QualType constructorReverseForwTagT =
          GetCladConstructorReverseForwTagOfType(m_Context.getRecordType(RD));
      Expr* constructorReverseForwTagArg =
          m_Sema
              .BuildCXXTypeConstructExpr(
                  m_Context.getTrivialTypeSourceInfo(
                      constructorReverseForwTagT, utils::GetValidSLoc(m_Sema)),
                  utils::GetValidSLoc(m_Sema), MultiExprArg{},
                  utils::GetValidSLoc(m_Sema),
                  /*ListInitialization=*/false)
              .get();
      args.push_back(constructorReverseForwTagArg);
    }
    args.append(primalArgs.begin(), primalArgs.end());
    args.append(derivedArgs.begin(), derivedArgs.end());
    Expr* customForwPassCE =
        m_Builder.BuildCallToCustomDerivativeOrNumericalDiff(
            forwPassFnName, args, getCurrentScope(),
            const_cast<DeclContext*>(FD->getDeclContext()));
    return customForwPassCE;
  }

  void ReverseModeVisitor::ConstructorPullbackCallInfo::updateThisParmArgs(
      Expr* thisE, Expr* dThisE) const {
    pullbackCE->setArg(0, thisE);
    pullbackCE->setArg(thisAdjointArgIdx, dThisE);
  }
} // end namespace clad
