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
#include "clang/AST/Expr.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/TemplateBase.h"
#include "clang/Basic/TokenKinds.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Overload.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaInternal.h"
#include "clang/Sema/Template.h"

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
    VD->setLocation(m_Function->getLocation());
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

  ReverseModeVisitor::ReverseModeVisitor(DerivativeBuilder& builder)
      : VisitorBase(builder), m_Result(nullptr) {}

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
    std::size_t totalDerivedParamsSize = m_Function->getNumParams() * 2;
    std::size_t numOfDerivativeParams = m_Function->getNumParams();

    // Account for the this pointer.
    if (isa<CXXMethodDecl>(m_Function) && !utils::IsStaticMethod(m_Function))
      ++numOfDerivativeParams;
    // All output parameters will be of type `clad::array_ref<void>`. These
    // parameters will be casted to correct type before the call to the actual
    // derived function.
    // We require each output parameter to be of same type in the overloaded
    // derived function due to limitations of generating the exact derived
    // function type at the compile-time (without clad plugin help).
    QualType outputParamType = GetCladArrayRefOfType(m_Context.VoidTy);

    llvm::SmallVector<QualType, 16> paramTypes;

    // Add types for representing original function parameters.
    for (auto* PVD : m_Function->parameters())
      paramTypes.push_back(PVD->getType());
    // Add types for representing parameter derivatives.
    // FIXME: We are assuming all function parameters are differentiable. We
    // should not make any such assumptions.
    for (std::size_t i = 0; i < numOfDerivativeParams; ++i)
      paramTypes.push_back(outputParamType);

    auto gradFuncOverloadEPI =
        dyn_cast<FunctionProtoType>(m_Function->getType())->getExtProtoInfo();
    QualType gradientFunctionOverloadType =
        m_Context.getFunctionType(m_Context.VoidTy, paramTypes,
                                  // Cast to function pointer.
                                  gradFuncOverloadEPI);

    auto* DC = const_cast<DeclContext*>(m_Function->getDeclContext());
    m_Sema.CurContext = DC;
    DeclWithContext gradientOverloadFDWC =
        m_Builder.cloneFunction(m_Function, *this, DC, noLoc, gradientNameInfo,
                                gradientFunctionOverloadType);
    FunctionDecl* gradientOverloadFD = gradientOverloadFDWC.first;

    beginScope(Scope::FunctionPrototypeScope | Scope::FunctionDeclarationScope |
               Scope::DeclScope);
    m_Sema.PushFunctionScope();
    m_Sema.PushDeclContext(getCurrentScope(), gradientOverloadFD);

    llvm::SmallVector<ParmVarDecl*, 4> overloadParams;
    llvm::SmallVector<Expr*, 4> callArgs;

    overloadParams.reserve(totalDerivedParamsSize);
    callArgs.reserve(gradientParams.size());

    for (auto* PVD : m_Function->parameters()) {
      auto* VD = utils::BuildParmVarDecl(
          m_Sema, gradientOverloadFD, PVD->getIdentifier(), PVD->getType(),
          PVD->getStorageClass(), /*defArg=*/nullptr, PVD->getTypeSourceInfo());
      overloadParams.push_back(VD);
      callArgs.push_back(BuildDeclRef(VD));
    }

    for (std::size_t i = 0; i < numOfDerivativeParams; ++i) {
      IdentifierInfo* II = nullptr;
      StorageClass SC = StorageClass::SC_None;
      std::size_t effectiveGradientIndex = m_Function->getNumParams() + i;
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
    for (std::size_t i = m_Function->getNumParams(); i < gradientParams.size();
         ++i) {
      auto* overloadParam = overloadParams[i];
      auto* gradientParam = gradientParams[i];

      auto* gradientVD =
          BuildGlobalVarDecl(gradientParam->getType(), gradientParam->getName(),
                             BuildDeclRef(overloadParam));
      callArgs.push_back(BuildDeclRef(gradientVD));
      addToCurrentBlock(BuildDeclStmt(gradientVD));
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

  DerivativeAndOverload
  ReverseModeVisitor::Derive(const FunctionDecl* FD,
                             const DiffRequest& request) {
    if (m_ExternalSource)
      m_ExternalSource->ActOnStartOfDerive();
    silenceDiags = !request.VerboseDiags;
    m_Function = FD;

    // reverse mode plugins may have request mode other than
    // `DiffMode::reverse`, but they still need the `DiffMode::reverse` mode
    // specific behaviour, because they are "reverse" mode plugins.
    m_Mode = DiffMode::reverse;
    if (request.Mode == DiffMode::jacobian)
      m_Mode = DiffMode::jacobian;
    m_Pullback =
        ConstantFolder::synthesizeLiteral(m_Context.IntTy, m_Context, 1);
    assert(m_Function && "Must not be null.");

    DiffParams args{};
    DiffInputVarsInfo DVI;
    if (request.Args) {
      DVI = request.DVI;
      for (const auto& dParam : DVI)
        args.push_back(dParam.param);
    }
    else
      std::copy(FD->param_begin(), FD->param_end(), std::back_inserter(args));
    if (args.empty())
      return {};

    if (m_ExternalSource)
      m_ExternalSource->ActAfterParsingDiffArgs(request, args);
    // Save the type of the output parameter(s) that is add by clad to the
    // derived function
    if (request.Mode == DiffMode::jacobian) {
      isVectorValued = true;
      unsigned lastArgN = m_Function->getNumParams() - 1;
      outputArrayStr = m_Function->getParamDecl(lastArgN)->getNameAsString();
    }

    // Check if DiffRequest asks for TBR analysis to be enabled
    if (request.EnableTBRAnalysis)
      enableTBR = true;

    // Check if DiffRequest asks for use of enzyme as backend
    if (request.use_enzyme)
      use_enzyme = true;

    auto derivativeBaseName = request.BaseFunctionName;
    std::string gradientName = derivativeBaseName + funcPostfix();
    // To be consistent with older tests, nothing is appended to 'f_grad' if
    // we differentiate w.r.t. all the parameters at once.
    if(isVectorValued){
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
    }else{
      if (args.size() != FD->getNumParams()){
        for (const auto* arg : args) {
          const auto* it = std::find(FD->param_begin(), FD->param_end(), arg);
          auto idx = std::distance(FD->param_begin(), it);
          gradientName += ('_' + std::to_string(idx));
        }
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
    if (!isVectorValued && numExtraParam == 0)
      shouldCreateOverload = true;

    const auto* originalFnType =
        dyn_cast<FunctionProtoType>(m_Function->getType());
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

    // Create the gradient function declaration.
    llvm::SaveAndRestore<DeclContext*> SaveContext(m_Sema.CurContext);
    llvm::SaveAndRestore<Scope*> SaveScope(getCurrentScope(),
                                           getEnclosingNamespaceOrTUScope());
    auto* DC = const_cast<DeclContext*>(m_Function->getDeclContext());
    m_Sema.CurContext = DC;
    DeclWithContext result = m_Builder.cloneFunction(
        m_Function, *this, DC, noLoc, name, gradientFunctionType);
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

    llvm::ArrayRef<ParmVarDecl*> paramsRef =
        clad_compat::makeArrayRef(params.data(), params.size());
    gradientFD->setParams(paramsRef);
    gradientFD->setBody(nullptr);

    if (isVectorValued) {
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
      m_ExternalSource->ActOnStartOfDerivedFnBody(request);

    Stmt* gradientBody = nullptr;

    if (!use_enzyme)
      DifferentiateWithClad();
    else
      DifferentiateWithEnzyme();

    gradientBody = endBlock();
    m_Derivative->setBody(gradientBody);
    endScope(); // Function body scope
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

  DerivativeAndOverload
  ReverseModeVisitor::DerivePullback(const clang::FunctionDecl* FD,
                                     const DiffRequest& request) {
    if (request.EnableTBRAnalysis)
      enableTBR = true;
    TBRAnalyzer analyzer(m_Context);
    if (enableTBR) {
      analyzer.Analyze(FD);
      m_ToBeRecorded = analyzer.getResult();
    }

    // FIXME: Duplication of external source here is a workaround
    // for the two 'Derive's being different functions.
    if (m_ExternalSource)
      m_ExternalSource->ActOnStartOfDerive();
    silenceDiags = !request.VerboseDiags;
    m_Function = FD;
    m_Mode = DiffMode::experimental_pullback;
    assert(m_Function && "Must not be null.");

    DiffParams args{};
    std::copy(FD->param_begin(), FD->param_end(), std::back_inserter(args));

#ifndef NDEBUG
    bool isStaticMethod = utils::IsStaticMethod(FD);
    assert((!args.empty() || !isStaticMethod) &&
           "Cannot generate pullback function of a function "
           "with no differentiable arguments");
#endif

    if (m_ExternalSource)
      m_ExternalSource->ActAfterParsingDiffArgs(request, args);

    auto derivativeName =
        utils::ComputeEffectiveFnName(m_Function) + "_pullback";
    auto DNI = utils::BuildDeclarationNameInfo(m_Sema, derivativeName);

    auto paramTypes = ComputeParamTypes(args);
    const auto* originalFnType =
        dyn_cast<FunctionProtoType>(m_Function->getType());

    if (m_ExternalSource)
      m_ExternalSource->ActAfterCreatingDerivedFnParamTypes(paramTypes);

    QualType pullbackFnType = m_Context.getFunctionType(
        m_Context.VoidTy, paramTypes, originalFnType->getExtProtoInfo());

    llvm::SaveAndRestore<DeclContext*> saveContext(m_Sema.CurContext);
    llvm::SaveAndRestore<Scope*> saveScope(getCurrentScope(),
                                           getEnclosingNamespaceOrTUScope());
    m_Sema.CurContext = const_cast<DeclContext*>(m_Function->getDeclContext());

    DeclWithContext fnBuildRes = m_Builder.cloneFunction(
        m_Function, *this, m_Sema.CurContext, noLoc, DNI, pullbackFnType);
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
    m_Derivative->setBody(nullptr);

    if (m_ExternalSource)
      m_ExternalSource->ActBeforeCreatingDerivedFnBodyScope();

    beginScope(Scope::FnScope | Scope::DeclScope);
    m_DerivativeFnScope = getCurrentScope();

    beginBlock();
    if (m_ExternalSource)
      m_ExternalSource->ActOnStartOfDerivedFnBody(request);

    StmtDiff bodyDiff = Visit(m_Function->getBody());
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
    m_Sema.PopFunctionScopeInfo();
    m_Sema.PopDeclContext();
    endScope(); // Function decl scope

    return DerivativeAndOverload{fnBuildRes.first, nullptr};
  }

  void ReverseModeVisitor::DifferentiateWithClad() {
    TBRAnalyzer analyzer(m_Context);
    if (enableTBR) {
      analyzer.Analyze(m_Function);
      m_ToBeRecorded = analyzer.getResult();
    }

    llvm::ArrayRef<ParmVarDecl*> paramsRef = m_Derivative->parameters();

    // create derived variables for parameters which are not part of
    // independent variables (args).
    for (std::size_t i = 0; i < m_Function->getNumParams(); ++i) {
      ParmVarDecl* param = paramsRef[i];
      // derived variables are already created for independent variables.
      if (m_Variables.count(param))
        continue;
      // in vector mode last non diff parameter is output parameter.
      if (isVectorValued && i == m_Function->getNumParams() - 1)
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
    StmtDiff BodyDiff = Visit(m_Function->getBody());
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
    unsigned numParams = m_Function->getNumParams();
    auto origParams = m_Function->parameters();
    llvm::ArrayRef<ParmVarDecl*> paramsRef = m_Derivative->parameters();
    const auto* originalFnType =
        dyn_cast<FunctionProtoType>(m_Function->getType());

    // Extract Pointer from Clad Array Ref
    llvm::SmallVector<VarDecl*, 8> cladRefParams;
    for (unsigned i = 0; i < numParams; i++) {
      QualType paramType = origParams[i]->getOriginalType();
      if (paramType->isRealType()) {
        cladRefParams.push_back(nullptr);
        continue;
      }

      paramType = m_Context.getPointerType(
          QualType(paramType->getPointeeOrArrayElementType(), 0));
      auto* arrayRefNameExpr = BuildDeclRef(paramsRef[numParams + i]);
      auto* getPointerExpr = BuildCallExprToMemFn(arrayRefNameExpr, "ptr", {});
      auto* arrayRefToArrayStmt = BuildVarDecl(
          paramType, "d_" + paramsRef[i]->getNameAsString(), getPointerExpr);
      addToCurrentBlock(BuildDeclStmt(arrayRefToArrayStmt), direction::forward);
      cladRefParams.push_back(arrayRefToArrayStmt);
    }
    // Prepare Arguments and Parameters to enzyme_autodiff
    llvm::SmallVector<Expr*, 16> enzymeArgs;
    llvm::SmallVector<ParmVarDecl*, 16> enzymeParams;
    llvm::SmallVector<ParmVarDecl*, 16> enzymeRealParams;
    llvm::SmallVector<ParmVarDecl*, 16> enzymeRealParamsRef;

    // First add the function itself as a parameter/argument
    enzymeArgs.push_back(BuildDeclRef(const_cast<FunctionDecl*>(m_Function)));
    auto* fdDeclContext =
        const_cast<DeclContext*>(m_Function->getDeclContext());
    enzymeParams.push_back(m_Sema.BuildParmVarDeclForTypedef(
        fdDeclContext, noLoc, m_Function->getType()));

    // Add rest of the parameters/arguments
    for (unsigned i = 0; i < numParams; i++) {
      // First Add the original parameter
      enzymeArgs.push_back(BuildDeclRef(paramsRef[i]));
      enzymeParams.push_back(m_Sema.BuildParmVarDeclForTypedef(
          fdDeclContext, noLoc, paramsRef[i]->getType()));

      // If the original parameter is not of array/pointer type, then we don't
      // have to extract its pointer from clad array_ref and add it to the
      // enzyme parameters, so we can skip the rest of the code
      if (!cladRefParams[i]) {
        // If original parameter is of a differentiable real type(but not
        // array/pointer), then add it to the list of params whose gradient must
        // be extracted later from the EnzymeGradient structure
        if (paramsRef[i]->getOriginalType()->isRealFloatingType()) {
          enzymeRealParams.push_back(paramsRef[i]);
          enzymeRealParamsRef.push_back(paramsRef[numParams + i]);
        }
        continue;
      }
      // Then add the corresponding clad array ref pointer variable
      enzymeArgs.push_back(BuildDeclRef(cladRefParams[i]));
      enzymeParams.push_back(m_Sema.BuildParmVarDeclForTypedef(
          fdDeclContext, noLoc, cladRefParams[i]->getType()));
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
        "__enzyme_autodiff_" + m_Function->getNameAsString();
    IdentifierInfo* IIEnzyme = &m_Context.Idents.get(enzymeCallName);
    DeclarationName nameEnzyme(IIEnzyme);
    QualType enzymeFunctionType =
        m_Sema.BuildFunctionType(QT, enzymeParamsType, noLoc, nameEnzyme,
                                 originalFnType->getExtProtoInfo());
    FunctionDecl* enzymeCallFD = FunctionDecl::Create(
        m_Context, fdDeclContext, noLoc, noLoc, nameEnzyme, enzymeFunctionType,
        m_Function->getTypeSourceInfo(), SC_Extern);
    enzymeCallFD->setParams(enzymeParams);
    Expr* enzymeCall = BuildCallExprToFunction(enzymeCallFD, enzymeArgs);

    // Prepare the statements that assign the gradients to
    // non array/pointer type parameters of the original function
    if (!enzymeRealParams.empty()) {
      auto* gradDeclStmt = BuildVarDecl(QT, "grad", enzymeCall, true);
      addToCurrentBlock(BuildDeclStmt(gradDeclStmt), direction::forward);

      for (unsigned i = 0; i < enzymeRealParams.size(); i++) {
        auto* LHSExpr = BuildOp(UO_Deref, BuildDeclRef(enzymeRealParamsRef[i]));

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

  static Stmt* unwrapIfSingleStmt(Stmt* S) {
    if (!S)
      return nullptr;
    if (!isa<CompoundStmt>(S))
      return S;
    auto* CS = cast<CompoundStmt>(S);
    if (CS->size() == 0)
      return nullptr;
    if (CS->size() == 1)
      return CS->body_front();
    return CS;
  }

  StmtDiff ReverseModeVisitor::VisitIfStmt(const clang::IfStmt* If) {
    // Control scope of the IfStmt. E.g., in if (double x = ...) {...}, x goes
    // to this scope.
    beginScope(Scope::DeclScope | Scope::ControlScope);

    StmtDiff cond = Clone(If->getCond());
    // Condition has to be stored as a "global" variable, to take the correct
    // branch in the reverse pass.
    // If we are inside loop, the condition has to be stored in a stack after
    // the if statement.
    Expr* PushCond = nullptr;
    Expr* PopCond = nullptr;
    auto condExpr = Visit(cond.getExpr());
    if (isInsideLoop) {
      // If we are inside for loop, cond will be stored in the following way:
      // forward:
      // _t = cond;
      // if (_t) { ... }
      // clad::push(..., _t);
      // reverse:
      // if (clad::pop(...)) { ... }
      // Simply doing
      // if (clad::push(..., _t) { ... }
      // is incorrect when if contains return statement inside: return will
      // skip corresponding push.
      cond = StoreAndRef(condExpr.getExpr(), direction::forward, "_t",
                         /*forceDeclCreation=*/true);
      StmtDiff condPushPop = GlobalStoreAndRef(cond.getExpr(), "_cond",
                                               /*force=*/true);
      PushCond = condPushPop.getExpr();
      PopCond = condPushPop.getExpr_dx();
    } else
      cond = GlobalStoreAndRef(condExpr.getExpr(), "_cond");
    // Convert cond to boolean condition. We are modifying each Stmt in
    // StmtDiff.
    for (Stmt*& S : cond.getBothStmts())
      if (S)
        S = m_Sema
                .ActOnCondition(getCurrentScope(), noLoc, cast<Expr>(S),
                                Sema::ConditionKind::Boolean)
                .get()
                .second;

    // Create a block "around" if statement, e.g:
    // {
    //   ...
    //  if (...) {...}
    // }
    beginBlock(direction::forward);
    beginBlock(direction::reverse);
    const Stmt* init = If->getInit();
    StmtDiff initResult = init ? Visit(init) : StmtDiff{};
    // If there is Init, it's derivative will be output in the block before if:
    // E.g., for:
    // if (int x = 1; ...) {...}
    // result will be:
    // {
    //   int _d_x = 0;
    //   if (int x = 1; ...) {...}
    // }
    // This is done to avoid variable names clashes.
    addToCurrentBlock(initResult.getStmt_dx());

    VarDecl* condVarClone = nullptr;
    if (const VarDecl* condVarDecl = If->getConditionVariable()) {
      VarDeclDiff condVarDeclDiff = DifferentiateVarDecl(condVarDecl);
      condVarClone = condVarDeclDiff.getDecl();
      if (condVarDeclDiff.getDecl_dx())
        addToBlock(BuildDeclStmt(condVarDeclDiff.getDecl_dx()), m_Globals);
    }

    // Condition is just cloned as it is, not derived.
    // FIXME: if condition changes one of the variables, it may be reasonable
    // to derive it, e.g.
    // if (x += x) {...}
    // should result in:
    // {
    //   _d_y += _d_x
    //   if (y += x) {...}
    // }

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

      Stmt* Forward = unwrapIfSingleStmt(endBlock(direction::forward));
      Stmt* Reverse = unwrapIfSingleStmt(BranchDiff.getStmt_dx());
      return StmtDiff(Forward, Reverse);
    };

    StmtDiff thenDiff = VisitBranch(If->getThen());
    StmtDiff elseDiff = VisitBranch(If->getElse());

    // It is problematic to specify both condVarDecl and cond thorugh
    // Sema::ActOnIfStmt, therefore we directly use the IfStmt constructor.
    Stmt* Forward = clad_compat::IfStmt_Create(m_Context,
                                               noLoc,
                                               If->isConstexpr(),
                                               initResult.getStmt(),
                                               condVarClone,
                                               cond.getExpr(),
                                               noLoc,
                                               noLoc,
                                               thenDiff.getStmt(),
                                               noLoc,
                                               elseDiff.getStmt());
    addToCurrentBlock(Forward, direction::forward);

    Expr* reverseCond = cond.getExpr_dx();
    if (isInsideLoop) {
      addToCurrentBlock(PushCond, direction::forward);
      reverseCond = PopCond;
    }
    Stmt* Reverse = clad_compat::IfStmt_Create(m_Context,
                                               noLoc,
                                               If->isConstexpr(),
                                               initResult.getStmt_dx(),
                                               condVarClone,
                                               reverseCond,
                                               noLoc,
                                               noLoc,
                                               thenDiff.getStmt_dx(),
                                               noLoc,
                                               elseDiff.getStmt_dx());
    addToCurrentBlock(Reverse, direction::reverse);
    CompoundStmt* ForwardBlock = endBlock(direction::forward);
    CompoundStmt* ReverseBlock = endBlock(direction::reverse);
    endScope();
    return StmtDiff(unwrapIfSingleStmt(ForwardBlock),
                    unwrapIfSingleStmt(ReverseBlock));
  }

  StmtDiff ReverseModeVisitor::VisitConditionalOperator(
      const clang::ConditionalOperator* CO) {
    StmtDiff cond = Clone(CO->getCond());
    // Condition has to be stored as a "global" variable, to take the correct
    // branch in the reverse pass.
    cond = GlobalStoreAndRef(Visit(cond.getExpr()).getExpr(), "_cond");
    // Convert cond to boolean condition. We are modifying each Stmt in
    // StmtDiff.
    for (Stmt*& S : cond.getBothStmts())
      S = m_Sema
              .ActOnCondition(getCurrentScope(), noLoc, cast<Expr>(S),
                              Sema::ConditionKind::Boolean)
              .get()
              .second;

    auto* ifTrue = CO->getTrueExpr();
    auto* ifFalse = CO->getFalseExpr();

    auto VisitBranch = [&](const Expr* Branch,
                           Expr* dfdx) -> std::pair<StmtDiff, StmtDiff> {
      auto Result = DifferentiateSingleExpr(Branch, dfdx);
      StmtDiff BranchDiff = Result.first;
      StmtDiff ExprDiff = Result.second;
      Stmt* Forward = unwrapIfSingleStmt(BranchDiff.getStmt());
      Stmt* Reverse = unwrapIfSingleStmt(BranchDiff.getStmt_dx());
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
        BuildIf(cond.getExpr(), ifTrueDiff.getStmt(), ifFalseDiff.getStmt());
    Stmt* Reverse = BuildIf(cond.getExpr_dx(),
                            ifTrueDiff.getStmt_dx(),
                            ifFalseDiff.getStmt_dx());
    if (Forward)
      addToCurrentBlock(Forward, direction::forward);
    if (Reverse)
      addToCurrentBlock(Reverse, direction::reverse);

    Expr* condExpr = m_Sema
                         .ActOnConditionalOp(noLoc,
                                             noLoc,
                                             cond.getExpr(),
                                             ifTrueExprDiff.getExpr(),
                                             ifFalseExprDiff.getExpr())
                         .get();
    // If result is a glvalue, we should keep it as it can potentially be
    // assigned as in (c ? a : b) = x;
    if ((CO->isModifiableLvalue(m_Context) == Expr::MLV_Valid) &&
        ifTrueExprDiff.getExpr_dx() && ifFalseExprDiff.getExpr_dx()) {
      Expr* ResultRef = m_Sema
                            .ActOnConditionalOp(noLoc,
                                                noLoc,
                                                cond.getExpr_dx(),
                                                ifTrueExprDiff.getExpr_dx(),
                                                ifFalseExprDiff.getExpr_dx())
                            .get();
      if (ResultRef->isModifiableLvalue(m_Context) != Expr::MLV_Valid)
        ResultRef = nullptr;
      return StmtDiff(condExpr, ResultRef);
    }
    return StmtDiff(condExpr);
  }

  StmtDiff ReverseModeVisitor::VisitForStmt(const ForStmt* FS) {
    beginScope(Scope::DeclScope | Scope::ControlScope | Scope::BreakScope |
               Scope::ContinueScope);

    LoopCounter loopCounter(*this);
    if (loopCounter.getPush())
      addToCurrentBlock(loopCounter.getPush());
    beginBlock(direction::forward);
    beginBlock(direction::reverse);
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

    // FIXME: for now we assume that cond has no differentiable effects,
    // but it is not generally true, e.g. for (...; (x = y); ...)...
    StmtDiff cond;
    if (FS->getCond())
      cond = Visit(FS->getCond());
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
    Expr* forwardCond = cond.getExpr();
    /// If there is a declaration in the condition, `cond` will be
    /// a DeclRefExpr of the declared variable. There is no point in
    /// inserting it since condVarRes.getExpr() represents an assignment with
    /// that variable on the LHS.
    /// e.g. for condition `int x = y`,
    /// condVarRes.getExpr() will represent `x = y`
    if (condVarRes.getExpr() != nullptr && isa<Expr>(condVarRes.getExpr()))
      forwardCond = cast<Expr>(condVarRes.getExpr());

    Stmt* Forward = new (m_Context)
        ForStmt(m_Context, initResult.getStmt(), forwardCond, condVarClone,
                incResult, BodyDiff.getStmt(), noLoc, noLoc, noLoc);

    // Create a condition testing counter for being zero, and its decrement.
    // To match the number of iterations in the forward pass, the reverse loop
    // will look like: for(; Counter; Counter--) ...
    Expr*
        CounterCondition = loopCounter.getCounterConditionResult().get().second;
    Expr* CounterDecrement = loopCounter.getCounterDecrement();

    Stmt* ReverseResult = BodyDiff.getStmt_dx();
    if (!ReverseResult)
      ReverseResult = new (m_Context) NullStmt(noLoc);
    Stmt* Reverse = new (m_Context) ForStmt(m_Context,
                                            nullptr,
                                            CounterCondition,
                                            nullptr,
                                            CounterDecrement,
                                            ReverseResult,
                                            noLoc,
                                            noLoc,
                                            noLoc);
    addToCurrentBlock(Forward, direction::forward);
    Forward = endBlock(direction::forward);
    addToCurrentBlock(loopCounter.getPop(), direction::reverse);
    addToCurrentBlock(initResult.getStmt_dx(), direction::reverse);
    addToCurrentBlock(Reverse, direction::reverse);
    Reverse = endBlock(direction::reverse);
    endScope();

    return {unwrapIfSingleStmt(Forward), unwrapIfSingleStmt(Reverse)};
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
    if (isa<FloatingLiteral>(dfdf) || isa<IntegerLiteral>(dfdf)) {
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
    // If the original function returns at this point, some part of the reverse
    // pass (corresponding to other branches that do not return here) must be
    // skipped. We create a label in the reverse pass and jump to it via goto.
    LabelDecl* LD = LabelDecl::Create(
        m_Context, m_Sema.CurContext, noLoc, CreateUniqueIdentifier("_label"));
    m_Sema.PushOnScopeChains(LD, m_DerivativeFnScope, true);
    // Attach label to the last Stmt in the corresponding Reverse Stmt.
    if (!Reverse)
      Reverse = m_Sema.ActOnNullStmt(noLoc).get();
    Stmt* LS = m_Sema.ActOnLabelStmt(noLoc, LD, noLoc, Reverse).get();
    addToCurrentBlock(LS, direction::reverse);
    for (Stmt* S : cast<CompoundStmt>(ReturnDiff.getStmt())->body())
      addToCurrentBlock(S, direction::forward);

    // FIXME: When the return type of a function is a class, ExprDiff.getExpr()
    // returns nullptr, which is a bug. For the time being, the only use case of
    // a return type being class is in pushforwards. Hence a special case has
    // been made to to not do the StoreAndRef operation when return type is
    // ValueAndPushforward.
    if (!isCladValueAndPushforwardType(type)) {
      if (m_ExternalSource)
        m_ExternalSource->ActBeforeFinalizingVisitReturnStmt(ExprDiff);
    }

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
    if (!isCladValueAndPushforwardType(ILEType) && ILEType->isRecordType() &&
        ILEType->getAsCXXRecordDecl()->isStruct()) {
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
    bool isValueAndPushforward = isCladValueAndPushforwardType(ILEType);
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
      auto* add_assign = BuildOp(BO_AddAssign, result, dfdx());
      // Add it to the body statements.
      addToCurrentBlock(add_assign, direction::reverse);
    }
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
      if (VD->getDeclContext() != m_Sema.CurContext)
        clonedDRE = cast<DeclRefExpr>(BuildDeclRef(VD));
      // This case happens when ref-type variables have to become function
      // global. Ref-type declarations cannot be moved to the function global
      // scope because they can't be separated from their inits.
      if (DRE->getDecl()->getType()->isReferenceType() &&
          clonedDRE->getType()->isPointerType())
        clonedDRE = BuildOp(UO_Deref, clonedDRE);
      if (isVectorValued) {
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
            auto* add_assign = BuildOp(BO_AddAssign, it->second, dfdx());
            // Add it to the body statements.
            addToCurrentBlock(add_assign, direction::reverse);
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

    // Stores the call arguments for the function to be derived
    llvm::SmallVector<Expr*, 16> CallArgs{};
    // Stores the dx of the call arguments for the function to be derived
    llvm::SmallVector<Expr*, 16> CallArgDx{};
    // Stores the call arguments for the derived function
    llvm::SmallVector<Expr*, 16> DerivedCallArgs{};
    // Stores tape decl and pushes for multiarg numerically differentiated
    // calls.
    llvm::SmallVector<Stmt*, 16> NumericalDiffMultiArg{};

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
              .ActOnCallExpr(getCurrentScope(), Clone(CE->getCallee()), noLoc,
                             llvm::MutableArrayRef<Expr*>(CallArgs), noLoc)
              .get();
      Expr* call_dx =
          m_Sema
              .ActOnCallExpr(getCurrentScope(), Clone(CE->getCallee()), noLoc,
                             llvm::MutableArrayRef<Expr*>(DerivedCallArgs),
                             noLoc)
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
        DerivedCallArgs.push_back(ArgDiff.getExpr_dx());
      }
      Expr* call =
          m_Sema
              .ActOnCallExpr(getCurrentScope(), Clone(CE->getCallee()), noLoc,
                             llvm::MutableArrayRef<Expr*>(CallArgs), noLoc)
              .get();
      Expr* call_dx =
          m_Sema
              .ActOnCallExpr(getCurrentScope(), Clone(CE->getCallee()), noLoc,
                             llvm::MutableArrayRef<Expr*>(DerivedCallArgs),
                             noLoc)
              .get();
      m_DeallocExprs.push_back(call);
      m_DeallocExprs.push_back(call_dx);
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
      Expr* call = m_Sema
                       .ActOnCallExpr(getCurrentScope(),
                                      Clone(CE->getCallee()),
                                      noLoc,
                                      llvm::MutableArrayRef<Expr*>(CallArgs),
                                      noLoc)
                       .get();
      return call;
    }

    llvm::SmallVector<VarDecl*, 16> ArgResultDecls{};
    llvm::SmallVector<DeclStmt*, 16> ArgDeclStmts{};
    // Save current index in the current block, to potentially put some
    // statements there later.
    std::size_t insertionPoint = getCurrentBlock(direction::reverse).size();

    // FIXME: We should add instructions for handling non-differentiable
    // arguments. Currently we are implicitly assuming function call only
    // contains differentiable arguments.
    bool isCXXOperatorCall = isa<CXXOperatorCallExpr>(CE);

    for (std::size_t i = static_cast<std::size_t>(isCXXOperatorCall),
                     e = CE->getNumArgs();
         i != e; ++i) {
      const Expr* arg = CE->getArg(i);
      const auto* PVD =
          FD->getParamDecl(i - static_cast<unsigned long>(isCXXOperatorCall));
      StmtDiff argDiff{};
      bool passByRef = utils::IsReferenceOrPointerType(PVD->getType());
      if (passByRef && isa<MaterializeTemporaryExpr>(arg)) {
        // If the argument is a temporary variable, this means that param type
        // is a reference to a const type and we are passing a temporary
        // variable to it. In this case, we should not pass the derivative
        // argument by reference.
        passByRef = false;
      }
      QualType argDiffType;
      // We do not need to create result arg for arguments passed by reference
      // because the derivatives of arguments passed by reference are directly
      // modified by the derived callee function.
      if (passByRef) {
        argDiff = Visit(arg);
        Expr* dArg = nullptr;
        argDiffType = argDiff.getExpr()->getType();
        QualType argResultValueType =
            utils::GetValueType(argDiffType).getNonReferenceType();
        // Create ArgResult variable for each reference argument because it is
        // required by error estimator. For automatic differentiation, we do not need
        // to create ArgResult variable for arguments passed by reference.
        // ```
        // _r0 = _d_a;
        // ```
        if (argDiff.getExpr_dx() && utils::IsLiteral(argDiff.getExpr_dx())) {
          dArg = StoreAndRef(argDiff.getExpr_dx(), arg->getType(),
                             direction::reverse, "_r",
                             /*forceDeclCreation=*/true);
        } else if (argDiffType->isArrayType()) {
          Expr* init = argDiff.getExpr_dx();
          if (isa<ConstantArrayType>(argDiff.getExpr_dx()->getType()))
            init = utils::BuildCladArrayInitByConstArray(m_Sema,
                                                         argDiff.getExpr_dx());

          dArg = StoreAndRef(init, GetCladArrayOfType(argResultValueType),
                             direction::reverse, "_r",
                             /*forceDeclCreation=*/true,
                             VarDecl::InitializationStyle::CallInit);
        } else if (argDiffType->isPointerType()) {
          dArg = StoreAndRef(argDiff.getExpr_dx(), argDiffType,
                             direction::reverse, "_r",
                             /*forceDeclCreation=*/true);
        } else {
          dArg = StoreAndRef(argDiff.getExpr_dx(), argResultValueType,
                             direction::reverse, "_r",
                             /*forceDeclCreation=*/true);
        }
        ArgResultDecls.push_back(
            cast<VarDecl>(cast<DeclRefExpr>(dArg)->getDecl()));
      } else {
        assert(!utils::isArrayOrPointerType(arg->getType()) &&
               "Arguments passed by pointers should be covered in pass by "
               "reference calls");
        // Create temporary variables corresponding to derivative of each
        // argument, so that they can be referred to when arguments is visited.
        // Variables will be initialized later after arguments is visited. This
        // is done to reduce cloning complexity and only clone once. The type is
        // same as the call expression as it is the type used to declare the
        // _gradX array
        Expr* dArg = nullptr;
        QualType argType = utils::GetValueType(arg->getType());
        dArg = StoreAndRef(/*E=*/nullptr, argType, direction::reverse, "_r",
                           /*forceDeclCreation=*/true);
        ArgResultDecls.push_back(
            cast<VarDecl>(cast<DeclRefExpr>(dArg)->getDecl()));
        // Visit using uninitialized reference.
        argDiff = Visit(arg, dArg);
        argDiffType = argDiff.getExpr()->getType();
      }

      // FIXME: We may use same argDiff.getExpr_dx at two places. This can
      // lead to inconsistent pushes and pops. If `isInsideLoop` is true and
      // actual argument is something like "a[i]", then argDiff.getExpr() and
      // argDiff.getExpr_dx() will respectively be:
      // ```
      // a[clad::push(_t0, i)];
      // a[clad::pop(_t0)];
      // ```
      // The expression `a[clad::pop(_t0)]` might already be used in the AST if
      // visit was called with a dfdx() present.
      // And thus using this expression in the AST explicitly may lead to size
      // assertion failed.
      //
      // We should modify the design so that the behaviour of obtained StmtDiff
      // expression is consistent both inside and outside loops.
      CallArgDx.push_back(argDiff.getExpr_dx());
      // Save cloned arg in a "global" variable, so that it is accessible from
      // the reverse pass.
      // FIXME: At this point, we assume all the variables passed by reference
      // may be changed since we have no way to determine otherwise.
      // FIXME: We cannot use GlobalStoreAndRef to store a whole array so now
      // arrays are not stored.
      StmtDiff argDiffStore;
      if (passByRef && !argDiffType->isArrayType() &&
          !argDiff.getExpr()->isEvaluatable(m_Context))
        argDiffStore =
            GlobalStoreAndRef(argDiff.getExpr(), "_t", /*force=*/true);
      else
        argDiffStore = {argDiff.getExpr(), argDiff.getExpr()};

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
      if (passByRef && !argDiffType->isArrayType()) {
        if (isInsideLoop) {
          // Add tape push expression. We need to explicitly add it here because
          // we cannot add it as call expression argument -- we need to pass the
          // actual argument there.
          addToCurrentBlock(argDiffStore.getExpr());
          // For reference arguments, we cannot pass `clad::pop(_t0)` to the
          // derived function. Because it will throw "lvalue reference cannot
          // bind to rvalue error". Thus we are proceeding as follows:
          // ```
          // double _r0 = clad::pop(_t0);
          // derivedCalleeFunction(_r0, ...)
          // ```
          VarDecl* argDiffLocalVD = BuildVarDecl(
              argDiffStore.getExpr_dx()->getType(),
              CreateUniqueIdentifier("_r"), argDiffStore.getExpr_dx(),
              /*DirectInit=*/false, /*TSI=*/nullptr,
              VarDecl::InitializationStyle::CInit);
          auto& block = getCurrentBlock(direction::reverse);
          block.insert(block.begin() + insertionPoint,
                       BuildDeclStmt(argDiffLocalVD));
          // Restore agrs
          auto* op = BuildOp(BinaryOperatorKind::BO_Assign, argDiff.getExpr(),
                             BuildDeclRef(argDiffLocalVD));
          block.insert(block.begin() + insertionPoint + 1, op);

          Expr* argDiffLocalE = BuildDeclRef(argDiffLocalVD);

          // We added local variable to store result of `clad::pop(...)` and
          // restoration of the original arg. Thus we need to correspondingly
          // adjust the insertion point.
          insertionPoint += 2;
          // We cannot use the already existing `argDiff.getExpr()` here because
          // it will cause inconsistent pushes and pops to the clad tape.
          // FIXME: Modify `GlobalStoreAndRef` such that its functioning is
          // consistent with `StoreAndRef`. This way we will not need to handle
          // inside loop and outside loop cases separately.
          Expr* newArgE = Visit(arg).getExpr();
          argDiffStore = {newArgE, argDiffLocalE};
        } else if (isa<DeclRefExpr>(argDiff.getExpr())) {
          // Restore args
          auto& block = getCurrentBlock(direction::reverse);
          auto* op = BuildOp(BinaryOperatorKind::BO_Assign, argDiff.getExpr(),
                             argDiffStore.getExpr());
          block.insert(block.begin() + insertionPoint, op);
          // We added restoration of the original arg. Thus we need to
          // correspondingly adjust the insertion point.
          insertionPoint += 1;

          argDiffStore = {argDiff.getExpr(), argDiffStore.getExpr_dx()};
        }
      }
      CallArgs.push_back(argDiffStore.getExpr());
      DerivedCallArgs.push_back(argDiffStore.getExpr_dx());
    }

    VarDecl* gradVarDecl = nullptr;
    Expr* gradVarExpr = nullptr;
    Expr* gradArgExpr = nullptr;
    IdentifierInfo* gradVarII = nullptr;
    Expr* OverloadedDerivedFn = nullptr;
    // If the function has a single arg and does not returns a reference or take
    // arg by reference, we look for a derivative w.r.t. to this arg using the
    // forward mode(it is unlikely that we need gradient of a one-dimensional'
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

    // Stores a list of arg result variable declaration (_r0) with the
    // corresponding grad variable expression (_grad0).
    llvm::SmallVector<std::pair<VarDecl*, Expr*>, 4> argResultsAndGrads;

    // Stores differentiation result of implicit `this` object, if any.
    StmtDiff baseDiff;
    // If it has more args or f_darg0 was not found, we look for its pullback
    // function.
    if (!OverloadedDerivedFn) {
      size_t idx = 0;

      /// Add base derivative expression in the derived call output args list if
      /// `CE` is a call to an instance member function.
      if (const auto* MD = dyn_cast<CXXMethodDecl>(FD)) {
        if (MD->isInstance()) {
          const Expr* baseOriginalE = nullptr;
          if (const auto* MCE = dyn_cast<CXXMemberCallExpr>(CE))
            baseOriginalE = MCE->getImplicitObjectArgument();
          else if (const auto* OCE = dyn_cast<CXXOperatorCallExpr>(CE))
            baseOriginalE = OCE->getArg(0);

          baseDiff = Visit(baseOriginalE);
          StmtDiff baseDiffStore = GlobalStoreAndRef(baseDiff.getExpr());
          if (isInsideLoop) {
            addToCurrentBlock(baseDiffStore.getExpr());
            VarDecl* baseLocalVD = BuildVarDecl(
                baseDiffStore.getExpr_dx()->getType(),
                CreateUniqueIdentifier("_r"), baseDiffStore.getExpr_dx(),
                /*DirectInit=*/false, /*TSI=*/nullptr,
                VarDecl::InitializationStyle::CInit);
            auto& block = getCurrentBlock(direction::reverse);
            block.insert(block.begin() + insertionPoint,
                         BuildDeclStmt(baseLocalVD));
            insertionPoint += 1;
            Expr* baseLocalE = BuildDeclRef(baseLocalVD);
            baseDiffStore = {baseDiffStore.getExpr(), baseLocalE};
          }
          baseDiff = {baseDiffStore.getExpr_dx(), baseDiff.getExpr_dx()};
          Expr* baseDerivative = baseDiff.getExpr_dx();
          if (!baseDerivative->getType()->isPointerType())
            baseDerivative =
                BuildOp(UnaryOperatorKind::UO_AddrOf, baseDerivative);
          DerivedCallOutputArgs.push_back(baseDerivative);
        }
      }

      for (auto* argDerivative : CallArgDx) {
        gradVarDecl = nullptr;
        gradVarExpr = nullptr;
        gradArgExpr = nullptr;
        gradVarII = CreateUniqueIdentifier(funcPostfix());

        const auto* PVD = FD->getParamDecl(idx);
        bool passByRef = utils::IsReferenceOrPointerType(PVD->getType());
        if (passByRef && isa<MaterializeTemporaryExpr>(CE->getArg(idx))) {
          // If the argument is a temporary variable, this means that param type
          // is a reference to a const type and we are passing a temporary
          // variable to it. In this case, we should not pass the derivative
          // argument by reference.
          passByRef = false;
        }
        if (passByRef) {
          // If derivative type is constant array type instead of
          // `clad::array_ref` or `clad::array` type, then create an
          // `clad::array_ref` variable that references this constant array. It
          // is required because the pullback function expects `clad::array_ref`
          // type for representing array derivatives. Currently, only constant
          // array data members have derivatives of constant array types.
          if ((argDerivative != nullptr) &&
              isa<ConstantArrayType>(argDerivative->getType())) {
            Expr* init =
                utils::BuildCladArrayInitByConstArray(m_Sema, argDerivative);
            auto* derivativeArrayRefVD = BuildVarDecl(
                GetCladArrayRefOfType(argDerivative->getType()
                                          ->getPointeeOrArrayElementType()
                                          ->getCanonicalTypeInternal()),
                "_t", init);
            ArgDeclStmts.push_back(BuildDeclStmt(derivativeArrayRefVD));
            argDerivative = BuildDeclRef(derivativeArrayRefVD);
          }
          if ((argDerivative != nullptr) &&
              (isCladArrayType(argDerivative->getType()) ||
               argDerivative->getType()->isPointerType() ||
               !argDerivative->isLValue()))
            gradArgExpr = argDerivative;
          else
            gradArgExpr = BuildOp(UnaryOperatorKind::UO_AddrOf, argDerivative);
        } else {
          // Declare: diffArgType _grad;
          Expr* initVal = nullptr;
          QualType gradVarType = utils::GetValueType(PVD->getType());
          if (!gradVarType->isRecordType()) {
            // If the argument is not a class type, then initialize the grad
            // variable with 0.
            initVal =
                ConstantFolder::synthesizeLiteral(gradVarType, m_Context, 0);
          }
          gradVarDecl = BuildVarDecl(gradVarType, gradVarII, initVal);
          // Pass the address of the declared variable
          gradVarExpr = BuildDeclRef(gradVarDecl);
          gradArgExpr =
              BuildOp(UO_AddrOf, gradVarExpr, m_Function->getLocation());
          argResultsAndGrads.push_back({ArgResultDecls[idx], gradVarExpr});
        }
        DerivedCallOutputArgs.push_back(gradArgExpr);
        if (gradVarDecl)
          ArgDeclStmts.push_back(BuildDeclStmt(gradVarDecl));
        idx++;
      }
      Expr* pullback = dfdx();

      if ((pullback == nullptr) && FD->getReturnType()->isLValueReferenceType())
        pullback = getZeroInit(FD->getReturnType().getNonReferenceType());

      // FIXME: Remove this restriction.
      if (!FD->getReturnType()->isVoidType()) {
        assert((pullback && !FD->getReturnType()->isVoidType()) &&
               "Call to function returning non-void type with no dfdx() is not "
               "supported!");
      }

      if (FD->getReturnType()->isVoidType()) {
        assert(pullback == nullptr && FD->getReturnType()->isVoidType() &&
               "Call to function returning void type should not have any "
               "corresponding dfdx().");
      }

      DerivedCallArgs.insert(DerivedCallArgs.end(),
                             DerivedCallOutputArgs.begin(),
                             DerivedCallOutputArgs.end());
      pullbackCallArgs = DerivedCallArgs;

      if (pullback)
        pullbackCallArgs.insert(pullbackCallArgs.begin() + CE->getNumArgs() -
                                    static_cast<int>(isCXXOperatorCall),
                                pullback);

      // Try to find it in builtin derivatives
      if (baseDiff.getExpr())
        pullbackCallArgs.insert(
            pullbackCallArgs.begin(),
            BuildOp(UnaryOperatorKind::UO_AddrOf, baseDiff.getExpr()));
      std::string customPullback =
          clad::utils::ComputeEffectiveFnName(FD) + "_pullback";
      OverloadedDerivedFn =
          m_Builder.BuildCallToCustomDerivativeOrNumericalDiff(
              customPullback, pullbackCallArgs, getCurrentScope(),
              const_cast<DeclContext*>(FD->getDeclContext()));
      if (baseDiff.getExpr())
        pullbackCallArgs.erase(pullbackCallArgs.begin());
    }

    // should be true if we are using numerical differentiation to differentiate
    // the callee function.
    bool usingNumericalDiff = false;
    // Derivative was not found, check if it is a recursive call
    if (!OverloadedDerivedFn) {
      if (FD == m_Function && m_Mode == DiffMode::experimental_pullback) {
        // Recursive call.
        auto* selfRef =
            m_Sema
                .BuildDeclarationNameExpr(
                    CXXScopeSpec(), m_Derivative->getNameInfo(), m_Derivative)
                .get();

        OverloadedDerivedFn =
            m_Sema
                .ActOnCallExpr(getCurrentScope(), selfRef, noLoc,
                               llvm::MutableArrayRef<Expr*>(DerivedCallArgs),
                               noLoc)
                .get();
      } else {
        if (m_ExternalSource)
          m_ExternalSource->ActBeforeDifferentiatingCallExpr(
              pullbackCallArgs, ArgDeclStmts, dfdx());
        // Overloaded derivative was not found, request the CladPlugin to
        // derive the called function.
        DiffRequest pullbackRequest{};
        pullbackRequest.Function = FD;
        pullbackRequest.BaseFunctionName =
            clad::utils::ComputeEffectiveFnName(FD);
        pullbackRequest.Mode = DiffMode::experimental_pullback;
        // Silence diag outputs in nested derivation process.
        pullbackRequest.VerboseDiags = false;
        pullbackRequest.EnableTBRAnalysis = enableTBR;
        FunctionDecl* pullbackFD = plugin::ProcessDiffRequest(m_CladPlugin, pullbackRequest);
        // Clad failed to derive it.
        // FIXME: Add support for reference arguments to the numerical diff. If
        // it already correctly support reference arguments then confirm the
        // support and add tests for the same.
        if (!pullbackFD && !utils::HasAnyReferenceOrPointerArgument(FD) &&
            !isa<CXXMethodDecl>(FD)) {
          // Try numerically deriving it.
          // Build a clone call expression so that we can correctly
          // scope the function to be differentiated.
          Expr* call = m_Sema
                     .ActOnCallExpr(getCurrentScope(),
                                    Clone(CE->getCallee()),
                                    noLoc,
                                    llvm::MutableArrayRef<Expr*>(CallArgs),
                                    noLoc)
                     .get();
          Expr* fnCallee = cast<CallExpr>(call)->getCallee();
          if (NArgs == 1) {
            OverloadedDerivedFn = GetSingleArgCentralDiffCall(fnCallee,
                                                              DerivedCallArgs
                                                                  [0],
                                                              /*targetPos=*/0,
                                                              /*numArgs=*/1,
                                                              DerivedCallArgs);
            asGrad = !OverloadedDerivedFn;
          } else {
            auto CEType = getNonConstType(CE->getType(), m_Context, m_Sema);
            OverloadedDerivedFn = GetMultiArgCentralDiffCall(
                fnCallee, CEType.getCanonicalType(), CE->getNumArgs(),
                NumericalDiffMultiArg, DerivedCallArgs, DerivedCallOutputArgs);
          }
          CallExprDiffDiagnostics(FD->getNameAsString(), CE->getBeginLoc(),
                                  OverloadedDerivedFn);
          if (!OverloadedDerivedFn) {
            auto& block = getCurrentBlock(direction::reverse);
            block.insert(block.begin(), ArgDeclStmts.begin(),
                         ArgDeclStmts.end());
            return StmtDiff(Clone(CE));
          }
          usingNumericalDiff = true;
        } else if (pullbackFD) {
          if (baseDiff.getExpr()) {
            Expr* baseE = baseDiff.getExpr();
            OverloadedDerivedFn = BuildCallExprToMemFn(
                baseE, pullbackFD->getName(), pullbackCallArgs, pullbackFD);
          } else {
            OverloadedDerivedFn =
                m_Sema
                    .ActOnCallExpr(getCurrentScope(), BuildDeclRef(pullbackFD),
                                   noLoc, pullbackCallArgs, noLoc)
                    .get();
          }
        }
      }
    }

    if (OverloadedDerivedFn) {
      // Derivative was found.
      FunctionDecl* fnDecl = dyn_cast<CallExpr>(OverloadedDerivedFn)
                                 ->getDirectCallee();
      if (!asGrad) {
        if (utils::IsCladValueAndPushforwardType(fnDecl->getReturnType()))
          OverloadedDerivedFn = utils::BuildMemberExpr(
              m_Sema, getCurrentScope(), OverloadedDerivedFn, "pushforward");
        // If the derivative is called through _darg0 instead of _grad.
        Expr* d = BuildOp(BO_Mul, dfdx(), OverloadedDerivedFn);

        PerformImplicitConversionAndAssign(ArgResultDecls[0], d);
      } else {
        // Put Result array declaration in the function body.
        // Call the gradient, passing Result as the last Arg.
        auto& block = getCurrentBlock(direction::reverse);
        auto* it = std::begin(block) + insertionPoint;

        // Insert the _gradX declaration statements
        it = block.insert(it, ArgDeclStmts.begin(), ArgDeclStmts.end());
        it += ArgDeclStmts.size();
        it = block.insert(it, NumericalDiffMultiArg.begin(),
                          NumericalDiffMultiArg.end());
        it += NumericalDiffMultiArg.size();
        // Insert the CallExpr to the derived function
        block.insert(it, OverloadedDerivedFn);

        if (usingNumericalDiff) {
          for (auto resAndGrad : argResultsAndGrads) {
            VarDecl* argRes = resAndGrad.first;
            Expr* grad = resAndGrad.second;
            if (isCladArrayType(grad->getType())) {
              Expr* E = BuildOp(BO_MulAssign, grad, dfdx());
              // Visit each arg with df/dargi = df/dxi * Result.
              PerformImplicitConversionAndAssign(argRes, E);
            } else {
              //  Visit each arg with df/dargi = df/dxi * Result.
              PerformImplicitConversionAndAssign(argRes,
                                                 BuildOp(BO_Mul, dfdx(), grad));
            }
          }
        } else {
          for (auto resAndGrad : argResultsAndGrads) {
            VarDecl* argRes = resAndGrad.first;
            Expr* grad = resAndGrad.second;
            PerformImplicitConversionAndAssign(argRes, grad);
          }
        }
      }
    }
    if (m_ExternalSource)
      m_ExternalSource->ActBeforeFinalizingVisitCallExpr(
        CE, OverloadedDerivedFn, DerivedCallArgs, ArgResultDecls, asGrad);

    Expr* call = nullptr;

    QualType returnType = FD->getReturnType();
    if (returnType->isReferenceType() &&
        !returnType.getNonReferenceType().isConstQualified()) {
      DiffRequest calleeFnForwPassReq;
      calleeFnForwPassReq.Function = FD;
      calleeFnForwPassReq.Mode = DiffMode::reverse_mode_forward_pass;
      calleeFnForwPassReq.BaseFunctionName =
          clad::utils::ComputeEffectiveFnName(FD);
      calleeFnForwPassReq.VerboseDiags = true;
      FunctionDecl* calleeFnForwPassFD =
          plugin::ProcessDiffRequest(m_CladPlugin, calleeFnForwPassReq);

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
            BuildOp(UnaryOperatorKind::UO_AddrOf, derivedBase, noLoc));
      }

      for (std::size_t i = static_cast<std::size_t>(isCXXOperatorCall),
                       e = CE->getNumArgs();
           i != e; ++i) {
        const Expr* arg = CE->getArg(i);
        const ParmVarDecl* PVD =
            FD->getParamDecl(i - static_cast<unsigned long>(isCXXOperatorCall));
        StmtDiff argDiff = Visit(arg);
        if ((argDiff.getExpr_dx() != nullptr) &&
            PVD->getType()->isReferenceType()) {
          Expr* derivedArg = argDiff.getExpr_dx();
          // FIXME: We may need this if-block once we support pointers, and
          // passing pointers-by-reference if
          // (isCladArrayType(derivedArg->getType()))
          //   CallArgs.push_back(derivedArg);
          // else
          CallArgs.push_back(
              BuildOp(UnaryOperatorKind::UO_AddrOf, derivedArg, noLoc));
        } else
          CallArgs.push_back(m_Sema.ActOnCXXNullPtrLiteral(noLoc).get());
      }
      if (baseDiff.getExpr()) {
        Expr* baseE = baseDiff.getExpr();
        call = BuildCallExprToMemFn(baseE, calleeFnForwPassFD->getName(),
                                    CallArgs, calleeFnForwPassFD);
      } else {
        call = m_Sema
                   .ActOnCallExpr(getCurrentScope(),
                                  BuildDeclRef(calleeFnForwPassFD), noLoc,
                                  CallArgs, noLoc)
                   .get();
      }
      auto* callRes = StoreAndRef(call);
      auto* resValue =
          utils::BuildMemberExpr(m_Sema, getCurrentScope(), callRes, "value");
      auto* resAdjoint =
          utils::BuildMemberExpr(m_Sema, getCurrentScope(), callRes, "adjoint");
      return StmtDiff(resValue, nullptr, resAdjoint);
    } // Recreate the original call expression.
    call = m_Sema
               .ActOnCallExpr(getCurrentScope(), Clone(CE->getCallee()), noLoc,
                              CallArgs, noLoc)
               .get();
    return StmtDiff(call);

    return {};
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
      if (isPointerOp && isCladArrayType(diff_dx->getType()))
        diff_dx = BuildArrayRefPtrRefExpr(diff_dx);
      if (isPointerOp)
        addToCurrentBlock(BuildOp(opCode, diff_dx), direction::forward);
      if (UsefulToStoreGlobal(diff.getRevSweepAsExpr())) {
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
      if (isPointerOp && isCladArrayType(diff_dx->getType()))
        diff_dx = BuildArrayRefPtrRefExpr(diff_dx);
      if (isPointerOp)
        addToCurrentBlock(BuildOp(opCode, diff_dx), direction::forward);
      if (UsefulToStoreGlobal(diff.getRevSweepAsExpr())) {
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
      Expr* diff_dx = diff.getExpr_dx();
      bool specialDThisCase = false;
      Expr* derivedE = nullptr;
      if (const auto* MD = dyn_cast<CXXMethodDecl>(m_Function)) {
        if (MD->isInstance() && !diff_dx->getType()->isPointerType())
          specialDThisCase = true; // _d_this is already dereferenced.
      }
      if (specialDThisCase)
        derivedE = diff_dx;
      else {
        derivedE = BuildOp(UnaryOperatorKind::UO_Deref, diff_dx);
        // Create the (target += dfdx) statement.
        if (dfdx()) {
          auto* add_assign = BuildOp(BO_AddAssign, derivedE, dfdx());
          // Add it to the body statements.
          addToCurrentBlock(add_assign, direction::reverse);
        }
      }
      return {cloneE, derivedE};
    } else {
      if (opCode != UO_LNot)
        // We should only output warnings on visiting boolean conditions
        // when it is related to some indepdendent variable and causes
        // discontinuity in the function space.
        // FIXME: We should support boolean differentiation or ignore it
        // completely
        unsupportedOpWarn(UnOp->getEndLoc());

      if (isa<DeclRefExpr>(E))
        diff = Visit(E);
      else
        diff = StmtDiff(E);
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
      std::unique_ptr<DelayedStoreResult> RDelayed;
      StmtDiff RResult;
      // If R has no side effects, it can be just cloned
      // (no need to store it).
      if (!ShouldRecompute(R)) {
        RDelayed = std::unique_ptr<DelayedStoreResult>(
            new DelayedStoreResult(DelayedGlobalStoreAndRef(R)));
        RResult = RDelayed->Result;
      } else {
        RResult = StmtDiff(Clone(R));
      }

      Expr* dl = nullptr;
      if (dfdx())
        dl = BuildOp(BO_Mul, dfdx(), RResult.getRevSweepAsExpr());
      Ldiff = Visit(L, dl);
      // dxi/xr = xl
      // df/dxr += df/dxi * dxi/xr = df/dxi * xl
      // Store left multiplier and assign it with L.
      StmtDiff LStored = Ldiff;
      if (!ShouldRecompute(LStored.getExpr()))
        LStored = GlobalStoreAndRef(LStored.getExpr(), /*prefix=*/"_t",
                                    /*force=*/true);
      Expr::EvalResult dummy;
      if (RDelayed ||
          !clad_compat::Expr_EvaluateAsConstantExpr(R, dummy, m_Context)) {
        Expr* dr = nullptr;
        if (dfdx())
          dr = BuildOp(BO_Mul, LStored.getRevSweepAsExpr(), dfdx());
        Rdiff = Visit(R, dr);
        // Assign right multiplier's variable with R.
        if (RDelayed)
          RDelayed->Finalize(Rdiff.getExpr());
      }
      std::tie(Ldiff, Rdiff) =
          std::make_pair(LStored.getExpr(), RResult.getExpr());
    } else if (opCode == BO_Div) {
      // xi = xl / xr
      // dxi/xl = 1 / xr
      // df/dxl += df/dxi * dxi/xl = df/dxi * (1/xr)
      auto RDelayed = DelayedGlobalStoreAndRef(R);
      StmtDiff RResult = RDelayed.Result;
      Expr* RStored =
          StoreAndRef(RResult.getRevSweepAsExpr(), direction::reverse);
      Expr* dl = nullptr;
      if (dfdx())
        dl = BuildOp(BO_Div, dfdx(), RStored);
      Ldiff = Visit(L, dl);
      StmtDiff LStored = Ldiff;
      if (!ShouldRecompute(LStored.getExpr()))
        LStored = GlobalStoreAndRef(LStored.getExpr(), /*prefix=*/"_t",
                                    /*force=*/true);
      // dxi/xr = -xl / (xr * xr)
      // df/dxl += df/dxi * dxi/xr = df/dxi * (-xl /(xr * xr))
      // Wrap R * R in parentheses: (R * R). otherwise code like 1 / R * R is
      // produced instead of 1 / (R * R).
      if (!RDelayed.isConstant) {
        Expr* dr = nullptr;
        if (dfdx()) {
          Expr* RxR = BuildParens(BuildOp(BO_Mul, RStored, RStored));
          dr = BuildOp(
              BO_Mul, dfdx(),
              BuildOp(UO_Minus,
                      BuildOp(BO_Div, LStored.getRevSweepAsExpr(), RxR)));
          dr = StoreAndRef(dr, direction::reverse);
        }
        Rdiff = Visit(R, dr);
        RDelayed.Finalize(Rdiff.getExpr());
      }
      std::tie(Ldiff, Rdiff) =
          std::make_pair(LStored.getExpr(), RResult.getExpr());
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
          auto isIdxValid =
              clad_compat::Expr_EvaluateAsInt(ASE->getIdx(), intIdx, m_Context);

          if (DRE_str == outputArrayStr && isIdxValid) {
            if (isVectorValued) {
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
              m_VectorOutput.push_back(temp_m_Variables);
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
      auto* Lblock = endBlock(direction::reverse);
      llvm::SmallVector<Expr*, 4> ExprsToStore;
      utils::GetInnermostReturnExpr(Ldiff.getExpr(), ExprsToStore);

      // We need to store values of derivative pointer variables in forward pass
      // and restore them in reverse pass.
      if (isPointerOp) {
        Expr* Edx = Ldiff.getExpr_dx();
        ExprsToStore.push_back(Edx);
      }

      if (L->HasSideEffects(m_Context)) {
        Expr* E = Ldiff.getExpr();
        auto* storeE =
            StoreAndRef(E, m_Context.getLValueReferenceType(E->getType()));
        Ldiff.updateStmt(storeE);
      }

      Expr* LCloned = Ldiff.getExpr();
      // For x, AssignedDiff is _d_x, for x[i] its _d_x[i], for reference exprs
      // like (x = y) it propagates recursively, so _d_x is also returned.
      Expr* AssignedDiff = Ldiff.getExpr_dx();
      if (!AssignedDiff) {
        // If either LHS or RHS is a declaration reference, visit it to avoid
        // naming collision
        auto* LDRE = dyn_cast<DeclRefExpr>(L);
        auto* RDRE = dyn_cast<DeclRefExpr>(R);

        if (!LDRE && !RDRE)
          return Clone(BinOp);

        Expr* LExpr = LDRE ? Visit(L).getExpr() : L;
        Expr* RExpr = RDRE ? Visit(R).getExpr() : R;

        return BuildOp(opCode, LExpr, RExpr);
      }
      ResultRef = AssignedDiff;
      // If assigned expr is dependent, first update its derivative;
      auto Lblock_begin = Lblock->body_rbegin();
      auto Lblock_end = Lblock->body_rend();

      if (dfdx() && Lblock_begin != Lblock_end) {
        addToCurrentBlock(*Lblock_begin, direction::reverse);
        Lblock_begin = std::next(Lblock_begin);
      }

      for (auto& E : ExprsToStore) {
        auto pushPop = StoreAndRestore(E);
        addToCurrentBlock(pushPop.getExpr(), direction::forward);
        addToCurrentBlock(pushPop.getExpr_dx(), direction::reverse);
      }

      if (m_ExternalSource)
        m_ExternalSource->ActAfterCloningLHSOfAssignOp(LCloned, R, opCode);

      // Save old value for the derivative of LHS, to avoid problems with cases
      // like x = x.
      clang::Expr* oldValue = nullptr;

      // For pointer types, no need to store old derivatives.
      if (!isPointerOp)
        oldValue = StoreAndRef(AssignedDiff, direction::reverse, "_r_d",
                               /*forceDeclCreation=*/true);
      if (opCode == BO_Assign) {
        // Add the statement `dl -= oldValue;`
        addToCurrentBlock(BuildOp(BO_SubAssign, AssignedDiff, oldValue),
                          direction::reverse);
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
        // Add the statement `dl -= oldValue;`
        addToCurrentBlock(BuildOp(BO_SubAssign, AssignedDiff, oldValue),
                          direction::reverse);
        /// Capture all the emitted statements while visiting R
        /// and insert them after `dl += dl * R`
        beginBlock(direction::reverse);
        Expr* dr = BuildOp(BO_Mul, LCloned, oldValue);
        Rdiff = Visit(R, dr);
        Stmts RBlock = EndBlockWithoutCreatingCS(direction::reverse);
        addToCurrentBlock(
            BuildOp(BO_AddAssign, AssignedDiff,
                    BuildOp(BO_Mul, oldValue, Rdiff.getRevSweepAsExpr())),
            direction::reverse);
        for (auto& S : RBlock)
          addToCurrentBlock(S, direction::reverse);
        valueForRevPass = BuildOp(BO_Mul, Rdiff.getRevSweepAsExpr(),
                                  Ldiff.getRevSweepAsExpr());
        std::tie(Ldiff, Rdiff) = std::make_pair(LCloned, Rdiff.getExpr());
      } else if (opCode == BO_DivAssign) {
        // Add the statement `dl -= oldValue;`
        addToCurrentBlock(BuildOp(BO_SubAssign, AssignedDiff, oldValue),
                          direction::reverse);
        auto RDelayed = DelayedGlobalStoreAndRef(R);
        StmtDiff RResult = RDelayed.Result;
        Expr* RStored =
            StoreAndRef(RResult.getRevSweepAsExpr(), direction::reverse);
        addToCurrentBlock(BuildOp(BO_AddAssign, AssignedDiff,
                                  BuildOp(BO_Div, oldValue, RStored)),
                          direction::reverse);
        if (!RDelayed.isConstant) {
          if (isInsideLoop)
            addToCurrentBlock(LCloned, direction::forward);
          Expr* RxR = BuildParens(BuildOp(BO_Mul, RStored, RStored));
          Expr* dr = BuildOp(BO_Mul, oldValue,
                             BuildOp(UO_Minus, BuildOp(BO_Div, LCloned, RxR)));
          dr = StoreAndRef(dr, direction::reverse);
          Rdiff = Visit(R, dr);
          RDelayed.Finalize(Rdiff.getExpr());
        }
        valueForRevPass = BuildOp(BO_Div, Rdiff.getRevSweepAsExpr(),
                                  Ldiff.getRevSweepAsExpr());
        std::tie(Ldiff, Rdiff) = std::make_pair(LCloned, RResult.getExpr());
      } else
        llvm_unreachable("unknown assignment opCode");
      if (m_ExternalSource)
        m_ExternalSource->ActBeforeFinalizingAssignOp(LCloned, ResultRef, R,
                                                      opCode);

      // Output statements from Visit(L).
      for (auto it = Lblock_begin; it != Lblock_end; ++it)
        addToCurrentBlock(*it, direction::reverse);
    } else if (opCode == BO_Comma) {
      auto* zero =
          ConstantFolder::synthesizeLiteral(m_Context.IntTy, m_Context, 0);
      Ldiff = Visit(L, zero);
      Rdiff = Visit(R, dfdx());
      valueForRevPass = Ldiff.getRevSweepAsExpr();
      ResultRef = Ldiff.getExpr();
    } else {
      // We should not output any warning on visiting boolean conditions
      // FIXME: We should support boolean differentiation or ignore it
      // completely
      if (!BinOp->isComparisonOp() && !BinOp->isLogicalOp())
        unsupportedOpWarn(BinOp->getEndLoc());

      // If either LHS or RHS is a declaration reference, visit it to avoid
      // naming collision
      auto* LDRE = dyn_cast<DeclRefExpr>(L);
      auto* RDRE = dyn_cast<DeclRefExpr>(R);

      if (!LDRE && !RDRE)
        return Clone(BinOp);

      Expr* LExpr = LDRE ? Visit(L).getExpr() : L;
      Expr* RExpr = RDRE ? Visit(R).getExpr() : R;

      return BuildOp(opCode, LExpr, RExpr);
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

  VarDeclDiff ReverseModeVisitor::DifferentiateVarDecl(const VarDecl* VD) {
    StmtDiff initDiff;
    Expr* VDDerivedInit = nullptr;
    // Local declarations are promoted to the function global scope. This
    // procedure is done to make declarations visible in the reverse sweep.
    // The reverse_mode_forward_pass mode does not have a reverse pass so
    // declarations don't have to be moved to the function global scope.
    bool promoteToFnScope = !getCurrentScope()->isFunctionScope() &&
                            m_Mode != DiffMode::reverse_mode_forward_pass;
    auto VDDerivedType = ComputeAdjointType(VD->getType());
    auto VDCloneType = CloneType(VD->getType());
    // If the cloned declaration is moved to the function global scope,
    // change its type for the corresponding adjoint type.
    if (promoteToFnScope)
      VDCloneType = VDDerivedType;
    bool isDerivativeOfRefType = VD->getType()->isReferenceType();
    VarDecl* VDDerived = nullptr;
    bool isPointerType = VD->getType()->isPointerType();
    bool isInitializedByNewExpr = false;
    // Check if the variable is pointer type and initialized by new expression
    if (isPointerType && VD->getInit() && isa<CXXNewExpr>(VD->getInit()))
      isInitializedByNewExpr = true;

    // VDDerivedInit now serves two purposes -- as the initial derivative value
    // or the size of the derivative array -- depending on the primal type.
    if (const auto* AT = dyn_cast<ArrayType>(VD->getType())) {
      VDDerivedInit = getArraySizeExpr(AT, m_Context, *this);
      VDDerived = BuildGlobalVarDecl(
          VDDerivedType, "_d_" + VD->getNameAsString(), VDDerivedInit, false,
          nullptr, clang::VarDecl::InitializationStyle::CallInit);
      // If an array-type declaration is promoted to function global,
      // its type is changed for clad::array. In that case we should
      // initialize it with its size the same way the derived variable
      // is initialized.
      if (promoteToFnScope)
        initDiff = VDDerivedInit;
    } else {
      // If VD is a reference to a local variable, then the initial value is set
      // to the derived variable of the corresponding local variable.
      // If VD is a reference to a non-local variable (global variable, struct
      // member etc), then no derived variable is available, thus `VDDerived`
      // does not need to reference any variable, consequentially the
      // `VDDerivedType` is the corresponding non-reference type and the initial
      // value is set to 0.
      // Otherwise, for non-reference types, the initial value is set to 0.
      VDDerivedInit = getZeroInit(VD->getType());

      // `specialThisDiffCase` is only required for correctly differentiating
      // the following code:
      // ```
      // Class _d_this_obj;
      // Class* _d_this = &_d_this_obj;
      // ```
      // Computation of hessian requires this code to be correctly
      // differentiated.
      bool specialThisDiffCase = false;
      if (const auto* MD = dyn_cast<CXXMethodDecl>(m_Function)) {
        if (VDDerivedType->isPointerType() && MD->isInstance()) {
          specialThisDiffCase = true;
        }
      }

      if (isDerivativeOfRefType) {
        initDiff = Visit(VD->getInit());
        if (!initDiff.getForwSweepExpr_dx()) {
          VDDerivedType =
              ComputeAdjointType(VD->getType().getNonReferenceType());
          isDerivativeOfRefType = false;
        }
        VDDerivedInit = getZeroInit(VDDerivedType);
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
      else if (isPointerType && VD->getInit()) {
        VDDerivedType = getNonConstType(VDDerivedType, m_Context, m_Sema);
        // If it's a pointer to a constant type, then remove the constness.
        if (VD->getType()->getPointeeType().isConstQualified()) {
          // first extract the pointee type
          auto pointeeType = VD->getType()->getPointeeType();
          // then remove the constness
          pointeeType.removeLocalConst();
          // then create a new pointer type with the new pointee type
          VDDerivedType = m_Context.getPointerType(pointeeType);
        }
        VDDerivedInit = getZeroInit(VDDerivedType);
      }
      VDDerived =
          BuildGlobalVarDecl(VDDerivedType, "_d_" + VD->getNameAsString(),
                             VDDerivedInit, false, nullptr, VD->getInitStyle());
    }

    // If `VD` is a reference to a local variable, then it is already
    // differentiated and should not be differentiated again.
    // If `VD` is a reference to a non-local variable then also there's no
    // need to call `Visit` since non-local variables are not differentiated.
    if (!isDerivativeOfRefType && (!isPointerType || isInitializedByNewExpr)) {
      Expr* derivedE = BuildDeclRef(VDDerived);
      if (isInitializedByNewExpr)
        derivedE = BuildOp(UnaryOperatorKind::UO_Deref, derivedE);
      if (VD->getInit()) {
        if (isa<CXXConstructExpr>(VD->getInit()))
          initDiff = Visit(VD->getInit());
        else
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
      if (isInsideLoop) {
        Stmt* assignToZero = BuildOp(BinaryOperatorKind::BO_Assign,
                                     BuildDeclRef(VDDerived),
                                     getZeroInit(VDDerivedType));
        addToCurrentBlock(assignToZero, direction::reverse);
      }
    } else if (isPointerType && VD->getInit()) {
      initDiff = Visit(VD->getInit());
    }
    VarDecl* VDClone = nullptr;
    Expr* derivedVDE = BuildDeclRef(VDDerived);

    // FIXME: Add extra parantheses if derived variable pointer is pointing to a
    // class type object.
    if (isDerivativeOfRefType) {
      Expr* assignDerivativeE =
          BuildOp(BinaryOperatorKind::BO_Assign, derivedVDE,
                  BuildOp(UnaryOperatorKind::UO_AddrOf,
                          initDiff.getForwSweepExpr_dx()));
      addToCurrentBlock(assignDerivativeE);
      if (isInsideLoop) {
        StmtDiff pushPop =
            StoreAndRestore(derivedVDE, /*prefix=*/"_t", /*force=*/true);
        addToCurrentBlock(pushPop.getExpr(), direction::forward);
        m_LoopBlock.back().push_back(pushPop.getExpr_dx());
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
    if (isDerivativeOfRefType && promoteToFnScope)
      VDClone = BuildGlobalVarDecl(
          VDCloneType, VD->getNameAsString(),
          BuildOp(UnaryOperatorKind::UO_AddrOf, initDiff.getExpr()),
          VD->isDirectInit());
    else
      VDClone = BuildGlobalVarDecl(VDCloneType, VD->getNameAsString(),
                                   initDiff.getExpr(), VD->isDirectInit(),
                                   nullptr, VD->getInitStyle());
    if (isPointerType) {
      Expr* assignDerivativeE = BuildOp(BinaryOperatorKind::BO_Assign,
                                        derivedVDE, initDiff.getExpr_dx());
      addToCurrentBlock(assignDerivativeE, direction::forward);
      if (isInsideLoop) {
        auto tape = MakeCladTapeFor(derivedVDE);
        addToCurrentBlock(tape.Push);
        auto* reverseSweepDerivativePointerE =
            BuildVarDecl(derivedVDE->getType(), "_t", tape.Pop);
        m_LoopBlock.back().push_back(
            BuildDeclStmt(reverseSweepDerivativePointerE));
        derivedVDE = BuildDeclRef(reverseSweepDerivativePointerE);
      }
    }
    m_Variables.emplace(VDClone, derivedVDE);

    return VarDeclDiff(VDClone, VDDerived);
  }

  // TODO: 'shouldEmit' parameter should be removed after converting
  // Error estimation framework to callback style. Some more research
  // need to be done to
  StmtDiff
  ReverseModeVisitor::DifferentiateSingleStmt(const Stmt* S, Expr* dfdS) {
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

    if (dxInForward)
      addToCurrentBlock(stmtDx, direction::forward);
    else
      addToCurrentBlock(SDiff.getStmt_dx(), direction::reverse);
    CompoundStmt* RCS = endBlock(direction::reverse);
    std::reverse(RCS->body_begin(), RCS->body_end());
    Stmt* ReverseResult = unwrapIfSingleStmt(RCS);
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
    Stmt* ReverseResult = unwrapIfSingleStmt(RCS);
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
    bool promoteToFnScope = !getCurrentScope()->isFunctionScope() &&
                            m_Mode != DiffMode::reverse_mode_forward_pass;
    // For each variable declaration v, create another declaration _d_v to
    // store derivatives for potential reassignments. E.g.
    // double y = x;
    // ->
    // double _d_y = _d_x; double y = x;
    for (auto* D : DS->decls()) {
      if (auto* VD = dyn_cast<VarDecl>(D)) {
        VarDeclDiff VDDiff = DifferentiateVarDecl(VD);

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
        if (VDDiff.getDecl()->getDeclName() != VD->getDeclName() ||
            VD->getType() != VDDiff.getDecl()->getType())
          m_DeclReplacements[VD] = VDDiff.getDecl();

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
              auto pushPop =
                  StoreAndRestore(declRef, /*prefix=*/"_t", /*force=*/true);
              if (pushPop.getExpr() != declRef)
                addToCurrentBlock(pushPop.getExpr_dx(), direction::reverse);
              assignment = BuildOp(BO_Comma, pushPop.getExpr(), assignment);
            }
            inits.push_back(assignment);
            if (isa<ArrayType>(VD->getType())) {
              decl->setInitStyle(VarDecl::InitializationStyle::CallInit);
              decl->setInit(Clone(VDDiff.getDecl_dx()->getInit()));
            } else {
              decl->setInit(getZeroInit(VD->getType()));
            }
          }
        }

        decls.push_back(VDDiff.getDecl());
        if (isa<VariableArrayType>(VD->getType()))
          localDeclsDiff.push_back(VDDiff.getDecl_dx());
        else
          declsDiff.push_back(VDDiff.getDecl_dx());
      } else {
        diag(DiagnosticsEngine::Warning,
             D->getEndLoc(),
             "Unsupported declaration");
      }
    }

    Stmt* DSClone = BuildDeclStmt(decls);

    if (!localDeclsDiff.empty()) {
      Stmt* localDSDIff = BuildDeclStmt(localDeclsDiff);
      addToCurrentBlock(
          localDSDIff,
          clad::rmv::forward); // Doesnt work for arrays decl'd in loops.
    }
    if (!declsDiff.empty()) {
      Stmt* DSDiff = BuildDeclStmt(declsDiff);
      addToBlock(DSDiff, m_Globals);
    }

    if (m_ExternalSource) {
      declsDiff.append(localDeclsDiff.begin(), localDeclsDiff.end());
      m_ExternalSource->ActBeforeFinalizingVisitDeclStmt(decls, declsDiff);
    }

    // This part in necessary to replace local variables inside loops
    // with function globals and replace initializations with assignments.
    if (promoteToFnScope) {
      addToBlock(DSClone, m_Globals);
      Stmt* initAssignments = MakeCompoundStmt(inits);
      initAssignments = unwrapIfSingleStmt(initAssignments);
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

  StmtDiff ReverseModeVisitor::VisitMemberExpr(const MemberExpr* ME) {
    auto baseDiff = VisitWithExplicitNoDfDx(ME->getBase());
    auto* field = ME->getMemberDecl();
    assert(!isa<CXXMethodDecl>(field) &&
           "CXXMethodDecl nodes not supported yet!");
    MemberExpr* clonedME = utils::BuildMemberExpr(
        m_Sema, getCurrentScope(), baseDiff.getExpr(), field->getName());
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
    // We lack context to decide if this is useful to store or not. In the
    // current system that should have been decided by the parent expression.
    // FIXME: Here will be the entry point of the advanced activity analysis.
    if (isa<DeclRefExpr>(B) || isa<ArraySubscriptExpr>(B) ||
        isa<MemberExpr>(B)) {
      // If TBR analysis is off, assume E is useful to store.
      if (!enableTBR)
        return true;
      // FIXME: currently, we allow all pointer operations to be stored.
      // This is not correct, but we need to implement a more advanced analysis
      // to determine which pointer operations are useful to store.
      if (E->getType()->isPointerType())
        return true;
      auto found = m_ToBeRecorded.find(B->getBeginLoc());
      return found != m_ToBeRecorded.end();
    }

    // FIXME: Attach checkpointing.
    if (isa<CallExpr>(B))
      return false;

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

  StmtDiff ReverseModeVisitor::GlobalStoreAndRef(Expr* E,
                                                 QualType Type,
                                                 llvm::StringRef prefix,
                                                 bool force) {
    assert(E && "must be provided, otherwise use DelayedGlobalStoreAndRef");
    if (!force && !UsefulToStoreGlobal(E))
      return {E, E};

    auto pushPop = BuildPushPop(E, Type, prefix, force);
    if (!isInsideLoop) {
      if (E) {
        Expr* Set = BuildOp(BO_Assign, pushPop.getExpr(), E);
        addToCurrentBlock(Set, direction::forward);
      }
    }

    return pushPop;
  }

  StmtDiff ReverseModeVisitor::GlobalStoreAndRef(Expr* E,
                                                 llvm::StringRef prefix,
                                                 bool force) {
    assert(E && "cannot infer type");
    return GlobalStoreAndRef(
        E, getNonConstType(E->getType(), m_Context, m_Sema), prefix, force);
  }

  StmtDiff ReverseModeVisitor::BuildPushPop(clang::Expr* E,
                                            clang::QualType Type,
                                            llvm::StringRef prefix,
                                            bool force) {
    if (isInsideLoop) {
      auto CladTape = MakeCladTapeFor(Clone(E));
      Expr* Push = CladTape.Push;
      Expr* Pop = CladTape.Pop;
      return {Push, Pop};
    }
    Expr* init = nullptr;
    if (const auto* const AT = dyn_cast<ArrayType>(Type))
      init = getArraySizeExpr(AT, m_Context, *this);

    Expr* Ref = BuildDeclRef(GlobalStoreImpl(Type, prefix, init));
    return {Ref, Ref};
  }

  StmtDiff ReverseModeVisitor::StoreAndRestore(clang::Expr* E,
                                               llvm::StringRef prefix,
                                               bool force) {
    auto Type = getNonConstType(E->getType(), m_Context, m_Sema);

    if (!force && !UsefulToStoreGlobal(E))
      return {};

    if (isInsideLoop) {
      auto pushPop = BuildPushPop(E, Type, prefix, true);
      auto* popAssign = BuildOp(BinaryOperatorKind::BO_Assign, Clone(E),
                                pushPop.getExpr_dx());
      return {pushPop.getExpr(), popAssign};
    }

    Expr* init = nullptr;
    if (const auto* AT = dyn_cast<ArrayType>(Type))
      init = getArraySizeExpr(AT, m_Context, *this);

    Expr* Ref = BuildDeclRef(GlobalStoreImpl(Type, prefix, init));
    if (E) {
      Expr* Store = BuildOp(BO_Assign, Ref, Clone(E));
      Expr* Restore = nullptr;
      if (E->isModifiableLvalue(m_Context) == Expr::MLV_Valid) {
        auto* r = Clone(E);
        Restore = BuildOp(BO_Assign, r, Ref);
      }
      return {Store, Restore};
    }
    return {};
  }

  void ReverseModeVisitor::DelayedStoreResult::Finalize(Expr* New) {
    if (isConstant || !needsUpdate)
      return;
    if (isInsideLoop) {
      auto* Push = cast<CallExpr>(Result.getExpr());
      unsigned lastArg = Push->getNumArgs() - 1;
      Push->setArg(lastArg, V.m_Sema.DefaultLvalueConversion(New).get());
    } else {
      V.addToCurrentBlock(V.BuildOp(BO_Assign, Result.getExpr(), New),
                          direction::forward);
    }
  }

  ReverseModeVisitor::DelayedStoreResult
  ReverseModeVisitor::DelayedGlobalStoreAndRef(Expr* E,
                                               llvm::StringRef prefix) {
    assert(E && "must be provided");
    if (!UsefulToStore(E)) {
      StmtDiff Ediff = Visit(E);
      Expr::EvalResult evalRes;
      bool isConst =
          clad_compat::Expr_EvaluateAsConstantExpr(E, evalRes, m_Context);
      return DelayedStoreResult{*this, Ediff,
                                /*isConstant*/ isConst,
                                /*isInsideLoop*/ false,
                                /*pNeedsUpdate=*/false};
    }
    if (isInsideLoop) {
      Expr* dummy = E;
      auto CladTape = MakeCladTapeFor(dummy);
      Expr* Push = CladTape.Push;
      Expr* Pop = CladTape.Pop;
      return DelayedStoreResult{*this, StmtDiff{Push, nullptr, nullptr, Pop},
                                /*isConstant*/ false,
                                /*isInsideLoop*/ true, /*pNeedsUpdate=*/true};
    }
    Expr* Ref = BuildDeclRef(GlobalStoreImpl(
        getNonConstType(E->getType(), m_Context, m_Sema), prefix));
    // Return reference to the declaration instead of original expression.
    return DelayedStoreResult{*this, StmtDiff{Ref, nullptr, nullptr, Ref},
                              /*isConstant*/ false,
                              /*isInsideLoop*/ false, /*pNeedsUpdate=*/true};
  }

  ReverseModeVisitor::LoopCounter::LoopCounter(ReverseModeVisitor& RMV)
      : m_RMV(RMV) {
    ASTContext& C = m_RMV.m_Context;
    if (RMV.isInsideLoop) {
      auto* zero = ConstantFolder::synthesizeLiteral(C.getSizeType(), C,
                                                     /*val=*/0);
      auto counterTape = m_RMV.MakeCladTapeFor(zero);
      m_Ref = counterTape.Last();
      m_Pop = counterTape.Pop;
      m_Push = counterTape.Push;
    } else {
      m_Ref = m_RMV
                  .GlobalStoreAndRef(m_RMV.getZeroInit(C.IntTy),
                                     C.getSizeType(), "_t",
                                     /*force=*/true)
                  .getExpr();
    }
  }

  StmtDiff ReverseModeVisitor::VisitWhileStmt(const WhileStmt* WS) {
    LoopCounter loopCounter(*this);
    if (loopCounter.getPush())
      addToCurrentBlock(loopCounter.getPush());

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
    Stmt* reverseBlock = reverseWS;
    // If loop counter have to be popped then create a compound statement
    // enclosing the reverse pass while statement and loop counter pop
    // expression.
    //
    // Therefore, reverse pass code will look like this:
    // {
    //   while (_t) {
    //
    //   }
    //   clad::pop(_t);
    // }
    if (loopCounter.getPop()) {
      beginBlock(direction::reverse);
      addToCurrentBlock(loopCounter.getPop(), direction::reverse);
      addToCurrentBlock(reverseWS, direction::reverse);
      reverseBlock = endBlock(direction::reverse);
    }
    return {forwardWS, reverseBlock};
  }

  StmtDiff ReverseModeVisitor::VisitDoStmt(const DoStmt* DS) {
    LoopCounter loopCounter(*this);
    if (loopCounter.getPush())
      addToCurrentBlock(loopCounter.getPush());

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
    Stmt* reverseBlock = reverseDS;
    // If loop counter have to be popped then create a compound statement
    // enclosing the reverse pass while statement and loop counter pop
    // expression.
    //
    // Therefore, reverse pass code will look like this:
    // {
    //   do {
    //
    //   } while (_t);
    //   clad::pop(_t);
    // }
    if (loopCounter.getPop()) {
      beginBlock(direction::reverse);
      addToCurrentBlock(loopCounter.getPop(), direction::reverse);
      addToCurrentBlock(reverseDS, direction::reverse);
      reverseBlock = endBlock(direction::reverse);
    }
    return {forwardDS, reverseBlock};
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
    Expr* condExpr = nullptr;
    clad_compat::llvm_Optional<CladTapeResult> condTape;

    if (isInsideLoop) {
      // If we are inside a loop, condition will be stored and used as follows:
      //
      // forward block:
      // switch (clad::push(..., cond)) { ... }
      //
      // reverse block:
      // switch (...) { ... }
      // clad::pop(...);
      condTape.emplace(MakeCladTapeFor(condDiff.getExpr(), "_cond"));
      condExpr = condTape->Push;
    } else {
      condExpr = GlobalStoreAndRef(condDiff.getExpr(), "_cond").getExpr();
    }

    auto* activeBreakContHandler = PushBreakContStmtHandler(
        /*forSwitchStmt=*/true);
    activeBreakContHandler->BeginCFSwitchStmtScope();
    auto* SSData = PushSwitchStmtInfo();

    if (isInsideLoop)
      SSData->switchStmtCond = condTape->Last();
    else
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
      if (isInsideLoop)
        addToCurrentBlock(condTape->Pop, direction::reverse);
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

    auto* newSC = clad_compat::CaseStmt_Create(m_Sema.getASTContext(), lhsClone,
                                               rhsClone, noLoc, noLoc, noLoc);

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

      Stmt* reverseBlock = unwrapIfSingleStmt(bodyDiff.getStmt_dx());
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
                unwrapIfSingleStmt(endBlock(direction::reverse))};
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
    CaseStmt* CS = clad_compat::CaseStmt_Create(m_RMV.m_Context, counterLiteral,
                                                nullptr, noLoc, noLoc, noLoc);

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

  // FIXME: Add support for differentiating calls to constructors.
  // We currently assume that constructor arguments are non-differentiable.
  StmtDiff
  ReverseModeVisitor::VisitCXXConstructExpr(const CXXConstructExpr* CE) {
    llvm::SmallVector<Expr*, 4> clonedArgs;
    for (const auto* arg : CE->arguments()) {
      auto argDiff = Visit(arg, dfdx());
      clonedArgs.push_back(argDiff.getExpr());
    }
    Expr* clonedArgsE = nullptr;

    if (CE->getNumArgs() != 1) {
      if (CE->isListInitialization()) {
        clonedArgsE = m_Sema.ActOnInitList(noLoc, clonedArgs, noLoc).get();
      } else {
        if (CE->getNumArgs() == 0) {
          // ParenList is empty -- default initialisation.
          // Passing empty parenList here will silently cause 'most vexing
          // parse' issue.
          return StmtDiff();
        }
        clonedArgsE = m_Sema.ActOnParenListExpr(noLoc, noLoc, clonedArgs).get();
      }
    } else {
      clonedArgsE = clonedArgs[0];
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

  QualType ReverseModeVisitor::GetParameterDerivativeType(QualType yType,
                                                          QualType xType) {

    if (m_Mode == DiffMode::reverse)
      assert(yType->isRealType() &&
             "yType should be a non-reference builtin-numerical scalar type!!");
    QualType xValueType = utils::GetValueType(xType);
    // derivative variables should always be of non-const type.
    xValueType.removeLocalConst();
    QualType nonRefXValueType = xValueType.getNonReferenceType();
    return GetCladArrayRefOfType(nonRefXValueType);
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
    if (isa<ArrayType>(T) && !isa<IncompleteArrayType>(T)) {
      QualType adjointType =
          GetCladArrayOfType(m_Context.getBaseElementType(T));
      return adjointType;
    }
    T.removeLocalConst();
    return T;
  }

  clang::QualType ReverseModeVisitor::ComputeParamType(clang::QualType T) {
      QualType TValueType = utils::GetValueType(T);
      TValueType.removeLocalConst();
      return GetCladArrayRefOfType(TValueType);
  }

  llvm::SmallVector<clang::QualType, 8>
  ReverseModeVisitor::ComputeParamTypes(const DiffParams& diffParams) {
    llvm::SmallVector<clang::QualType, 8> paramTypes;
    paramTypes.reserve(m_Function->getNumParams() * 2);
    for (auto* PVD : m_Function->parameters())
      paramTypes.push_back(PVD->getType());
    // TODO: Add DiffMode::experimental_pullback support here as well.
    if (m_Mode == DiffMode::reverse ||
        m_Mode == DiffMode::experimental_pullback) {
      QualType effectiveReturnType =
          m_Function->getReturnType().getNonReferenceType();
      if (m_Mode == DiffMode::experimental_pullback) {
        // FIXME: Generally, we use the function's return type as the argument's
        // derivative type. We cannot follow this strategy for `void` function
        // return type. Thus, temporarily use `double` type as the placeholder
        // type for argument derivatives. We should think of a more uniform and
        // consistent solution to this problem. One effective strategy that may
        // hold well: If we are differentiating a variable of type Y with
        // respect to variable of type X, then the derivative should be of type
        // X. Check this related issue for more details:
        // https://github.com/vgvassilev/clad/issues/385
        if (effectiveReturnType->isVoidType())
          effectiveReturnType = m_Context.DoubleTy;
        else
          paramTypes.push_back(effectiveReturnType);
      }

      if (const auto* MD = dyn_cast<CXXMethodDecl>(m_Function)) {
        const CXXRecordDecl* RD = MD->getParent();
        if (MD->isInstance() && !RD->isLambda()) {
          QualType thisType =
              clad_compat::CXXMethodDecl_getThisType(m_Sema, MD);
          paramTypes.push_back(
              GetParameterDerivativeType(effectiveReturnType, thisType));
        }
      }

      for (auto* PVD : m_Function->parameters()) {
        const auto* it =
            std::find(std::begin(diffParams), std::end(diffParams), PVD);
        if (it != std::end(diffParams))
          paramTypes.push_back(ComputeParamType(PVD->getType()));
      }
    } else if (m_Mode == DiffMode::jacobian) {
      std::size_t lastArgIdx = m_Function->getNumParams() - 1;
      QualType derivativeParamType =
          m_Function->getParamDecl(lastArgIdx)->getType();
      paramTypes.push_back(derivativeParamType);
    }
    return paramTypes;
  }

  llvm::SmallVector<clang::ParmVarDecl*, 8>
  ReverseModeVisitor::BuildParams(DiffParams& diffParams) {
    llvm::SmallVector<clang::ParmVarDecl*, 8> params;
    llvm::SmallVector<clang::ParmVarDecl*, 8> paramDerivatives;
    params.reserve(m_Function->getNumParams() + diffParams.size());
    const auto* derivativeFnType =
        cast<FunctionProtoType>(m_Derivative->getType());
    std::size_t dParamTypesIdx = m_Function->getNumParams();

    if (m_Mode == DiffMode::experimental_pullback &&
        !m_Function->getReturnType()->isVoidType()) {
      ++dParamTypesIdx;
    }

    if (const auto* MD = dyn_cast<CXXMethodDecl>(m_Function)) {
      const CXXRecordDecl* RD = MD->getParent();
      if (!isVectorValued && MD->isInstance() && !RD->isLambda()) {
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

    for (auto* PVD : m_Function->parameters()) {
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
        if (m_Mode == DiffMode::reverse ||
            m_Mode == DiffMode::experimental_pullback) {
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
            QualType valueType = DetermineCladArrayValueType(dPVD->getType());
            m_Variables[*it] = BuildOp(UO_Deref, BuildDeclRef(dPVD),
                                       m_Function->getLocation());
            // Add additional paranthesis if derivative is of record type
            // because `*derivative.someField` will be incorrectly evaluated if
            // the derived function is compiled standalone.
            if (valueType->isRecordType())
              m_Variables[*it] =
                  utils::BuildParenExpr(m_Sema, m_Variables[*it]);
          }
        }
      }
    }

    if (m_Mode == DiffMode::experimental_pullback &&
        !m_Function->getReturnType()->isVoidType()) {
      IdentifierInfo* pullbackParamII = CreateUniqueIdentifier("_d_y");
      QualType pullbackType =
          derivativeFnType->getParamType(m_Function->getNumParams());
      ParmVarDecl* pullbackPVD = utils::BuildParmVarDecl(
          m_Sema, m_Derivative, pullbackParamII, pullbackType);
      paramDerivatives.insert(paramDerivatives.begin(), pullbackPVD);

      if (pullbackPVD->getIdentifier())
        m_Sema.PushOnScopeChains(pullbackPVD, getCurrentScope(),
                                 /*AddToContext=*/false);

      m_Pullback = BuildDeclRef(pullbackPVD);
      ++dParamTypesIdx;
    }

    if (m_Mode == DiffMode::jacobian) {
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
    if (m_Mode == DiffMode::reverse ||
        m_Mode == DiffMode::experimental_pullback)
      m_IndependentVars.insert(m_IndependentVars.end(), diffParams.begin(),
                               diffParams.end());
    return params;
  }
} // end namespace clad
