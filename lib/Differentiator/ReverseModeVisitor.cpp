//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// version: $Id: ClangPlugin.cpp 7 2013-06-01 22:48:03Z v.g.vassilev@gmail.com $
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#include "clad/Differentiator/ReverseModeVisitor.h"

#include "ConstantFolder.h"

#include "clad/Differentiator/DiffPlanner.h"
#include "clad/Differentiator/ErrorEstimator.h"
#include "clad/Differentiator/StmtClone.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include "clang/AST/TemplateBase.h"
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
    QualType TapeType =
        GetCladTapeOfType(getNonConstType(E->getType(), m_Context, m_Sema));
    LookupResult& Push = GetCladTapePush();
    LookupResult& Pop = GetCladTapePop();
    Expr* TapeRef = BuildDeclRef(GlobalStoreImpl(TapeType, prefix));
    auto VD = cast<VarDecl>(cast<DeclRefExpr>(TapeRef)->getDecl());
    // Add fake location, since Clang AST does assert(Loc.isValid()) somewhere.
    VD->setLocation(m_Function->getLocation());
    m_Sema.AddInitializerToDecl(VD, getZeroInit(TapeType), false);
    CXXScopeSpec CSS;
    CSS.Extend(m_Context, GetCladNamespace(), noLoc, noLoc);
    auto PopDRE = m_Sema
                      .BuildDeclarationNameExpr(CSS, Pop,
                                                /*AcceptInvalidDecl=*/false)
                      .get();
    auto PushDRE = m_Sema
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

  ReverseModeVisitor::~ReverseModeVisitor() {}

  FunctionDecl* ReverseModeVisitor::CreateGradientOverload(
      SmallVectorImpl<QualType>& GradientParamTypes,
      SmallVectorImpl<ParmVarDecl*>& GradientParams,
      DeclarationNameInfo& GradientName, FunctionDecl* GradientFD) {
    // Calculate the total number of parameters that would be present in the
    // derived function if all args are requested
    size_t totalDerivedParamsSize = m_Function->getNumParams() * 2;
    // Calculate the number of arguments that we need to append for the
    // overload function. These arguments are provided default argument
    // like functionality during execution.
    size_t remainingArgs = totalDerivedParamsSize - GradientParamTypes.size();

    QualType DerivedOutputParamType =
        GetCladArrayRefOfType(m_Function->getReturnType());

    for (size_t i = 0; i < remainingArgs; i++) {
      GradientParamTypes.push_back(DerivedOutputParamType);
    }

    llvm::ArrayRef<QualType>
        gradFuncOverloadParamTyAR(GradientParamTypes.data(),
                                  GradientParamTypes.size());
    auto gradFuncOverloadEPI =
        dyn_cast<FunctionProtoType>(m_Function->getType())->getExtProtoInfo();
    QualType gradientFunctionOverloadType =
        m_Context.getFunctionType(m_Context.VoidTy, gradFuncOverloadParamTyAR,
                                  // Cast to function pointer.
                                  gradFuncOverloadEPI);

    DeclContext* DC = const_cast<DeclContext*>(m_Function->getDeclContext());
    m_Sema.CurContext = DC;
    DeclWithContext gradientOverloadFDWC =
        m_Builder.cloneFunction(m_Function, *this, DC, m_Sema, m_Context, noLoc,
                                GradientName, gradientFunctionOverloadType);
    FunctionDecl* gradientOverloadFD = gradientOverloadFDWC.first;

    beginScope(Scope::FunctionPrototypeScope | Scope::FunctionDeclarationScope |
               Scope::DeclScope);
    m_Sema.PushFunctionScope();
    m_Sema.PushDeclContext(getCurrentScope(), gradientOverloadFD);

    llvm::SmallVector<ParmVarDecl*, 4> overloadParams;
    llvm::SmallVector<Expr*, 4> callArgs;

    overloadParams.reserve(m_Function->getNumParams() * 2);
    callArgs.reserve(GradientParamTypes.size());

    for (auto* GVD : GradientParams) {
      Expr* GVDDefArg =
          GVD->hasDefaultArg() ? Clone(GVD->getDefaultArg()) : nullptr;
      auto* VD =
          ParmVarDecl::Create(m_Context, gradientOverloadFD, noLoc, noLoc,
                              GVD->getIdentifier(), GVD->getType(),
                              GVD->getTypeSourceInfo(), GVD->getStorageClass(),
                              // Clone default arg if present.
                              GVDDefArg);
      if (VD->getIdentifier())
        m_Sema.PushOnScopeChains(VD, getCurrentScope(),
                                 /*AddToContext=*/false);
      overloadParams.push_back(VD);
      callArgs.push_back((Expr*)BuildDeclRef(VD));
    }

    for (size_t i = 0; i < remainingArgs; i++) {
      IdentifierInfo* DVDII = &m_Context.Idents.get("_d_" + std::to_string(i));

      auto DVD = ParmVarDecl::
          Create(m_Context, gradientOverloadFD, noLoc, noLoc, DVDII,
                 DerivedOutputParamType,
                 m_Context.getTrivialTypeSourceInfo(DerivedOutputParamType,
                                                    noLoc),
                 StorageClass::SC_None,
                 /*DefArg=*/nullptr);
      if (DVD->getIdentifier())
        m_Sema.PushOnScopeChains(DVD, getCurrentScope(),
                                 /*AddToContext=*/false);
      overloadParams.push_back(DVD);
    }

    llvm::ArrayRef<ParmVarDecl*> overloadParamsRef =
        llvm::makeArrayRef(overloadParams.data(), overloadParams.size());

    llvm::MutableArrayRef<Expr*> callArgsRef =
        llvm::makeMutableArrayRef(callArgs.data(), callArgs.size());

    gradientOverloadFD->setParams(overloadParamsRef);
    gradientOverloadFD->setBody(/*B=*/nullptr);

    beginScope(Scope::FnScope | Scope::DeclScope);
    m_DerivativeFnScope = getCurrentScope();
    beginBlock();

    Expr* callExpr = BuildCallExprToFunction(GradientFD, callArgsRef,
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

  OverloadedDeclWithContext
  ReverseModeVisitor::Derive(const FunctionDecl* FD,
                             const DiffRequest& request) {
    silenceDiags = !request.VerboseDiags;
    m_Function = FD;
    assert(m_Function && "Must not be null.");
    m_ErrorEstimationEnabled = request.Mode == DiffMode::error_estimation;

    DiffParams args{};
    if (request.Args) {
      std::tie(args, std::ignore) = request.DiffParamsInfo;
    }
    else
      std::copy(FD->param_begin(), FD->param_end(), std::back_inserter(args));
    if (args.empty())
      return {};

    // Save the type of the output parameter(s) that is add by clad to the
    // derived function
    clang::QualType DerivedOutputParamType;
    if (request.Mode == DiffMode::jacobian) {
      isVectorValued = true;
      unsigned lastArgN = m_Function->getNumParams() - 1;
      outputArrayStr = m_Function->getParamDecl(lastArgN)->getNameAsString();
      DerivedOutputParamType = m_Function->getParamDecl(lastArgN)->getType();
    } else {
      DerivedOutputParamType =
          GetCladArrayRefOfType(m_Function->getReturnType());
    }

    auto derivativeBaseName = request.BaseFunctionName;
    std::string gradientName = derivativeBaseName + funcPostfix();
    // To be consistent with older tests, nothing is appended to 'f_grad' if
    // we differentiate w.r.t. all the parameters at once.
    if (!(args.size() == FD->getNumParams() &&
          std::equal(FD->param_begin(), FD->param_end(), std::begin(args)))) {
      for (auto arg : args) {
        auto it = std::find(FD->param_begin(), FD->param_end(), arg);
        auto idx = std::distance(FD->param_begin(), it);
        gradientName += ('_' + std::to_string(idx));
      }
    }

    if (isVectorValued)
      args.pop_back();

    IdentifierInfo* II = &m_Context.Idents.get(gradientName);
    DeclarationNameInfo name(II, noLoc);

    // A vector of types of the gradient function parameters.
    llvm::SmallVector<QualType, 16> paramTypes;
    llvm::SmallVector<QualType, 16> outputParamTypes;
    // If we are in error estimation mode, we have an extra `double&`
    // parameter that stores the final error
    unsigned numExtraParam = m_ErrorEstimationEnabled ? 1 : 0;
    paramTypes.reserve(m_Function->getNumParams() * 2);
    outputParamTypes.reserve(args.size());
    for (auto* PVD : m_Function->parameters()) {
      paramTypes.push_back(PVD->getType());

      if (!isVectorValued) {
        auto it = std::find(std::begin(args), std::end(args), PVD);
        if (it != std::end(args)) {
          outputParamTypes.push_back(DerivedOutputParamType);
        }
      }
    }
    if (isVectorValued) {
      paramTypes.push_back(DerivedOutputParamType);
    } else {
      paramTypes.insert(paramTypes.end(), outputParamTypes.begin(),
                        outputParamTypes.end());
    }
    // If we are performing error estimation, our gradient function
    // will have an extra argument which will hold the final error value
    if (m_ErrorEstimationEnabled) {
      paramTypes.push_back(
          m_Context.getLValueReferenceType(m_Context.DoubleTy));
    }
    // If reverse mode differentiates only part of the arguments it needs to
    // generate an overload that can take in all the diff variables
    bool shouldCreateOverload = false;
    // Calculate the total number of parameters that would be present in the
    // derived function if all args are requested
    size_t totalDerivedParamsSize = m_Function->getNumParams() * 2 +
                                    numExtraParam;
    if (paramTypes.size() != totalDerivedParamsSize && !isVectorValued)
      shouldCreateOverload = true;

    auto originalFnType = dyn_cast<FunctionProtoType>(m_Function->getType());
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
    llvm::SaveAndRestore<Scope*> SaveScope(m_CurScope);
    DeclContext* DC = const_cast<DeclContext*>(m_Function->getDeclContext());
    m_Sema.CurContext = DC;
    DeclWithContext result = m_Builder.cloneFunction(m_Function,
                                                     *this,
                                                     DC,
                                                     m_Sema,
                                                     m_Context,
                                                     noLoc,
                                                     name,
                                                     gradientFunctionType);
    FunctionDecl* gradientFD = result.first;
    m_Derivative = gradientFD;

    // Function declaration scope
    beginScope(Scope::FunctionPrototypeScope | Scope::FunctionDeclarationScope |
               Scope::DeclScope);
    m_Sema.PushFunctionScope();
    m_Sema.PushDeclContext(getCurrentScope(), m_Derivative);

    // Create parameter declarations.
    llvm::SmallVector<ParmVarDecl*, 4> params;
    llvm::SmallVector<ParmVarDecl*, 4> outputParams;
    params.reserve(m_Function->getNumParams() + args.size() + numExtraParam);
    for (auto* PVD : m_Function->parameters()) {
      Expr* PVDDefArg =
          PVD->hasDefaultArg() ? Clone(PVD->getDefaultArg()) : nullptr;
      auto VD =
          ParmVarDecl::Create(m_Context, gradientFD, noLoc, noLoc,
                              PVD->getIdentifier(), PVD->getType(),
                              PVD->getTypeSourceInfo(), PVD->getStorageClass(),
                              // Clone default arg if present.
                              PVDDefArg);
      if (VD->getIdentifier())
        m_Sema.PushOnScopeChains(VD, getCurrentScope(),
                                 /*AddToContext=*/false);
      params.push_back(VD);

      // Create the diff params in the derived function for independent
      // variables
      auto it = std::find(std::begin(args), std::end(args), PVD);
      if (it != std::end(args)) {
        *it = VD;

        if (!isVectorValued) {
          IdentifierInfo* DVDII =
              &m_Context.Idents.get("_d_" + PVD->getNameAsString());

          auto DVD = ParmVarDecl::
              Create(m_Context, gradientFD, noLoc, noLoc, DVDII,
                     DerivedOutputParamType,
                     m_Context.getTrivialTypeSourceInfo(DerivedOutputParamType,
                                                        noLoc),
                     PVD->getStorageClass(),
                     // Clone default arg if present.
                     /*DefArg=*/nullptr);
          if (DVD->getIdentifier())
            m_Sema.PushOnScopeChains(DVD, getCurrentScope(),
                                     /*AddToContext=*/false);
          outputParams.push_back(DVD);
          if (isArrayOrPointerType(PVD->getType())) {
            m_Variables[*it] = (Expr*)BuildDeclRef(DVD);
          } else {
            m_Variables[*it] =
                BuildOp(UO_Deref, BuildDeclRef(DVD), m_Function->getLocation());
          }
        }
      }
    }
    auto nonDiffParams = params;
    if (isVectorValued) {
      TypeSourceInfo* paramTSI =
          m_Context.getTrivialTypeSourceInfo(DerivedOutputParamType, noLoc);
      // The output parameter "_jacobianMatrix".
      params.push_back(
          ParmVarDecl::Create(m_Context, gradientFD, noLoc, noLoc,
                              &m_Context.Idents.get("jacobianMatrix"),
                              DerivedOutputParamType, paramTSI,
                              params.front()->getStorageClass(),
                              /*DefArg=*/nullptr));
      if (params.back()->getIdentifier())
        m_Sema.PushOnScopeChains(params.back(), getCurrentScope(),
                                 /*AddToContext=*/false);
    } else {
      params.insert(params.end(), outputParams.begin(), outputParams.end());
      m_IndependentVars.insert(m_IndependentVars.end(), args.begin(),
                               args.end());
    }
    // If in error estimation mode, create the error parameter
    if (m_ErrorEstimationEnabled) {
      // Repeat the above but for the error ouput var "_final_error"
      ParmVarDecl *errorVarDecl = ParmVarDecl::Create(
          m_Context, gradientFD, noLoc, noLoc,
          &m_Context.Idents.get("_final_error"), paramTypes.back(),
          m_Context.getTrivialTypeSourceInfo(paramTypes.back(), noLoc),
          params.front()->getStorageClass(),
          /*DefArg=*/nullptr);
      params.push_back(errorVarDecl);
      m_Sema.PushOnScopeChains(params.back(), getCurrentScope(),
                               /*AddToContext=*/false);
    }
    llvm::ArrayRef<ParmVarDecl*> paramsRef =
        llvm::makeArrayRef(params.data(), params.size());
    gradientFD->setParams(paramsRef);
    gradientFD->setBody(nullptr);

    if (isVectorValued) {
      // Reference to the output parameter.
      m_Result = BuildDeclRef(params.back());
      numParams = args.size();

      // Creates the ArraySubscriptExprs for the independent variables
      size_t idx = 0;
      for (auto arg : args) {
        // FIXME: fix when adding array inputs, now we are just skipping all
        // array/pointer inputs (not treating them as independent variables).
        if (isArrayOrPointerType(arg->getType())) {
          if (arg->getName() == "p")
            m_Variables[arg] = m_Result;
          idx += 1;
          continue;
        }
        auto size_type = m_Context.getSizeType();
        unsigned size_type_bits = m_Context.getIntWidth(size_type);
        // Create the idx literal.
        auto i =
            IntegerLiteral::Create(m_Context, llvm::APInt(size_type_bits, idx),
                                   size_type, noLoc);
        // Create the jacobianMatrix[idx] expression.
        auto result_at_i =
            m_Sema.CreateBuiltinArraySubscriptExpr(m_Result, noLoc, i, noLoc)
                .get();
        m_Variables[arg] = result_at_i;
        idx += 1;
        m_IndependentVars.push_back(arg);
      }
    }
    // Reference to the final error statement
    if (m_ErrorEstimationEnabled)
      errorEstHandler->SetFinalErrorExpr(BuildDeclRef(params.back()));
    // Function body scope.
    beginScope(Scope::FnScope | Scope::DeclScope);
    m_DerivativeFnScope = getCurrentScope();
    beginBlock();
    // create derived variables for parameters which are not part of
    // independent variables (args).
    for (ParmVarDecl* param : nonDiffParams) {
      // derived variables are already created for independent variables.
      if (m_Variables.count(param))
        continue;
      // in vector mode last non diff parameter is output parameter.
      if (isVectorValued && param == nonDiffParams.back())
        continue;
      auto VDDerivedType = param->getType();
      // We cannot initialize derived variable for pointer types because
      // we do not know the correct size.
      if (isArrayOrPointerType(VDDerivedType))
        continue;
      auto VDDerived =
          BuildVarDecl(VDDerivedType, "_d_" + param->getNameAsString(),
                       getZeroInit(VDDerivedType));
      m_Variables[param] = BuildDeclRef(VDDerived);
      addToBlock(BuildDeclStmt(VDDerived), m_Globals);
    }
    // Start the visitation process which outputs the statements in the current
    // block.
    StmtDiff BodyDiff = Visit(FD->getBody());
    Stmt* Forward = BodyDiff.getStmt();
    Stmt* Reverse = BodyDiff.getStmt_dx();
    // Create the body of the function.
    // Firstly, all "global" Stmts are put into fn's body.
    for (Stmt* S : m_Globals)
      addToCurrentBlock(S, forward);
    // Forward pass.
    if (auto CS = dyn_cast<CompoundStmt>(Forward))
      for (Stmt* S : CS->body())
        addToCurrentBlock(S, forward);
    else
      addToCurrentBlock(Forward, forward);
    // Reverse pass.
    if (auto RCS = dyn_cast<CompoundStmt>(Reverse))
      for (Stmt* S : RCS->body())
        addToCurrentBlock(S, forward);
    else
      addToCurrentBlock(Reverse, forward);

    // Since 'return' is not an assignment, add its error to _final_error
    // given it is not a DeclRefExpr.
    if (m_ErrorEstimationEnabled)
      errorEstHandler->EmitFinalErrorStmts(params, m_Function->getNumParams());
    Stmt* gradientBody = endBlock();
    m_Derivative->setBody(gradientBody);
    endScope(); // Function body scope
    m_Sema.PopFunctionScopeInfo();
    m_Sema.PopDeclContext();
    endScope(); // Function decl scope

    FunctionDecl* gradientOverloadFD = nullptr;
    if (shouldCreateOverload) {
      gradientOverloadFD =
          CreateGradientOverload(paramTypes, params, name, gradientFD);
    }

    return OverloadedDeclWithContext{result.first, result.second,
                                     gradientOverloadFD};
  }

  StmtDiff ReverseModeVisitor::VisitStmt(const Stmt* S) {
    diag(
        DiagnosticsEngine::Warning,
        S->getBeginLoc(),
        "attempted to differentiate unsupported statement, no changes applied");
    // Unknown stmt, just clone it.
    return StmtDiff(Clone(S));
  }

  StmtDiff ReverseModeVisitor::VisitCompoundStmt(const CompoundStmt* CS) {
    beginScope(Scope::DeclScope);
    beginBlock(forward);
    beginBlock(reverse);
    for (Stmt* S : CS->body()) {
      StmtDiff SDiff = DifferentiateSingleStmt(S);
      addToCurrentBlock(SDiff.getStmt(), forward);
      addToCurrentBlock(SDiff.getStmt_dx(), reverse);
      // In error estimation mode, if we have any residual statements
      // to be emitted into the forward or revese blocks, we should
      // emit them here. This is to maintain the correct order of
      // statements generated.
      if (m_ErrorEstimationEnabled) {
        errorEstHandler->EmitErrorEstimationStmts(forward);
        errorEstHandler->EmitErrorEstimationStmts(reverse);
      }
    }
    CompoundStmt* Forward = endBlock(forward);
    CompoundStmt* Reverse = endBlock(reverse);
    endScope();
    return StmtDiff(Forward, Reverse);
  }

  static Stmt* unwrapIfSingleStmt(Stmt* S) {
    if (!S)
      return nullptr;
    if (!isa<CompoundStmt>(S))
      return S;
    auto CS = cast<CompoundStmt>(S);
    if (CS->size() == 0)
      return nullptr;
    else if (CS->size() == 1)
      return CS->body_front();
    else
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
      cond = StoreAndRef(condExpr.getExpr(), forward, "_t",
                         /*forceDeclCreation=*/true);
      StmtDiff condPushPop = GlobalStoreAndRef(cond.getExpr(), "_cond");
      PushCond = condPushPop.getExpr();
      PopCond = condPushPop.getExpr_dx();
    } else
      cond = GlobalStoreAndRef(condExpr.getExpr(), "_cond");
    // Convert cond to boolean condition. We are modifying each Stmt in
    // StmtDiff.
    for (Stmt*& S : cond.getBothStmts())
      if (S)
        S = m_Sema
                .ActOnCondition(m_CurScope,
                                noLoc,
                                cast<Expr>(S),
                                Sema::ConditionKind::Boolean)
                .get()
                .second;

    // Create a block "around" if statement, e.g:
    // {
    //   ...
    //  if (...) {...}
    // }
    beginBlock(forward);
    beginBlock(reverse);
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
      } else {
        beginBlock(forward);
        StmtDiff BranchDiff = DifferentiateSingleStmt(Branch, /*dfdS=*/nullptr,
                                                      /*shouldEmit=*/false);
        addToCurrentBlock(BranchDiff.getStmt(), forward);
        // In error estimation, manually emit the code here instead of
        // DifferentiateSingleStmt to maintain correct order.
        if (m_ErrorEstimationEnabled) {
          errorEstHandler->EmitErrorEstimationStmts(forward);
        }
        Stmt* Forward = unwrapIfSingleStmt(endBlock(forward));
        Stmt* Reverse = unwrapIfSingleStmt(BranchDiff.getStmt_dx());
        return StmtDiff(Forward, Reverse);
      }
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
    addToCurrentBlock(Forward, forward);

    Expr* reverseCond = cond.getExpr_dx();
    if (isInsideLoop) {
      addToCurrentBlock(PushCond, forward);
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
    addToCurrentBlock(Reverse, reverse);
    CompoundStmt* ForwardBlock = endBlock(forward);
    CompoundStmt* ReverseBlock = endBlock(reverse);
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
              .ActOnCondition(m_CurScope,
                              noLoc,
                              cast<Expr>(S),
                              Sema::ConditionKind::Boolean)
              .get()
              .second;

    auto ifTrue = CO->getTrueExpr();
    auto ifFalse = CO->getFalseExpr();

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
      addToCurrentBlock(Forward, forward);
    if (Reverse)
      addToCurrentBlock(Reverse, reverse);

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
    beginBlock(forward);
    beginBlock(reverse);
    const Stmt* init = FS->getInit();
    StmtDiff initResult = init ? DifferentiateSingleStmt(init) : StmtDiff{};

    // Save the isInsideLoop value (we may be inside another loop).
    llvm::SaveAndRestore<bool> SaveIsInsideLoop(isInsideLoop);
    isInsideLoop = true;

    StmtDiff condVarRes;
    VarDecl* condVarClone = nullptr;
    if (FS->getConditionVariable()) {
      condVarRes = DifferentiateSingleStmt(FS->getConditionVariableDeclStmt());
      Decl* decl = cast<DeclStmt>(condVarRes.getStmt())->getSingleDecl();
      condVarClone = cast<VarDecl>(decl);
    }

    // FIXME: for now we assume that cond has no differentiable effects,
    // but it is not generally true, e.g. for (...; (x = y); ...)...
    StmtDiff cond;
    if (FS->getCond())
      cond = Visit(FS->getCond());
    auto IDRE = dyn_cast<DeclRefExpr>(FS->getInc());
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
    CompoundStmt* Additional = cast<CompoundStmt>(incDiff.getStmt());
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

    Stmt* Forward = new (m_Context) ForStmt(m_Context,
                                            initResult.getStmt(),
                                            cond.getExpr(),
                                            condVarClone,
                                            incResult,
                                            BodyDiff.getStmt(),
                                            noLoc,
                                            noLoc,
                                            noLoc);

    // Create a condition testing counter for being zero, and its decrement.
    // To match the number of iterations in the forward pass, the reverse loop
    // will look like: for(; Counter; Counter--) ...
    Expr*
        CounterCondition = loopCounter.getCounterConditionResult().get().second;
    Expr* CounterDecrement = loopCounter.getCounterDecrement();

    Stmt* ReverseBody = BodyDiff.getStmt_dx();
    Stmt* ReverseResult = unwrapIfSingleStmt(ReverseBody);
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
    addToCurrentBlock(Forward, forward);
    Forward = endBlock(forward);
    addToCurrentBlock(loopCounter.getPop(), reverse);
    addToCurrentBlock(Reverse, reverse);
    Reverse = endBlock(reverse);
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

  StmtDiff ReverseModeVisitor::VisitReturnStmt(const ReturnStmt* RS) {
    // Initially, df/df = 1.
    const Expr* value = RS->getRetValue();
    QualType type = value->getType();
    auto dfdf =
        ConstantFolder::synthesizeLiteral(m_Context.IntTy, m_Context, 1);
    ExprResult tmp = dfdf;
    dfdf = m_Sema
               .ImpCastExprToType(tmp.get(),
                                  type,
                                  m_Sema.PrepareScalarCast(tmp, type))
               .get();
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
    addToCurrentBlock(LS, reverse);
    for (Stmt* S : cast<CompoundStmt>(ReturnDiff.getStmt())->body())
      addToCurrentBlock(S, forward);
    // Since returned expression may have some side effects affecting reverse
    // computation (e.g. assignments), we also have to emit it to execute it.
    Expr* retDeclRefExpr = StoreAndRef(ExprDiff.getExpr(), forward,
                                       utils::ComputeEffectiveFnName(
                                           m_Function) +
                                           "_return",
                                       /*forceDeclCreation=*/true);
    // If the return expression is not a DeclRefExpression and is of type
    // float, we should add it to the error estimate because returns are
    // similiar to implicit assigns.
    if (m_ErrorEstimationEnabled) {
      errorEstHandler->SaveReturnExpr(ExprDiff.getExpr(),
                                      cast<DeclRefExpr>(retDeclRefExpr));
    }
    // Create goto to the label.
    return m_Sema.ActOnGotoStmt(noLoc, noLoc, LD).get();
  }

  StmtDiff ReverseModeVisitor::VisitParenExpr(const ParenExpr* PE) {
    StmtDiff subStmtDiff = Visit(PE->getSubExpr(), dfdx());
    return StmtDiff(BuildParens(subStmtDiff.getExpr()),
                    BuildParens(subStmtDiff.getExpr_dx()));
  }

  StmtDiff ReverseModeVisitor::VisitInitListExpr(const InitListExpr* ILE) {
    llvm::SmallVector<Expr*, 16> clonedExprs(ILE->getNumInits());
    for (unsigned i = 0, e = ILE->getNumInits(); i < e; i++) {
      Expr* I =
          ConstantFolder::synthesizeLiteral(m_Context.IntTy, m_Context, i);
      Expr* array_at_i = m_Sema
                             .ActOnArraySubscriptExpr(getCurrentScope(), dfdx(),
                                                      noLoc, I, noLoc)
                             .get();
      Expr* clonedEI = Visit(ILE->getInit(i), array_at_i).getExpr();
      clonedExprs[i] = clonedEI;
    }

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
    for (std::size_t i = 0; i < Indices.size(); i++) {
      StmtDiff IdxDiff = Visit(Indices[i]);
      StmtDiff IdxStored = GlobalStoreAndRef(IdxDiff.getExpr());
      if (isInsideLoop) {
        // Here we make sure that we are popping each time we push.
        // Since the max no of pushes = no. of array index expressions in the
        // loop.
        Expr* popExpr = IdxStored.getExpr_dx();
        VarDecl* popVal = BuildVarDecl(popExpr->getType(), "_t", popExpr,
                                       /*DirectInit=*/true);
        if (dfdx())
          addToCurrentBlock(BuildDeclStmt(popVal), reverse);
        else
          m_PopIdxValues.push_back(BuildDeclStmt(popVal));
        IdxStored = StmtDiff(IdxStored.getExpr(), BuildDeclRef(popVal));
      }
      clonedIndices[i] = IdxStored.getExpr();
      reverseIndices[i] = IdxStored.getExpr_dx();
    }
    auto cloned = BuildArraySubscript(BaseDiff.getExpr(), clonedIndices);

    Expr* target = BaseDiff.getExpr_dx();
    if (!target)
      return cloned;
    Expr* result = nullptr;
    if (isArrayOrPointerType(target->getType()))
      // Create the target[idx] expression.
      result = BuildArraySubscript(target, reverseIndices);
    else if (isCladArrayType(target->getType())) {
      result = m_Sema
                   .ActOnArraySubscriptExpr(getCurrentScope(), target,
                                            ASE->getExprLoc(),
                                            reverseIndices.back(), noLoc)
                   .get();
    } else
      result = target;
    // Create the (target += dfdx) statement.
    if (dfdx()) {
      auto add_assign = BuildOp(BO_AddAssign, result, dfdx());
      // Add it to the body statements.
      addToCurrentBlock(add_assign, reverse);
    }
    return StmtDiff(cloned, result);
  }

  StmtDiff ReverseModeVisitor::VisitDeclRefExpr(const DeclRefExpr* DRE) {
    DeclRefExpr* clonedDRE = nullptr;
    // Check if referenced Decl was "replaced" with another identifier inside
    // the derivative
    if (auto VD = dyn_cast<VarDecl>(DRE->getDecl())) {
      auto it = m_DeclReplacements.find(VD);
      if (it != std::end(m_DeclReplacements))
        clonedDRE = BuildDeclRef(it->second);
      else
        clonedDRE = cast<DeclRefExpr>(Clone(DRE));
      // If current context is different than the context of the original
      // declaration (e.g. we are inside lambda), rebuild the DeclRefExpr
      // with Sema::BuildDeclRefExpr. This is required in some cases, e.g.
      // Sema::BuildDeclRefExpr is responsible for adding captured fields
      // to the underlying struct of a lambda.
      if (clonedDRE->getDecl()->getDeclContext() != m_Sema.CurContext) {
        auto referencedDecl = cast<VarDecl>(clonedDRE->getDecl());
        clonedDRE = cast<DeclRefExpr>(BuildDeclRef(referencedDecl));
      }
    } else
      clonedDRE = cast<DeclRefExpr>(Clone(DRE));

    if (auto decl = dyn_cast<VarDecl>(clonedDRE->getDecl())) {
      if (isVectorValued) {
        if (m_VectorOutput.size() <= outputArrayCursor)
          return StmtDiff(clonedDRE);

        auto it = m_VectorOutput[outputArrayCursor].find(decl);
        if (it == std::end(m_VectorOutput[outputArrayCursor])) {
          // Is not an independent variable, ignored.
          return StmtDiff(clonedDRE);
        }
        // Create the (jacobianMatrix[idx] += dfdx) statement.
        if (dfdx()) {
          auto add_assign = BuildOp(BO_AddAssign, it->second, dfdx());
          // Add it to the body statements.
          addToCurrentBlock(add_assign, reverse);
        }
      } else {
        // Check DeclRefExpr is a reference to an independent variable.
        auto it = m_Variables.find(decl);
        if (it == std::end(m_Variables)) {
          // Is not an independent variable, ignored.
          return StmtDiff(clonedDRE);
        }
        // Create the (_d_param[idx] += dfdx) statement.
        if (dfdx()) {
          Expr* add_assign = BuildOp(BO_AddAssign, it->second, dfdx());
          // Add it to the body statements.
          addToCurrentBlock(add_assign, reverse);
        }
        return StmtDiff(clonedDRE, it->second);
      }
    }

    return StmtDiff(clonedDRE);
  }

  StmtDiff ReverseModeVisitor::VisitIntegerLiteral(const IntegerLiteral* IL) {
    return StmtDiff(Clone(IL));
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
    // If the function has no args then we assume that it is not related
    // to independent variables and does not contribute to gradient.
    if (!NArgs)
      return StmtDiff(Clone(CE));

    // Stores the call arguments for the function to be derived
    llvm::SmallVector<Expr*, 16> CallArgs{};
    // Stores the dx of the call arguments for the function to be derived
    llvm::SmallVector<Expr*, 16> CallArgDx{};
    // Stores the call arguments for the derived function
    llvm::SmallVector<Expr*, 16> DerivedCallArgs{};
    // Stores tape decl and pushes for multiarg numerically differentiated
    // calls.
    llvm::SmallVector<Stmt*, 16> NumericalDiffMultiArg{};
    // If the result does not depend on the result of the call, just clone
    // the call and visit arguments (since they may contain side-effects like
    // f(x = y))
    // FIXME: What about pass-by-reference calls?
    if (!dfdx()) {
      for (const Expr* Arg : CE->arguments()) {
        StmtDiff ArgDiff = Visit(Arg, dfdx());
        CallArgs.push_back(ArgDiff.getExpr());
        // If in error estimation, we want to store the errors in each input
        // variable to a call expression given it is reference type, hence
        // we should build an error expression here and store it to emit later.
        // FIXME: We need a derivative wrt each pass-by-reference input here.
        // if (m_ErrorEstimationEnabled) {
        //   errorEstHandler->EmitNestedFunctionParamError(...);
        // }
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
    std::size_t insertionPoint = getCurrentBlock(reverse).size();

    // Store the type to reduce call overhead that would occur if used in the
    // loop
    auto CEType = getNonConstType(CE->getType(), m_Context, m_Sema);
    for (const Expr* Arg : CE->arguments()) {
      // Create temporary variables corresponding to derivative of each
      // argument, so that they can be referred to when arguments is visited.
      // Variables will be initialized later after arguments is visited. This is
      // done to reduce cloning complexity and only clone once. The type is same
      // as the call expression as it is the type used to declare the _gradX
      // array
      Expr* dArg;
      if (isArrayOrPointerType(Arg->getType())) {
        Expr* nullptrLiteral = m_Sema.ActOnCXXNullPtrLiteral(noLoc).get();
        dArg = StoreAndRef(nullptrLiteral, GetCladArrayRefOfType(CEType),
                           reverse, "_r",
                           /*forceDeclCreation=*/true,
                           VarDecl::InitializationStyle::CallInit);
      } else {
        dArg = StoreAndRef(/*E=*/nullptr, CEType, reverse, "_r",
                           /*forceDeclCreation=*/true);
      }
      ArgResultDecls.push_back(
          cast<VarDecl>(cast<DeclRefExpr>(dArg)->getDecl()));
      // Visit using uninitialized reference.
      StmtDiff ArgDiff = Visit(Arg, dArg);

      // Save cloned arg in a "global" variable, so that it is accessible from
      // the reverse pass.
      CallArgDx.push_back(ArgDiff.getExpr_dx());
      ArgDiff = GlobalStoreAndRef(ArgDiff.getExpr());
      CallArgs.push_back(ArgDiff.getExpr());
      DerivedCallArgs.push_back(ArgDiff.getExpr_dx());
    }

    VarDecl* ResultDecl = nullptr;
    Expr* Result = nullptr;
    Expr* ResultExpr = nullptr;
    IdentifierInfo* ResultII = nullptr;
    Expr* OverloadedDerivedFn = nullptr;
    // If the function has a single arg, we look for a derivative w.r.t. to
    // this arg (it is unlikely that we need gradient of a one-dimensional'
    // function).
    bool asGrad = true;
    if (NArgs == 1) {
      IdentifierInfo* II =
          &m_Context.Idents.get(FD->getNameAsString() + "_pushforward");
      // Try to find it in builtin derivatives
      DeclarationName name(II);
      DeclarationNameInfo DNInfo(name, noLoc);
      auto pushforwardCallArgs = DerivedCallArgs;
      pushforwardCallArgs.push_back(ConstantFolder::synthesizeLiteral(
          DerivedCallArgs.front()->getType(), m_Context, 1));
      OverloadedDerivedFn =
          m_Builder.BuildCallToCustomDerivativeOrNumericalDiff(
              DNInfo, pushforwardCallArgs, getCurrentScope(),
              const_cast<DeclContext*>(FD->getDeclContext()));
      if (OverloadedDerivedFn)
        asGrad = false;
    }
    // Store all the derived call output args (if any)
    llvm::SmallVector<Expr*, 16> DerivedCallOutputArgs{};

    // If it has more args or f_darg0 was not found, we look for its gradient.
    if (!OverloadedDerivedFn) {
      IdentifierInfo* II =
          &m_Context.Idents.get(FD->getNameAsString() + funcPostfix());
      DeclarationName name(II);
      DeclarationNameInfo DNInfo(name, noLoc);

      unsigned size_type_bits = m_Context.getIntWidth(m_Context.getSizeType());

      size_t idx = 0;

      auto ArrayDiffArgType = GetCladArrayOfType(CEType.getCanonicalType());
      for (auto arg : CallArgDx) {
        ResultDecl = nullptr;
        Result = nullptr;
        ResultExpr = nullptr;
        ResultII = CreateUniqueIdentifier(funcPostfix());
        if (arg && (isCladArrayType(arg->getType()) ||
                    isArrayOrPointerType(arg->getType()))) {
          Expr* SizeE;
          if (auto CAT = dyn_cast<ConstantArrayType>(arg->getType())) {
            SizeE = ConstantFolder::synthesizeLiteral(
                m_Context.getSizeType(), m_Context,
                CAT->getSize().getZExtValue());
          } else {
            assert(isCladArrayType(arg->getType()) &&
                   "Size couldn't be determined. Please make sure the diff "
                   "variables are either constant sized arrays or of type "
                   "clad::array_ref");
            SizeE = BuildArrayRefSizeExpr(arg);
          }

          // Declare: clad::array_ref<ArrayDiffArgType> _gradX(arrLen);
          ResultDecl = BuildVarDecl(ArrayDiffArgType, ResultII, SizeE,
                                    /*DirectInit=*/false,
                                    /*TSI=*/nullptr,
                                    VarDecl::InitializationStyle::CallInit);
          Result = BuildDeclRef(ResultDecl);
          ResultExpr = Result;
          Expr* E = BuildOp(BO_MulAssign, Result, dfdx());
          // Visit each arg with df/dargi = df/dxi * Result.
          PerformImplicitConversionAndAssign(ArgResultDecls[idx], E);
        } else {
          // Declare: diffArgType _grad = 0;
          ResultDecl = BuildVarDecl(
              CEType, ResultII,
              ConstantFolder::synthesizeLiteral(CEType, m_Context, 0));
          // Pass the address of the declared variable
          Result = BuildDeclRef(ResultDecl);
          ResultExpr = BuildOp(UO_AddrOf, Result, m_Function->getLocation());
          // Visit each arg with df/dargi = df/dxi * Result.
          PerformImplicitConversionAndAssign(ArgResultDecls[idx],
                                             BuildOp(BO_Mul, dfdx(), Result));
        }
        DerivedCallOutputArgs.push_back(ResultExpr);
        ArgDeclStmts.push_back(BuildDeclStmt(ResultDecl));
        idx++;
      }

      DerivedCallArgs.insert(DerivedCallArgs.end(),
                             DerivedCallOutputArgs.begin(),
                             DerivedCallOutputArgs.end());

      // Try to find it in builtin derivatives
      OverloadedDerivedFn =
          m_Builder.BuildCallToCustomDerivativeOrNumericalDiff(
              DNInfo, DerivedCallArgs, getCurrentScope(),
              /*OriginalFnDC=*/nullptr);
    }
    // Derivative was not found, check if it is a recursive call
    if (!OverloadedDerivedFn) {
      if (FD == m_Function) {
        // Recursive call.
        auto selfRef =
            m_Sema
                .BuildDeclarationNameExpr(CXXScopeSpec(),
                                          m_Derivative->getNameInfo(),
                                          m_Derivative)
                .get();

        OverloadedDerivedFn =
            m_Sema
                .ActOnCallExpr(getCurrentScope(), selfRef, noLoc,
                               llvm::MutableArrayRef<Expr*>(DerivedCallArgs),
                               noLoc)
                .get();
      } else {
        // Overloaded derivative was not found, request the CladPlugin to
        // derive the called function.
        DiffRequest request{};
        request.Function = FD;
        request.BaseFunctionName = FD->getNameAsString();
        request.Mode = DiffMode::reverse;
        // Silence diag outputs in nested derivation process.
        request.VerboseDiags = false;

        FunctionDecl* derivedFD =
            plugin::ProcessDiffRequest(m_CladPlugin, request);
        // Clad failed to derive it.
        if (!derivedFD) {
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
            OverloadedDerivedFn = GetMultiArgCentralDiffCall(
                fnCallee, CEType.getCanonicalType(), CE->getNumArgs(),
                NumericalDiffMultiArg, DerivedCallArgs, DerivedCallOutputArgs);
          }
          CallExprDiffDiagnostics(FD->getNameAsString(), CE->getBeginLoc(),
                                  OverloadedDerivedFn);
          if (!OverloadedDerivedFn) {
            auto& block = getCurrentBlock(reverse);
            block.insert(block.begin(), ArgDeclStmts.begin(),
                         ArgDeclStmts.end());
            return StmtDiff(Clone(CE));
          }
        } else {
          OverloadedDerivedFn = m_Sema
                                    .ActOnCallExpr(getCurrentScope(),
                                                   BuildDeclRef(derivedFD),
                                                   noLoc,
                                                   llvm::MutableArrayRef<Expr*>(
                                                       DerivedCallArgs),
                                                   noLoc)
                                    .get();
        }    
      }
    }

    if (OverloadedDerivedFn) {
      // Derivative was found.
      FunctionDecl* fnDecl = dyn_cast<CallExpr>(OverloadedDerivedFn)
                                 ->getDirectCallee();
      if (!asGrad) {
        // If the derivative is called through _darg0 instead of _grad.
        Expr* d = BuildOp(BO_Mul, dfdx(), OverloadedDerivedFn);

        PerformImplicitConversionAndAssign(ArgResultDecls[0], d);
        // If in error estimation, build the statement for the error
        // in the input prameters (if they are reference types) to call and
        // save to emit them later.
        if (m_ErrorEstimationEnabled) {
          errorEstHandler->EmitNestedFunctionParamError(fnDecl, CallArgs,
                                                        ArgResultDecls,
                                                        /*numArgs=*/1);
        }
      } else {
        // Put Result array declaration in the function body.
        // Call the gradient, passing Result as the last Arg.
        auto& block = getCurrentBlock(reverse);
        auto it = std::begin(block) + insertionPoint;

        // Insert the _gradX declaration statements
        it = block.insert(it, ArgDeclStmts.begin(), ArgDeclStmts.end());
        it += ArgDeclStmts.size();
        it = block.insert(it, NumericalDiffMultiArg.begin(),
                          NumericalDiffMultiArg.end());
        it += NumericalDiffMultiArg.size();
        // Insert the CallExpr to the derived function
        block.insert(it, OverloadedDerivedFn);
        // If in error estimation, build the statement for the error
        // in the input prameters (if of reference type) to call and save to
        // emit them later.
        if (m_ErrorEstimationEnabled) {
          errorEstHandler->EmitNestedFunctionParamError(
              fnDecl, CallArgs, ArgResultDecls, CE->getNumArgs());
        }
      }
    }

    // Re-clone function arguments again, since they are required at 2 places:
    // call to gradient and call to original function.
    // At this point, each arg is either a simple expression or a reference
    // to a temporary variable. Therefore cloning it has constant complexity.
    std::transform(std::begin(CallArgs),
                   std::end(CallArgs),
                   std::begin(CallArgs),
                   [this](Expr* E) { return Clone(E); });
    // Recreate the original call expression.
    Expr* call = m_Sema
                     .ActOnCallExpr(getCurrentScope(),
                                    Clone(CE->getCallee()),
                                    noLoc,
                                    llvm::MutableArrayRef<Expr*>(CallArgs),
                                    noLoc)
                     .get();
    return StmtDiff(call);
  }

  StmtDiff ReverseModeVisitor::VisitUnaryOperator(const UnaryOperator* UnOp) {
    auto opCode = UnOp->getOpcode();
    StmtDiff diff{};
    // If it is a post-increment/decrement operator, its result is a reference
    // and we should return it.
    Expr* ResultRef = nullptr;
    if (opCode == UO_Plus)
      // xi = +xj
      // dxi/dxj = +1.0
      // df/dxj += df/dxi * dxi/dxj = df/dxi
      diff = Visit(UnOp->getSubExpr(), dfdx());
    else if (opCode == UO_Minus) {
      // xi = -xj
      // dxi/dxj = -1.0
      // df/dxj += df/dxi * dxi/dxj = -df/dxi
      auto d = BuildOp(UO_Minus, dfdx());
      diff = Visit(UnOp->getSubExpr(), d);
    } else if (opCode == UO_PostInc || opCode == UO_PostDec) {
      diff = Visit(UnOp->getSubExpr(), dfdx());
      ResultRef = diff.getExpr_dx();
      if (m_ErrorEstimationEnabled)
        errorEstHandler->EmitUnaryOpErrorStmts(diff, isInsideLoop);
    } else if (opCode == UO_PreInc || opCode == UO_PreDec) {
      diff = Visit(UnOp->getSubExpr(), dfdx());
    } else {
      // We should not output any warning on visiting boolean conditions
      // FIXME: We should support boolean differentiation or ignore it
      // completely
      if (opCode != UO_LNot)
        unsupportedOpWarn(UnOp->getEndLoc());

      Expr* subExpr = UnOp->getSubExpr();
      if (isa<DeclRefExpr>(subExpr))
        diff = Visit(subExpr);
      else
        diff = StmtDiff(subExpr);
    }
    Expr* op = BuildOp(opCode, diff.getExpr());
    return StmtDiff(op, ResultRef);
  }

  StmtDiff
  ReverseModeVisitor::VisitBinaryOperator(const BinaryOperator* BinOp) {
    auto opCode = BinOp->getOpcode();
    StmtDiff Ldiff{};
    StmtDiff Rdiff{};
    auto L = BinOp->getLHS();
    auto R = BinOp->getRHS();
    // If it is an assignment operator, its result is a reference to LHS and
    // we should return it.
    Expr* ResultRef = nullptr;

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
      auto dr = BuildOp(UO_Minus, dfdx());
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
      auto RDelayed = DelayedGlobalStoreAndRef(R);
      StmtDiff RResult = RDelayed.Result;
      Expr* dl = nullptr;
      if (dfdx()) {
        dl = BuildOp(BO_Mul, dfdx(), RResult.getExpr_dx());
        dl = StoreAndRef(dl, reverse);
      }
      Ldiff = Visit(L, dl);
      // dxi/xr = xl
      // df/dxr += df/dxi * dxi/xr = df/dxi * xl
      // Store left multiplier and assign it with L.
      Expr* LStored = Ldiff.getExpr();
      // RDelayed.isConstant == true implies that R is a constant expression,
      // therefore we can skip visiting it.
      if (!RDelayed.isConstant) {
        Expr* dr = nullptr;
        if (dfdx()) {
          StmtDiff LResult = GlobalStoreAndRef(LStored);
          LStored = LResult.getExpr();
          dr = BuildOp(BO_Mul, LResult.getExpr_dx(), dfdx());
          dr = StoreAndRef(dr, reverse);
        }
        Rdiff = Visit(R, dr);
        // Assign right multiplier's variable with R.
        RDelayed.Finalize(Rdiff.getExpr());
      }
      std::tie(Ldiff, Rdiff) = std::make_pair(LStored, RResult.getExpr());
    } else if (opCode == BO_Div) {
      // xi = xl / xr
      // dxi/xl = 1 / xr
      // df/dxl += df/dxi * dxi/xl = df/dxi * (1/xr)
      auto RDelayed = DelayedGlobalStoreAndRef(R);
      StmtDiff RResult = RDelayed.Result;
      Expr* RStored = StoreAndRef(RResult.getExpr_dx(), reverse);
      Expr* dl = nullptr;
      if (dfdx()) {
        dl = BuildOp(BO_Div, dfdx(), RStored);
        dl = StoreAndRef(dl, reverse);
      }
      Ldiff = Visit(L, dl);
      // dxi/xr = -xl / (xr * xr)
      // df/dxl += df/dxi * dxi/xr = df/dxi * (-xl /(xr * xr))
      // Wrap R * R in parentheses: (R * R). otherwise code like 1 / R * R is
      // produced instead of 1 / (R * R).
      Expr* LStored = Ldiff.getExpr();
      if (!RDelayed.isConstant) {
        Expr* dr = nullptr;
        if (dfdx()) {
          StmtDiff LResult = GlobalStoreAndRef(LStored);
          LStored = LResult.getExpr();
          Expr* RxR = BuildParens(BuildOp(BO_Mul, RStored, RStored));
          dr = BuildOp(BO_Mul,
                       dfdx(),
                       BuildOp(UO_Minus,
                               BuildOp(BO_Div, LResult.getExpr_dx(), RxR)));
          dr = StoreAndRef(dr, reverse);
        }
        Rdiff = Visit(R, dr);
        RDelayed.Finalize(Rdiff.getExpr());
      }
      std::tie(Ldiff, Rdiff) = std::make_pair(LStored, RResult.getExpr());
    } else if (BinOp->isAssignmentOp()) {
      if (L->isModifiableLvalue(m_Context) != Expr::MLV_Valid) {
        diag(DiagnosticsEngine::Warning,
             BinOp->getEndLoc(),
             "derivative of an assignment attempts to assign to unassignable "
             "expr, assignment ignored");
        auto LDRE = dyn_cast<DeclRefExpr>(L);
        auto RDRE = dyn_cast<DeclRefExpr>(R);

        if (!LDRE && !RDRE)
          return Clone(BinOp);
        Expr* LExpr = LDRE ? Visit(L).getExpr() : L;
        Expr* RExpr = RDRE ? Visit(R).getExpr() : R;

        return BuildOp(opCode, LExpr, RExpr);
      }

      if (auto ASE = dyn_cast<ArraySubscriptExpr>(L)) {
        auto DRE = dyn_cast<DeclRefExpr>(ASE->getBase()->IgnoreImplicit());
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
              auto idx =
                  IntegerLiteral::Create(m_Context, idxValue, size_type, noLoc);
              // Create the jacobianMatrix[idx] expression.
              auto result_at_i =
                  m_Sema
                      .CreateBuiltinArraySubscriptExpr(m_Result, noLoc, idx,
                                                       noLoc)
                      .get();
              temp_m_Variables[m_IndependentVars[i]] = result_at_i;
            }
            m_VectorOutput.push_back(temp_m_Variables);
          }

          auto dfdf =
              ConstantFolder::synthesizeLiteral(m_Context.IntTy, m_Context, 1);
          ExprResult tmp = dfdf;
          dfdf = m_Sema
                     .ImpCastExprToType(tmp.get(),
                                        type,
                                        m_Sema.PrepareScalarCast(tmp, type))
                     .get();
          auto ReturnResult = DifferentiateSingleExpr(R, dfdf);
          StmtDiff ReturnDiff = ReturnResult.first;
          Stmt* Reverse = ReturnDiff.getStmt_dx();
          addToCurrentBlock(Reverse, reverse);
          for (Stmt* S : cast<CompoundStmt>(ReturnDiff.getStmt())->body())
            addToCurrentBlock(S, forward);
        }
      }

      // Visit LHS, but delay emission of its derivative statements, save them
      // in Lblock
      beginBlock(reverse);
      Ldiff = Visit(L, dfdx());
      auto Lblock = endBlock(reverse);
      Expr* LCloned = Ldiff.getExpr();
      // For x, AssignedDiff is _d_x, for x[i] its _d_x[i], for reference exprs
      // like (x = y) it propagates recursively, so _d_x is also returned.
      Expr* AssignedDiff = Ldiff.getExpr_dx();
      if (!AssignedDiff) {
        // If either LHS or RHS is a declaration reference, visit it to avoid
        // naming collision
        auto LDRE = dyn_cast<DeclRefExpr>(L);
        auto RDRE = dyn_cast<DeclRefExpr>(R);

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
      if (dfdx() && Lblock->size()) {
        addToCurrentBlock(*Lblock_begin, reverse);
        Lblock_begin = std::next(Lblock_begin);
      }
      while(!m_PopIdxValues.empty())
        addToCurrentBlock(m_PopIdxValues.pop_back_val(), reverse);
      Expr* deltaVar = nullptr;
      if (m_ErrorEstimationEnabled)
        deltaVar = errorEstHandler->RegisterBinaryOpLHS(LCloned, R,
                                                        /*isAssign=*/opCode ==
                                                            BO_Assign);
      // Save old value for the derivative of LHS, to avoid problems with cases
      // like x = x.
      auto oldValue = StoreAndRef(AssignedDiff, reverse, "_r_d",
                                  /*forceDeclCreation=*/true);
      if (opCode == BO_Assign) {
        Rdiff = Visit(R, oldValue);
      } else if (opCode == BO_AddAssign) {
        addToCurrentBlock(BuildOp(BO_AddAssign, AssignedDiff, oldValue),
                          reverse);
        Rdiff = Visit(R, oldValue);
      } else if (opCode == BO_SubAssign) {
        addToCurrentBlock(BuildOp(BO_AddAssign, AssignedDiff, oldValue),
                          reverse);
        Rdiff = Visit(R, BuildOp(UO_Minus, oldValue));
      } else if (opCode == BO_MulAssign) {
        auto RDelayed = DelayedGlobalStoreAndRef(R);
        StmtDiff RResult = RDelayed.Result;
        addToCurrentBlock(
            BuildOp(BO_AddAssign,
                    AssignedDiff,
                    BuildOp(BO_Mul, oldValue, RResult.getExpr_dx())),
            reverse);
        Expr* LRef = LCloned;
        if (!RDelayed.isConstant) {
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
          if (LCloned->HasSideEffects(m_Context)) {
            auto RefType = getNonConstType(L->getType(), m_Context, m_Sema);
            LRef = StoreAndRef(LCloned, RefType, forward, "_ref",
                               /*forceDeclCreation=*/true);
          }
          StmtDiff LResult = GlobalStoreAndRef(LRef);
          if (isInsideLoop)
            addToCurrentBlock(LResult.getExpr(), forward);
          Expr* dr = BuildOp(BO_Mul, LResult.getExpr_dx(), oldValue);
          dr = StoreAndRef(dr, reverse);
          Rdiff = Visit(R, dr);
          RDelayed.Finalize(Rdiff.getExpr());
        }
        std::tie(Ldiff, Rdiff) = std::make_pair(LRef, RResult.getExpr());
      } else if (opCode == BO_DivAssign) {
        auto RDelayed = DelayedGlobalStoreAndRef(R);
        StmtDiff RResult = RDelayed.Result;
        Expr* RStored = StoreAndRef(RResult.getExpr_dx(), reverse);
        addToCurrentBlock(BuildOp(BO_AddAssign,
                                  AssignedDiff,
                                  BuildOp(BO_Div, oldValue, RStored)),
                          reverse);
        Expr* LRef = LCloned;
        if (!RDelayed.isConstant) {
          if (LCloned->HasSideEffects(m_Context)) {
            QualType RefType = m_Context.getLValueReferenceType(
                getNonConstType(L->getType(), m_Context, m_Sema));
            LRef = StoreAndRef(LCloned, RefType, forward, "_ref",
                               /*forceDeclCreation=*/true);
          }
          StmtDiff LResult = GlobalStoreAndRef(LRef);
          if (isInsideLoop)
            addToCurrentBlock(LResult.getExpr(), forward);
          Expr* RxR = BuildParens(BuildOp(BO_Mul, RStored, RStored));
          Expr* dr = BuildOp(
              BO_Mul,
              oldValue,
              BuildOp(UO_Minus, BuildOp(BO_Div, LResult.getExpr_dx(), RxR)));
          dr = StoreAndRef(dr, reverse);
          Rdiff = Visit(R, dr);
          RDelayed.Finalize(Rdiff.getExpr());
        }
        std::tie(Ldiff, Rdiff) = std::make_pair(LRef, RResult.getExpr());
      } else
        llvm_unreachable("unknown assignment opCode");
      // Now, we should emit the delta for LHS if it met all the
      // requirements previously.
      if (m_ErrorEstimationEnabled)
        errorEstHandler->EmitBinaryOpErrorStmts(LCloned, oldValue, deltaVar,
                                                isInsideLoop);
      // Update the derivative.
      addToCurrentBlock(BuildOp(BO_SubAssign, AssignedDiff, oldValue), reverse);
      // Output statements from Visit(L).
      for (auto it = Lblock_begin; it != Lblock_end; ++it)
        addToCurrentBlock(*it, reverse);
    } else if (opCode == BO_Comma) {
      auto zero =
          ConstantFolder::synthesizeLiteral(m_Context.IntTy, m_Context, 0);
      Ldiff = Visit(L, zero);
      Rdiff = Visit(R, dfdx());
      ResultRef = Ldiff.getExpr();
    } else {
      // We should not output any warning on visiting boolean conditions
      // FIXME: We should support boolean differentiation or ignore it
      // completely
      if (!BinOp->isComparisonOp() && !BinOp->isLogicalOp())
        unsupportedOpWarn(BinOp->getEndLoc());

      // If either LHS or RHS is a declaration reference, visit it to avoid
      // naming collision
      auto LDRE = dyn_cast<DeclRefExpr>(L);
      auto RDRE = dyn_cast<DeclRefExpr>(R);

      if (!LDRE && !RDRE)
        return Clone(BinOp);

      Expr* LExpr = LDRE ? Visit(L).getExpr() : L;
      Expr* RExpr = RDRE ? Visit(R).getExpr() : R;

      return BuildOp(opCode, LExpr, RExpr);
    }
    Expr* op = BuildOp(opCode, Ldiff.getExpr(), Rdiff.getExpr());
    return StmtDiff(op, ResultRef);
  }

  VarDeclDiff ReverseModeVisitor::DifferentiateVarDecl(const VarDecl* VD) {
    StmtDiff initDiff;
    Expr* VDDerivedInit = nullptr;
    auto VDDerivedType = getNonConstType(VD->getType(), m_Context, m_Sema);
    bool isVDRefType = VD->getType()->isReferenceType();
    VarDecl* VDDerived = nullptr;
    // If VD is a reference to a local variable, then the initial value is set
    // to the derived variable of the corresponding local variable.
    // If VD is a reference to a non-local variable (global variable, struct
    // member etc), then no derived variable is available, thus `VDDerived` 
    // does not need to reference any variable, consequentially the 
    // `VDDerivedType` is the corresponding non-reference type and the initial
    // value is set to 0. 
    // Otherwise, for non-reference types, the initial value is set to 0.
    if (isVDRefType) {
      initDiff = Visit(VD->getInit());
      VDDerivedInit = initDiff.getExpr_dx();
      if (!VDDerivedInit) {
        VDDerivedType = VDDerivedType.getNonReferenceType();
        VDDerivedInit = getZeroInit(VDDerivedType);
      }
      VDDerived = BuildVarDecl(VDDerivedType, "_d_" + VD->getNameAsString(),
                               VDDerivedInit);
    } else if (auto VDCAT = dyn_cast<ConstantArrayType>(VD->getType())) {
      VDDerivedType =
          GetCladArrayOfType(QualType(VDCAT->getPointeeOrArrayElementType(),
                                      VDCAT->getIndexTypeCVRQualifiers()));
      VDDerivedInit = ConstantFolder::synthesizeLiteral(
          m_Context.getSizeType(), m_Context, VDCAT->getSize().getZExtValue());
      VDDerived = BuildVarDecl(VDDerivedType, "_d_" + VD->getNameAsString(),
                               VDDerivedInit, false, nullptr,
                               clang::VarDecl::InitializationStyle::CallInit);
    } else {
      VDDerivedInit = getZeroInit(VD->getType());
      VDDerived = BuildVarDecl(VDDerivedType, "_d_" + VD->getNameAsString(),
                               VDDerivedInit);
    }

    // If `VD` is a reference to a local variable, then it is already
    // differentiated and should not be differentiated again.
    // If `VD` is a reference to a non-local variable then also there's no
    // need to call `Visit` since non-local variables are not differentiated.
    if (!isVDRefType) {
      initDiff = VD->getInit() ? Visit(VD->getInit(), BuildDeclRef(VDDerived))
                               : StmtDiff{};

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
        addToCurrentBlock(assignToZero, reverse);
      }
    }
    VarDecl* VDClone = BuildVarDecl(VD->getType(),
                                    VD->getNameAsString(),
                                    initDiff.getExpr(),
                                    VD->isDirectInit());
    m_Variables.emplace(VDClone, BuildDeclRef(VDDerived));
    return VarDeclDiff(VDClone, VDDerived);
  }

  StmtDiff
  ReverseModeVisitor::DifferentiateSingleStmt(const Stmt* S, Expr* dfdS,
                                              bool shouldEmit /*=true*/) {
    beginBlock(reverse);
    StmtDiff SDiff = Visit(S, dfdS);
    // We might have some expressions to emit, so do that here.
    if (m_ErrorEstimationEnabled && shouldEmit)
      errorEstHandler->EmitErrorEstimationStmts(reverse);
    addToCurrentBlock(SDiff.getStmt_dx(), reverse);
    CompoundStmt* RCS = endBlock(reverse);
    std::reverse(RCS->body_begin(), RCS->body_end());
    Stmt* ReverseResult = unwrapIfSingleStmt(RCS);
    return StmtDiff(SDiff.getStmt(), ReverseResult);
  }

  std::pair<StmtDiff, StmtDiff>
  ReverseModeVisitor::DifferentiateSingleExpr(const Expr* E, Expr* dfdE) {
    beginBlock(forward);
    beginBlock(reverse);
    StmtDiff EDiff = Visit(E, dfdE);
    // We might have some expressions to emit, so do that here.
    if (m_ErrorEstimationEnabled)
      errorEstHandler->EmitErrorEstimationStmts(reverse);
    CompoundStmt* RCS = endBlock(reverse);
    Stmt* ForwardResult = endBlock(forward);
    std::reverse(RCS->body_begin(), RCS->body_end());
    Stmt* ReverseResult = unwrapIfSingleStmt(RCS);
    return {StmtDiff(ForwardResult, ReverseResult), EDiff};
  }

  StmtDiff ReverseModeVisitor::VisitDeclStmt(const DeclStmt* DS) {
    llvm::SmallVector<Decl*, 4> decls;
    llvm::SmallVector<Decl*, 4> declsDiff;
    // For each variable declaration v, create another declaration _d_v to
    // store derivatives for potential reassignments. E.g.
    // double y = x;
    // ->
    // double _d_y = _d_x; double y = x;
    for (auto D : DS->decls()) {
      if (auto VD = dyn_cast<VarDecl>(D)) {
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
        //   double _d_y = x; // copied from original funcion, collides with
        //   _d_y
        // }
        if (VDDiff.getDecl()->getDeclName() != VD->getDeclName())
          m_DeclReplacements[VD] = VDDiff.getDecl();
        decls.push_back(VDDiff.getDecl());
        declsDiff.push_back(VDDiff.getDecl_dx());
      } else {
        diag(DiagnosticsEngine::Warning,
             D->getEndLoc(),
             "Unsupported declaration");
      }
    }

    Stmt* DSClone = BuildDeclStmt(decls);
    Stmt* DSDiff = BuildDeclStmt(declsDiff);
    addToBlock(DSDiff, m_Globals);

    // For all dependent variables, we register them for estimation
    // here.
    if (m_ErrorEstimationEnabled) {
      for (size_t i = 0; i < decls.size(); i++) {
        VarDeclDiff VDDiff(static_cast<VarDecl*>(decls[0]),
                           static_cast<VarDecl*>(declsDiff[0]));
        errorEstHandler->EmitDeclErrorStmts(VDDiff, isInsideLoop);
      }
    }
    return StmtDiff(DSClone);
  }

  StmtDiff
  ReverseModeVisitor::VisitImplicitCastExpr(const ImplicitCastExpr* ICE) {
    StmtDiff subExprDiff = Visit(ICE->getSubExpr(), dfdx());
    // Casts should be handled automatically when the result is used by
    // Sema::ActOn.../Build...
    return StmtDiff(subExprDiff.getExpr(), subExprDiff.getExpr_dx());
  }

  StmtDiff ReverseModeVisitor::VisitMemberExpr(const MemberExpr* ME) {
    // We do not treat struct members as independent variables, so they are not
    // differentiated.
    return StmtDiff(Clone(ME));
  }

  StmtDiff
  ReverseModeVisitor::VisitExprWithCleanups(const ExprWithCleanups* EWC) {
    StmtDiff subExprDiff = Visit(EWC->getSubExpr(), dfdx());
    // FIXME: We are unable to create cleanup objects currently, this can be
    // potentially problematic
    return StmtDiff(subExprDiff.getExpr(), subExprDiff.getExpr_dx());
  }

  bool ReverseModeVisitor::UsefulToStoreGlobal(Expr* E) {
    if (isInsideLoop)
      return !E->isEvaluatable(m_Context, Expr::SE_NoSideEffects);
    if (!E)
      return false;
    // Use stricter policy when inside loops: IsEvaluatable is also true for
    // arithmetical expressions consisting of constants, e.g. (1 + 2)*3. This
    // chech is more expensive, but it doesn't make sense to push such constants
    // into stack.
    if (isInsideLoop)
      return !E->isEvaluatable(m_Context, Expr::SE_NoSideEffects);
    Expr* B = E->IgnoreParenImpCasts();
    // FIXME: find a more general way to determine that or add more options.
    if (isa<FloatingLiteral>(B) || isa<IntegerLiteral>(B))
      return false;
    if (isa<UnaryOperator>(B)) {
      auto UO = cast<UnaryOperator>(B);
      auto OpKind = UO->getOpcode();
      if (OpKind == UO_Plus || OpKind == UO_Minus)
        return UsefulToStoreGlobal(UO->getSubExpr());
      return true;
    }
    return true;
  }

  VarDecl* ReverseModeVisitor::GlobalStoreImpl(QualType Type,
                                               llvm::StringRef prefix,
                                               Expr* init) {
    // Create identifier before going to topmost scope
    // to let Sema::LookupName see the whole scope.
    auto identifier = CreateUniqueIdentifier(prefix);
    // Save current scope and temporarily go to topmost function scope.
    llvm::SaveAndRestore<Scope*> SaveScope(m_CurScope);
    assert(m_DerivativeFnScope && "must be set");
    m_CurScope = m_DerivativeFnScope;

    VarDecl* Var = nullptr;
    if (auto CAT = dyn_cast<ConstantArrayType>(Type)) {
      Type = GetCladArrayOfType(QualType(CAT->getPointeeOrArrayElementType(),
                                         Type.getCVRQualifiers()));
      init = ConstantFolder::synthesizeLiteral(
          m_Context.getSizeType(), m_Context, CAT->getSize().getZExtValue());
      Var = BuildVarDecl(Type, identifier, init, false, nullptr,
                         clang::VarDecl::InitializationStyle::CallInit);
    } else {
      Var = BuildVarDecl(Type, identifier, init);
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

    if (isInsideLoop) {
      auto CladTape = MakeCladTapeFor(E);
      Expr* Push = CladTape.Push;
      Expr* Pop = CladTape.Pop;
      return {Push, Pop};
    } else {
      Expr* Ref = BuildDeclRef(GlobalStoreImpl(Type, prefix));
      if (E) {
        Expr* Set = BuildOp(BO_Assign, Ref, E);
        addToCurrentBlock(Set, forward);
      }
      return {Ref, Ref};
    }
  }

  StmtDiff ReverseModeVisitor::GlobalStoreAndRef(Expr* E,
                                                 llvm::StringRef prefix,
                                                 bool force) {
    assert(E && "cannot infer type");
    return GlobalStoreAndRef(
        E, getNonConstType(E->getType(), m_Context, m_Sema), prefix, force);
  }

  void ReverseModeVisitor::DelayedStoreResult::Finalize(Expr* New) {
    if (isConstant)
      return;
    if (isInsideLoop) {
      auto Push = cast<CallExpr>(Result.getExpr());
      unsigned lastArg = Push->getNumArgs() - 1;
      Push->setArg(lastArg, V.m_Sema.DefaultLvalueConversion(New).get());
    } else {
      V.addToCurrentBlock(V.BuildOp(BO_Assign, Result.getExpr(), New),
                          ReverseModeVisitor::forward);
    }
  }

  ReverseModeVisitor::DelayedStoreResult
  ReverseModeVisitor::DelayedGlobalStoreAndRef(Expr* E,
                                               llvm::StringRef prefix) {
    assert(E && "must be provided");
    if (!UsefulToStoreGlobal(E)) {
      Expr* Cloned = Clone(E);
      return DelayedStoreResult{*this,
                                StmtDiff{Cloned, Cloned},
                                /*isConstant*/ true,
                                /*isInsideLoop*/ false};
    }
    if (isInsideLoop) {
      Expr* dummy = E;
      auto CladTape = MakeCladTapeFor(dummy);
      Expr* Push = CladTape.Push;
      Expr* Pop = CladTape.Pop;
      return DelayedStoreResult{*this,
                                StmtDiff{Push, Pop},
                                /*isConstant*/ false,
                                /*isInsideLoop*/ true};
    } else {
      Expr* Ref = BuildDeclRef(GlobalStoreImpl(
          getNonConstType(E->getType(), m_Context, m_Sema), prefix));
      // Return reference to the declaration instead of original expression.
      return DelayedStoreResult{*this,
                                StmtDiff{Ref, Ref},
                                /*isConstant*/ false,
                                /*isInsideLoop*/ false};
    }
  }

  ReverseModeVisitor::LoopCounter::LoopCounter(ReverseModeVisitor& RMV)
      : m_RMV(RMV) {
    ASTContext& C = m_RMV.m_Context;
    if (RMV.isInsideLoop) {
      auto zero = ConstantFolder::synthesizeLiteral(C.getSizeType(), C,
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
      Decl* condVarClone = cast<DeclStmt>(condVarRes.getStmt())
                               ->getSingleDecl();
      condResult = m_Sema.ActOnConditionVariable(condVarClone, noLoc,
                                                 Sema::ConditionKind::Boolean);
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
      beginBlock(reverse);
      addToCurrentBlock(loopCounter.getPop(), reverse);
      addToCurrentBlock(reverseWS, reverse);
      reverseBlock = endBlock(reverse);
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
                                       /*RCondRParen=*/noLoc)
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
      beginBlock(reverse);
      addToCurrentBlock(loopCounter.getPop(), reverse);
      addToCurrentBlock(reverseDS, reverse);
      reverseBlock = endBlock(reverse);
    }
    return {forwardDS, reverseBlock};
  }

  StmtDiff ReverseModeVisitor::DifferentiateLoopBody(const Stmt* body,
                                                     LoopCounter& loopCounter,
                                                     Stmt* condVarDiff,
                                                     Stmt* forLoopIncDiff,
                                                     bool isForLoop) {
    Expr* counterIncrement = loopCounter.getCounterIncrement();
    auto activeBreakContHandler = PushBreakContStmtHandler();
    activeBreakContHandler->BeginCFSwitchStmtScope();
    // differentiate loop body and add loop increment expression
    // in the forward block.
    StmtDiff bodyDiff = nullptr;
    if (isa<CompoundStmt>(body)) {
      bodyDiff = Visit(body);
      beginBlock(forward);
      addToCurrentBlock(counterIncrement);
      for (Stmt* S : cast<CompoundStmt>(bodyDiff.getStmt())->body())
        addToCurrentBlock(S);
      bodyDiff = {endBlock(forward), bodyDiff.getStmt_dx()};
    } else {
      // for forward-pass loop statement body
      beginScope(Scope::DeclScope);
      beginBlock(forward);
      addToCurrentBlock(counterIncrement);
      bodyDiff = DifferentiateSingleStmt(body, /*dfdS=*/nullptr,
                                         /*shouldEmit=*/false);
      addToCurrentBlock(bodyDiff.getStmt());
      // Emit some statemnts later to maintain correct statement order.
      if (m_ErrorEstimationEnabled) {
        errorEstHandler->EmitErrorEstimationStmts(forward);
      }
      Stmt* reverseBlock = unwrapIfSingleStmt(bodyDiff.getStmt_dx());
      bodyDiff = {endBlock(forward), reverseBlock};
      // for forward-pass loop statement body
      endScope();
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
    beginBlock(reverse);
    // `for` loops have counter decrement expression in the
    // loop iteration-expression.
    if (!isForLoop)
      addToCurrentBlock(counterDecrement, reverse);
    addToCurrentBlock(condVarDiff, reverse);
    addToCurrentBlock(bodyDiff.getStmt_dx(), reverse);
    addToCurrentBlock(forLoopIncDiff, reverse);
    bodyDiff = {bodyDiff.getStmt(), endBlock(reverse)};
    return bodyDiff;
  }

  StmtDiff ReverseModeVisitor::VisitContinueStmt(const ContinueStmt* CS) {
    beginBlock(forward);
    Stmt* newCS = m_Sema.ActOnContinueStmt(noLoc, getCurrentScope()).get();
    auto activeBreakContHandler = GetActiveBreakContStmtHandler();
    Stmt* CFCaseStmt = activeBreakContHandler->GetNextCFCaseStmt();
    Stmt* pushExprToCurrentCase = activeBreakContHandler
                                      ->CreateCFTapePushExprToCurrentCase();
    addToCurrentBlock(pushExprToCurrentCase);
    addToCurrentBlock(newCS);
    return {endBlock(forward), CFCaseStmt};
  }

  StmtDiff ReverseModeVisitor::VisitBreakStmt(const BreakStmt* BS) {
    beginBlock(forward);
    Stmt* newBS = m_Sema.ActOnBreakStmt(noLoc, getCurrentScope()).get();
    auto activeBreakContHandler = GetActiveBreakContStmtHandler();
    Stmt* CFCaseStmt = activeBreakContHandler->GetNextCFCaseStmt();
    Stmt* pushExprToCurrentCase = activeBreakContHandler
                                      ->CreateCFTapePushExprToCurrentCase();
    addToCurrentBlock(pushExprToCurrentCase);
    addToCurrentBlock(newBS);
    return {endBlock(forward), CFCaseStmt};
  }

  Expr* ReverseModeVisitor::BreakContStmtHandler::CreateSizeTLiteralExpr(
      std::size_t value) {
    ASTContext& C = m_RMV.m_Context;
    auto literalExpr = ConstantFolder::synthesizeLiteral(C.getSizeType(), C,
                                                         value);
    return literalExpr;
  }

  void ReverseModeVisitor::BreakContStmtHandler::InitializeCFTape() {
    assert(!m_ControlFlowTape && "InitializeCFTape() should not be called if "
                                 "m_ControlFlowTape is already initialized");

    auto zeroLiteral = CreateSizeTLiteralExpr(0);
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
    // End scope for currenly active case statement, if any.
    if (!m_SwitchCases.empty())
      m_RMV.endScope();

    ++m_CaseCounter;
    auto counterLiteral = CreateSizeTLiteralExpr(m_CaseCounter);
    CaseStmt* CS = clad_compat::CaseStmt_Create(m_RMV.m_Context, counterLiteral,
                                                nullptr, noLoc, noLoc, noLoc);

    // Initialise switch case statements with null statement because it is
    // necessary for switch case statements to have a substatement but it
    // is possible that there are no statements after the corresponding
    // break/continue statement. It's also easier to just set null statement
    // as substatement instead of keeping track of switch cases and
    // corresponding next statements.
    CS->setSubStmt(m_RMV.m_Sema.ActOnNullStmt(noLoc).get());

    // begin scope for the new active switch case statement.
    m_RMV.beginScope(Scope::DeclScope);
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
    if (m_SwitchCases.empty())
      return;

    // end scope for last switch case.
    m_RMV.endScope();

    // Add case statement in the beginning of the reverse block
    // and corresponding push expression for this case statement
    // at the end of the forward block to cover the case when no
    // `break`/`continue` statements are hit.
    auto lastSC = GetNextCFCaseStmt();
    auto pushExprToCurrentCase = CreateCFTapePushExprToCurrentCase();

    Stmt *forwBlock, *revBlock;

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
    SwitchStmt* CFSS = clad_compat::Sema_ActOnStartOfSwitchStmt(m_RMV.m_Sema,
                                                                nullptr,
                                                                condResult)
                           .getAs<SwitchStmt>();
    // Registers all the switch cases
    for (auto SC : m_SwitchCases) {
      CFSS->addSwitchCase(SC);
    }
    m_RMV.m_Sema.ActOnFinishSwitchStmt(noLoc, CFSS, bodyDiff.getStmt_dx());

    bodyDiff = {bodyDiff.getStmt(), CFSS};
  }
} // end namespace clad
