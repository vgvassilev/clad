#include "clad/Differentiator/VectorForwardModeVisitor.h"

#include "ConstantFolder.h"
#include "clad/Differentiator/CladUtils.h"

#include "clang/AST/TemplateName.h"
#include "clang/Sema/Lookup.h"
#include "llvm/Support/SaveAndRestore.h"

using namespace clang;

namespace clad {
VectorForwardModeVisitor::VectorForwardModeVisitor(DerivativeBuilder& builder)
    : BaseForwardModeVisitor(builder) {}

VectorForwardModeVisitor::~VectorForwardModeVisitor() {}

DerivativeAndOverload
VectorForwardModeVisitor::DeriveVectorMode(const FunctionDecl* FD,
                                           const DiffRequest& request) {
  m_Function = FD;
  m_Mode = DiffMode::vector_forward_mode;

  DiffParams args{};
  DiffInputVarsInfo DVI;
  DVI = request.DVI;
  for (auto dParam : DVI)
    args.push_back(dParam.param);

  // Generate name for the derivative function.
  std::string derivedFnName = request.BaseFunctionName + "_dvec";
  if (args.size() != FD->getNumParams()) {
    for (auto arg : args) {
      auto it = std::find(FD->param_begin(), FD->param_end(), arg);
      auto idx = std::distance(FD->param_begin(), it);
      derivedFnName += ('_' + std::to_string(idx));
    }
  }
  IdentifierInfo* II = &m_Context.Idents.get(derivedFnName);
  SourceLocation loc{m_Function->getLocation()};
  DeclarationNameInfo name(II, loc);

  // Generate the function type for the derivative.
  llvm::SmallVector<clang::QualType, 8> paramTypes;
  paramTypes.reserve(m_Function->getNumParams() + args.size());
  for (auto PVD : m_Function->parameters()) {
    paramTypes.push_back(PVD->getType());
  }
  for (auto PVD : m_Function->parameters()) {
    auto it = std::find(std::begin(args), std::end(args), PVD);
    if (it == std::end(args))
      continue; // This parameter is not in the diff list.

    QualType ValueType = utils::GetValueType(PVD->getType());
    ValueType.removeLocalConst();
    // Generate pointer type for the derivative.
    QualType dParamType = m_Context.getPointerType(ValueType);
    paramTypes.push_back(dParamType);
  }

  QualType vectorDiffFunctionType = m_Context.getFunctionType(
      m_Context.VoidTy,
      llvm::ArrayRef<QualType>(paramTypes.data(), paramTypes.size()),
      // Cast to function pointer.
      dyn_cast<FunctionProtoType>(m_Function->getType())->getExtProtoInfo());

  // Create the function declaration for the derivative.
  DeclContext* DC = const_cast<DeclContext*>(m_Function->getDeclContext());
  m_Sema.CurContext = DC;
  DeclWithContext result =
      m_Builder.cloneFunction(m_Function, *this, DC, m_Sema, m_Context, loc,
                              name, vectorDiffFunctionType);
  FunctionDecl* vectorDiffFD = result.first;
  m_Derivative = vectorDiffFD;

  // Function declaration scope
  llvm::SaveAndRestore<DeclContext*> SaveContext(m_Sema.CurContext);
  llvm::SaveAndRestore<Scope*> SaveScope(m_CurScope);
  beginScope(Scope::FunctionPrototypeScope | Scope::FunctionDeclarationScope |
             Scope::DeclScope);
  m_Sema.PushFunctionScope();
  m_Sema.PushDeclContext(getCurrentScope(), m_Derivative);

  // Set the parameters for the derivative.
  auto params = BuildVectorModeParams(args);
  vectorDiffFD->setParams(
      clad_compat::makeArrayRef(params.data(), params.size()));
  vectorDiffFD->setBody(nullptr);

  // Create the body of the derivative.
  beginScope(Scope::FnScope | Scope::DeclScope);
  m_DerivativeFnScope = getCurrentScope();
  beginBlock();
  size_t independentVarIndex = 0;
  for (size_t i = 0; i < m_Function->getNumParams(); ++i) {
    auto param = params[i];
    QualType dParamType = clad::utils::GetValueType(param->getType());

    Expr* dVectorParam = nullptr;
    if (m_IndependentVars.size() > independentVarIndex &&
        m_IndependentVars[independentVarIndex] == m_Function->getParamDecl(i)) {
      // This parameter is an independent variable.
      // Create a one hot vector for the parameter.
      dVectorParam = getOneHotInitExpr(independentVarIndex,
                                       m_IndependentVars.size(), dParamType);
      ++independentVarIndex;
    } else {
      // This parameter is not an independent variable.
      // Initialize by all zeros.
      dVectorParam = getZeroInitListExpr(m_IndependentVars.size(), dParamType);
    }

    // For each function arg to be differentiated, create a variable
    // _d_vector_arg to store the vector of derivatives for that arg.
    // for ex: double f(double x, double y, double z);
    // and we want to differentiate w.r.t. x and z, then we will have
    // -> clad::array<double> _d_vector_x = {1, 0};
    // -> clad::array<double> _d_vector_y = {0, 0};
    // -> clad::array<double> _d_vector_z = {0, 1};
    auto dVectorParamDecl =
        BuildVarDecl(GetCladArrayOfType(dParamType),
                     "_d_vector_" + param->getNameAsString(), dVectorParam);
    addToCurrentBlock(BuildDeclStmt(dVectorParamDecl));
    dVectorParam = BuildDeclRef(dVectorParamDecl);
    // Memorize the derivative vector for the parameter.
    m_Variables[param] = dVectorParam;
  }

  // Traverse the function body and generate the derivative.
  Stmt* BodyDiff = Visit(FD->getBody()).getStmt();
  if (auto CS = dyn_cast<CompoundStmt>(BodyDiff))
    for (Stmt* S : CS->body())
      addToCurrentBlock(S);
  else
    addToCurrentBlock(BodyDiff);

  Stmt* vectorDiffBody = endBlock();
  m_Derivative->setBody(vectorDiffBody);
  endScope(); // Function body scope
  m_Sema.PopFunctionScopeInfo();
  m_Sema.PopDeclContext();
  endScope(); // Function decl scope

  // Create the overload declaration for the derivative.
  FunctionDecl* overloadFD = CreateVectorModeOverload();
  return DerivativeAndOverload{vectorDiffFD, overloadFD};
}

clang::FunctionDecl* VectorForwardModeVisitor::CreateVectorModeOverload() {
  auto vectorModeParams = m_Derivative->parameters();
  auto vectorModeNameInfo = m_Derivative->getNameInfo();

  // Calculate the total number of parameters that would be required for
  // automatic differentiation in the derived function if all args are
  // requested.
  std::size_t totalDerivedParamsSize = m_Function->getNumParams() * 2;
  std::size_t numDerivativeParams = m_Function->getNumParams();

  // Generate the function type for the derivative.
  llvm::SmallVector<clang::QualType, 8> paramTypes;
  paramTypes.reserve(totalDerivedParamsSize);
  for (auto* PVD : m_Function->parameters()) {
    paramTypes.push_back(PVD->getType());
  }

  // instantiate output parameter type as void*
  QualType outputParamType = m_Context.getPointerType(m_Context.VoidTy);

  // Push param types for derived params.
  for (std::size_t i = 0; i < m_Function->getNumParams(); ++i)
    paramTypes.push_back(outputParamType);

  auto vectorModeFuncOverloadEPI =
      dyn_cast<FunctionProtoType>(m_Function->getType())->getExtProtoInfo();
  QualType vectorModeFuncOverloadType = m_Context.getFunctionType(
      m_Context.VoidTy,
      llvm::ArrayRef<QualType>(paramTypes.data(), paramTypes.size()),
      vectorModeFuncOverloadEPI);

  // Create the function declaration for the derivative.
  auto* DC = const_cast<DeclContext*>(m_Function->getDeclContext());
  m_Sema.CurContext = DC;
  DeclWithContext result =
      m_Builder.cloneFunction(m_Function, *this, DC, m_Sema, m_Context, noLoc,
                              vectorModeNameInfo, vectorModeFuncOverloadType);
  FunctionDecl* vectorModeOverloadFD = result.first;

  // Function declaration scope
  beginScope(Scope::FunctionPrototypeScope | Scope::FunctionDeclarationScope |
             Scope::DeclScope);
  m_Sema.PushFunctionScope();
  m_Sema.PushDeclContext(getCurrentScope(), vectorModeOverloadFD);

  llvm::SmallVector<ParmVarDecl*, 4> overloadParams;
  overloadParams.reserve(totalDerivedParamsSize);

  llvm::SmallVector<Expr*, 4> callArgs; // arguments to the call the requested
                                        // vectormode function.
  callArgs.reserve(vectorModeParams.size());

  for (auto* PVD : m_Function->parameters()) {
    auto* VD = utils::BuildParmVarDecl(
        m_Sema, vectorModeOverloadFD, PVD->getIdentifier(), PVD->getType(),
        PVD->getStorageClass(), /*defArg=*/nullptr, PVD->getTypeSourceInfo());
    overloadParams.push_back(VD);
    callArgs.push_back(BuildDeclRef(VD));
  }

  for (std::size_t i = 0; i < numDerivativeParams; ++i) {
    ParmVarDecl* PVD = nullptr;
    std::size_t effectiveIndex = m_Function->getNumParams() + i;

    if (effectiveIndex < vectorModeParams.size()) {
      // This parameter represents an actual derivative parameter.
      auto* OriginalVD = vectorModeParams[effectiveIndex];
      PVD = utils::BuildParmVarDecl(
          m_Sema, vectorModeOverloadFD,
          CreateUniqueIdentifier("_temp_" + OriginalVD->getNameAsString()),
          outputParamType, OriginalVD->getStorageClass());
    } else {
      PVD = utils::BuildParmVarDecl(
          m_Sema, vectorModeOverloadFD,
          CreateUniqueIdentifier("_d_" + std::to_string(i)), outputParamType,
          StorageClass::SC_None);
    }
    overloadParams.push_back(PVD);
  }

  for (auto* PVD : overloadParams) {
    if (PVD->getIdentifier())
      m_Sema.PushOnScopeChains(PVD, getCurrentScope(),
                               /*AddToContext=*/false);
  }

  vectorModeOverloadFD->setParams(overloadParams);
  vectorModeOverloadFD->setBody(/*B=*/nullptr);

  // Create the body of the derivative.
  beginScope(Scope::FnScope | Scope::DeclScope);
  m_DerivativeFnScope = getCurrentScope();
  beginBlock();

  // Build derivatives to be used in the call to the actual derived function.
  // These are initialised by effectively casting the derivative parameters of
  // overloaded derived function to the correct type.
  for (std::size_t i = m_Function->getNumParams(); i < vectorModeParams.size();
       ++i) {
    auto* overloadParam = overloadParams[i];
    auto* vectorModeParam = vectorModeParams[i];

    // Create a cast expression to cast the derivative parameter to the correct
    // type.
    auto* castExpr =
        m_Sema
            .BuildCXXNamedCast(
                noLoc, tok::TokenKind::kw_static_cast,
                m_Context.getTrivialTypeSourceInfo(vectorModeParam->getType()),
                BuildDeclRef(overloadParam), noLoc, noLoc)
            .get();
    auto* vectorModeVD =
        BuildVarDecl(vectorModeParam->getType(),
                     vectorModeParam->getNameAsString(), castExpr);
    callArgs.push_back(BuildDeclRef(vectorModeVD));
    addToCurrentBlock(BuildDeclStmt(vectorModeVD));
  }

  Expr* callExpr = BuildCallExprToFunction(m_Derivative, callArgs,
                                           /*UseRefQualifiedThisObj=*/true);
  addToCurrentBlock(callExpr);
  Stmt* vectorModeOverloadBody = endBlock();

  vectorModeOverloadFD->setBody(vectorModeOverloadBody);

  endScope(); // Function body scope
  m_Sema.PopFunctionScopeInfo();
  m_Sema.PopDeclContext();
  endScope(); // Function decl scope

  return vectorModeOverloadFD;
}

clang::Expr* VectorForwardModeVisitor::getOneHotInitExpr(size_t index,
                                                         size_t size,
                                                         clang::QualType type) {
  // Build call expression for one_hot
  llvm::SmallVector<Expr*, 2> args = {
      ConstantFolder::synthesizeLiteral(m_Context.UnsignedLongTy, m_Context,
                                        size),
      ConstantFolder::synthesizeLiteral(m_Context.UnsignedLongTy, m_Context,
                                        index)};
  return BuildCallExprToCladFunction("one_hot_vector", args, {type},
                                     m_Function->getLocation());
}

clang::Expr*
VectorForwardModeVisitor::getZeroInitListExpr(size_t size,
                                              clang::QualType type) {
  // define a vector of size `size` with all elements set to 0.
  // Build call expression for zero_vector
  llvm::SmallVector<Expr*, 2> args = {ConstantFolder::synthesizeLiteral(
      m_Context.UnsignedLongTy, m_Context, size)};
  return BuildCallExprToCladFunction("zero_vector", args, {type},
                                     m_Function->getLocation());
}

llvm::SmallVector<clang::ParmVarDecl*, 8>
VectorForwardModeVisitor::BuildVectorModeParams(DiffParams& diffParams) {
  llvm::SmallVector<clang::ParmVarDecl*, 8> params, paramDerivatives;
  params.reserve(m_Function->getNumParams() + diffParams.size());
  auto derivativeFnType = cast<FunctionProtoType>(m_Derivative->getType());
  std::size_t dParamTypesIdx = m_Function->getNumParams();

  for (auto PVD : m_Function->parameters()) {
    auto newPVD = utils::BuildParmVarDecl(
        m_Sema, m_Derivative, PVD->getIdentifier(), PVD->getType(),
        PVD->getStorageClass(), /*DefArg=*/nullptr, PVD->getTypeSourceInfo());
    params.push_back(newPVD);

    if (newPVD->getIdentifier())
      m_Sema.PushOnScopeChains(newPVD, getCurrentScope(),
                               /*AddToContext=*/false);

    auto it = std::find(std::begin(diffParams), std::end(diffParams), PVD);
    if (it == std::end(diffParams))
      continue; // This parameter is not in the diff list.

    QualType dType = derivativeFnType->getParamType(dParamTypesIdx);
    IdentifierInfo* dII =
        CreateUniqueIdentifier("_d_" + PVD->getNameAsString());
    auto dPVD = utils::BuildParmVarDecl(m_Sema, m_Derivative, dII, dType,
                                        PVD->getStorageClass());
    paramDerivatives.push_back(dPVD);
    ++dParamTypesIdx;

    if (dPVD->getIdentifier())
      m_Sema.PushOnScopeChains(dPVD, getCurrentScope(),
                               /*AddToContext=*/false);

    m_ParamVariables[*it] = BuildOp(UO_Deref, BuildDeclRef(dPVD), noLoc);
  }
  // insert the derivative parameters at the end of the parameter list.
  params.insert(params.end(), paramDerivatives.begin(), paramDerivatives.end());
  // store the independent variables for later use.
  m_IndependentVars.insert(m_IndependentVars.end(), diffParams.begin(),
                           diffParams.end());
  return params;
}

StmtDiff VectorForwardModeVisitor::VisitReturnStmt(const ReturnStmt* RS) {
  StmtDiff retValDiff = Visit(RS->getRetValue());
  Expr* derivedRetValE = retValDiff.getExpr_dx();
  // If we are in vector mode, we need to wrap the return value in a
  // vector.
  auto dVectorParamDecl =
      BuildVarDecl(GetCladArrayOfType(
                       clad::utils::GetValueType(RS->getRetValue()->getType())),
                   "_d_vector_return", derivedRetValE);
  // Create an array of statements to hold the return statement and the
  // assignments to the derivatives of the parameters.
  Stmts returnStmts;
  returnStmts.push_back(BuildDeclStmt(dVectorParamDecl));
  // Assign values from return vector to the derivatives of the
  // parameters.
  auto dVectorRef = BuildDeclRef(dVectorParamDecl);
  for (size_t i = 0; i < m_IndependentVars.size(); ++i) {
    // Get the derivative of the ith parameter.
    auto dParam = m_ParamVariables[m_IndependentVars[i]];
    // Create an array subscript expression to access the ith element
    auto indexExpr =
        ConstantFolder::synthesizeLiteral(m_Context.IntTy, m_Context, i);
    auto dParamValue =
        m_Sema
            .ActOnArraySubscriptExpr(getCurrentScope(), dVectorRef,
                                     dVectorRef->getExprLoc(), indexExpr, noLoc)
            .get();
    // Create an assignment expression to assign the ith element of the
    // return vector to the derivative of the ith parameter.
    auto dParamAssign = BuildOp(BO_Assign, dParam, dParamValue);
    // Add the assignment statement to the array of statements.
    returnStmts.push_back(dParamAssign);
  }
  // Add an empty return statement to the array of statements.
  returnStmts.push_back(
      m_Sema.ActOnReturnStmt(noLoc, nullptr, getCurrentScope()).get());

  // Create a return statement from the compound statement.
  Stmt* returnStmt = MakeCompoundStmt(returnStmts);
  return StmtDiff(returnStmt);
}

VarDeclDiff VectorForwardModeVisitor::DifferentiateVarDecl(const VarDecl* VD) {
  StmtDiff initDiff = VD->getInit() ? Visit(VD->getInit()) : StmtDiff{};
  // Here we are assuming that derived type and the original type are same.
  // This may not necessarily be true in the future.
  VarDecl* VDClone =
      BuildVarDecl(VD->getType(), VD->getNameAsString(), initDiff.getExpr(),
                   VD->isDirectInit(), nullptr, VD->getInitStyle());
  // Create an expression to initialize the derivative vector of the
  // size of the number of parameters to be differentiated and initialize
  // the derivative vector to the derivative expression.
  //
  // For example:
  // clad::array<double> _d_vector_y(2, ...);
  //
  // This will also help in initializing the derivative vector in the
  // case when initExprDx is not an array.
  // So, for example, if we have:
  // clad::array<double> _d_vector_y(2, 1);
  // this means that we have to initialize the derivative vector of
  // size 2 with all elements equal to 1.
  Expr* size = ConstantFolder::synthesizeLiteral(
      m_Context.UnsignedLongTy, m_Context, m_IndependentVars.size());
  llvm::SmallVector<Expr*, 2> args = {size, initDiff.getExpr_dx()};
  Expr* constructorCallExpr =
      m_Sema
          .ActOnCXXTypeConstructExpr(
              OpaquePtr<QualType>::make(
                  GetCladArrayOfType(utils::GetValueType(VD->getType()))),
              noLoc, args, noLoc, false)
          .get();

  VarDecl* VDDerived =
      BuildVarDecl(GetCladArrayOfType(utils::GetValueType(VD->getType())),
                   "_d_vector_" + VD->getNameAsString(), constructorCallExpr,
                   false, nullptr, VarDecl::InitializationStyle::CallInit);

  m_Variables.emplace(VDClone, BuildDeclRef(VDDerived));
  return VarDeclDiff(VDClone, VDDerived);
}

} // namespace clad
