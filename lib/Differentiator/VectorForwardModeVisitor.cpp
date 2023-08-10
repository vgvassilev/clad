#include "clad/Differentiator/VectorForwardModeVisitor.h"

#include "ConstantFolder.h"
#include "clad/Differentiator/CladUtils.h"

#include "clang/AST/TemplateName.h"
#include "clang/Sema/Lookup.h"
#include "llvm/Support/SaveAndRestore.h"

using namespace clang;

namespace clad {
VectorForwardModeVisitor::VectorForwardModeVisitor(DerivativeBuilder& builder)
    : BaseForwardModeVisitor(builder), m_IndVarCountExpr(nullptr) {}

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

    QualType valueType = utils::GetValueType(PVD->getType());
    valueType.removeLocalConst();
    QualType dParamType;
    if (utils::isArrayOrPointerType(PVD->getType())) {
      // Generate array reference type for the derivative.
      dParamType = GetCladArrayRefOfType(valueType);
    } else {
      // Generate pointer type for the derivative.
      dParamType = m_Context.getPointerType(valueType);
    }
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

  // Instantiate a variable indepVarCount to store the total number of
  // independent variables requested.
  // size_t indepVarCount = m_IndVarCountExpr;
  auto* totalIndVars = BuildVarDecl(m_Context.UnsignedLongTy, "indepVarCount",
                                    m_IndVarCountExpr);
  addToCurrentBlock(BuildDeclStmt(totalIndVars));
  m_IndVarCountExpr = BuildDeclRef(totalIndVars);

  // Expression for maintaining the number of independent variables processed
  // till now present as array elements. This will be sum of sizes of all such
  // arrays.
  Expr* arrayIndVarCountExpr = nullptr;

  // Number of non-array independent variables processed till now.
  size_t nonArrayIndVarCount = 0;

  // Current Index of independent variable in the param list of the function.
  size_t independentVarIndex = 0;

  for (size_t i = 0; i < m_Function->getNumParams(); ++i) {
    bool is_array =
        utils::isArrayOrPointerType(m_Function->getParamDecl(i)->getType());
    auto param = params[i];
    QualType dParamType = clad::utils::GetValueType(param->getType());

    Expr* dVectorParam = nullptr;
    if (m_IndependentVars.size() > independentVarIndex &&
        m_IndependentVars[independentVarIndex] == m_Function->getParamDecl(i)) {

      // Current offset for independent variable.
      Expr* offsetExpr = arrayIndVarCountExpr;
      Expr* nonArrayIndVarCountExpr = ConstantFolder::synthesizeLiteral(
          m_Context.UnsignedLongTy, m_Context, nonArrayIndVarCount);
      if (!offsetExpr)
        offsetExpr = nonArrayIndVarCountExpr;
      else if (nonArrayIndVarCount != 0)
        offsetExpr = BuildOp(BinaryOperatorKind::BO_Add, offsetExpr,
                             nonArrayIndVarCountExpr);

      if (is_array) {
        // Get size of the array.
        Expr* getSize = BuildArrayRefSizeExpr(
            m_ParamVariables[m_Function->getParamDecl(i)]);

        // Create an identity matrix for the parameter,
        // with number of rows equal to the size of the array,
        // and number of columns equal to the number of independent variables
        llvm::SmallVector<Expr*, 3> args = {getSize, m_IndVarCountExpr,
                                            offsetExpr};
        dVectorParam = BuildIdentityMatrixExpr(dParamType, args, loc);

        // Update the array independent expression.
        if (!arrayIndVarCountExpr) {
          arrayIndVarCountExpr = getSize;
        } else {
          arrayIndVarCountExpr = BuildOp(BinaryOperatorKind::BO_Add,
                                         arrayIndVarCountExpr, getSize);
        }
      } else {
        // Create a one hot vector for the parameter.
        llvm::SmallVector<Expr*, 2> args = {m_IndVarCountExpr, offsetExpr};
        dVectorParam = BuildCallExprToCladFunction("one_hot_vector", args,
                                                   {dParamType}, loc);
        ++nonArrayIndVarCount;
      }
      ++independentVarIndex;
    } else {
      // We cannot initialize derived variable for pointer types because
      // we do not know the correct size.
      if (is_array)
        continue;
      // This parameter is not an independent variable.
      // Initialize by all zeros.
      dVectorParam = BuildCallExprToCladFunction(
          "zero_vector", {m_IndVarCountExpr}, {dParamType}, loc);
    }

    // For each function arg to be differentiated, create a variable
    // _d_vector_arg to store the vector of derivatives for that arg.
    // for ex: double f(double x, double y, double z);
    // and we want to differentiate w.r.t. x and z, then we will have
    // -> clad::array<double> _d_vector_x = {1, 0};
    // -> clad::array<double> _d_vector_y = {0, 0};
    // -> clad::array<double> _d_vector_z = {0, 1};
    QualType dVectorParamType;
    if (is_array)
      dVectorParamType = GetCladMatrixOfType(dParamType);
    else
      dVectorParamType = GetCladArrayOfType(dParamType);
    auto dVectorParamDecl =
        BuildVarDecl(dVectorParamType, "_d_vector_" + param->getNameAsString(),
                     dVectorParam);
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
  QualType outputParamType = GetCladArrayRefOfType(m_Context.VoidTy);

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
    Expr* toCastExpr = BuildDeclRef(overloadParam);
    if (!isCladArrayType(vectorModeParam->getType())) {
      toCastExpr =
          BuildCallExprToMemFn(toCastExpr, /*MemberFunctionName=*/"ptr", {});
    }
    auto* castedExpr =
        m_Sema
            .BuildCXXNamedCast(
                noLoc, tok::TokenKind::kw_static_cast,
                m_Context.getTrivialTypeSourceInfo(vectorModeParam->getType()),
                toCastExpr, noLoc, noLoc)
            .get();
    auto* vectorModeVD =
        BuildVarDecl(vectorModeParam->getType(),
                     vectorModeParam->getNameAsString(), castedExpr);
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

llvm::SmallVector<clang::ParmVarDecl*, 8>
VectorForwardModeVisitor::BuildVectorModeParams(DiffParams& diffParams) {
  llvm::SmallVector<clang::ParmVarDecl*, 8> params, paramDerivatives;
  params.reserve(m_Function->getNumParams() + diffParams.size());
  auto derivativeFnType = cast<FunctionProtoType>(m_Derivative->getType());
  std::size_t dParamTypesIdx = m_Function->getNumParams();

  // Count the number of non-array independent variables requested for
  // differentiation.
  size_t nonArrayIndVarCount = 0;

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

    if (utils::isArrayOrPointerType(PVD->getType())) {
      m_ParamVariables[*it] = (Expr*)BuildDeclRef(dPVD);
      // dPVD will be a clad::array or clad::array_ref, both have size() method.
      // If m_IndVarCountExpr is null, initialize it with dPVD.size().
      // Otherwise, increment it by dPVD.size().
      Expr* getSize = BuildArrayRefSizeExpr(m_ParamVariables[*it]);
      if (!m_IndVarCountExpr) {
        m_IndVarCountExpr = getSize;
      } else {
        m_IndVarCountExpr =
            BuildOp(BinaryOperatorKind::BO_Add, m_IndVarCountExpr, getSize);
      }
    } else {
      m_ParamVariables[*it] = BuildOp(UO_Deref, BuildDeclRef(dPVD), noLoc);
      nonArrayIndVarCount += 1;
    }
  }

  // Process the expression for the number independent variables.
  // This will be the sum of the sizes of all array parameters and the number
  // of non-array parameters.
  Expr* nonArrayIndVarCountExpr = ConstantFolder::synthesizeLiteral(
      m_Context.UnsignedLongTy, m_Context, nonArrayIndVarCount);
  if (!m_IndVarCountExpr) {
    m_IndVarCountExpr = nonArrayIndVarCountExpr;
  } else if (nonArrayIndVarCount != 0) {
    m_IndVarCountExpr = BuildOp(BinaryOperatorKind::BO_Add, m_IndVarCountExpr,
                                nonArrayIndVarCountExpr);
  }

  // insert the derivative parameters at the end of the parameter list.
  params.insert(params.end(), paramDerivatives.begin(), paramDerivatives.end());
  // store the independent variables for later use.
  m_IndependentVars.insert(m_IndependentVars.end(), diffParams.begin(),
                           diffParams.end());
  return params;
}

StmtDiff VectorForwardModeVisitor::VisitArraySubscriptExpr(
    const ArraySubscriptExpr* ASE) {
  auto ASI = SplitArraySubscript(ASE);
  const Expr* Base = ASI.first;
  StmtDiff BaseDiff = Visit(Base);
  const auto& Indices = ASI.second;
  Expr* clonedBase = BaseDiff.getExpr();
  llvm::SmallVector<Expr*, 4> clonedIndices(Indices.size());
  std::transform(std::begin(Indices), std::end(Indices),
                 std::begin(clonedIndices),
                 [this](const Expr* E) { return Clone(E); });
  auto* cloned = BuildArraySubscript(clonedBase, clonedIndices);

  auto* zero = ConstantFolder::synthesizeLiteral(ASE->getType(), m_Context, 0);
  Expr* diffExpr = zero;

  Expr* target = BaseDiff.getExpr_dx();
  if (target) {
    diffExpr = m_Sema
                   .ActOnArraySubscriptExpr(getCurrentScope(), target,
                                            target->getExprLoc(),
                                            clonedIndices.front(), noLoc)
                   .get();
  }
  return StmtDiff(cloned, diffExpr);
}

StmtDiff VectorForwardModeVisitor::VisitReturnStmt(const ReturnStmt* RS) {
  const Expr* retVal = RS->getRetValue();
  QualType retType = retVal->getType();
  StmtDiff retValDiff = Visit(retVal);
  Expr* derivedRetValE = retValDiff.getExpr_dx();
  // If we are in vector mode, we need to wrap the return value in a
  // vector.
  SourceLocation loc{m_Function->getLocation()};
  llvm::SmallVector<Expr*, 2> args = {m_IndVarCountExpr, derivedRetValE};
  QualType cladArrayType = GetCladArrayOfType(utils::GetValueType(retType));
  TypeSourceInfo* TSI = m_Context.getTrivialTypeSourceInfo(cladArrayType, loc);
  Expr* constructorCallExpr =
      m_Sema.BuildCXXTypeConstructExpr(TSI, loc, args, loc, false).get();
  auto dVectorParamDecl =
      BuildVarDecl(cladArrayType, "_d_vector_return", constructorCallExpr,
                   false, nullptr, VarDecl::InitializationStyle::CallInit);
  // Create an array of statements to hold the return statement and the
  // assignments to the derivatives of the parameters.
  Stmts returnStmts;
  returnStmts.push_back(BuildDeclStmt(dVectorParamDecl));
  // Assign values from return vector to the derivatives of the
  // parameters.
  auto dVectorRef = BuildDeclRef(dVectorParamDecl);

  // Expression for maintaining the number of independent variables processed
  // till now present as array elements. This will be sum of sizes of all such
  // arrays.
  Expr* arrayIndVarCountExpr = nullptr;
  // Number of non-array independent variables processed till now.
  size_t nonArrayIndVarCount = 0;

  for (size_t i = 0; i < m_IndependentVars.size(); ++i) {
    // Get the derivative of the ith parameter.
    auto dParam = m_ParamVariables[m_IndependentVars[i]];
    Expr* dParamValue = nullptr;

    // Current offset for independent variable.
    Expr* offsetExpr = arrayIndVarCountExpr;
    Expr* nonArrayIndVarCountExpr = ConstantFolder::synthesizeLiteral(
        m_Context.UnsignedLongTy, m_Context, nonArrayIndVarCount);
    if (!offsetExpr)
      offsetExpr = nonArrayIndVarCountExpr;
    else if (nonArrayIndVarCount != 0)
      offsetExpr = BuildOp(BinaryOperatorKind::BO_Add, offsetExpr,
                           nonArrayIndVarCountExpr);

    if (isCladArrayType(dParam->getType())) {
      // Get the size of the array.
      Expr* getSize = BuildArrayRefSizeExpr(dParam);

      // Create an expression to fetch slice of the return vector.
      llvm::SmallVector<Expr*, 2> args = {offsetExpr, getSize};
      dParamValue = BuildArrayRefSliceExpr(dVectorRef, args);

      // Update the array independent expression.
      if (!arrayIndVarCountExpr) {
        arrayIndVarCountExpr = getSize;
      } else {
        arrayIndVarCountExpr =
            BuildOp(BinaryOperatorKind::BO_Add, arrayIndVarCountExpr, getSize);
      }
    } else {
      dParamValue = m_Sema
                        .ActOnArraySubscriptExpr(getCurrentScope(), dVectorRef,
                                                 dVectorRef->getExprLoc(),
                                                 offsetExpr, noLoc)
                        .get();
      ++nonArrayIndVarCount;
    }
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
  SourceLocation loc{m_Function->getLocation()};
  llvm::SmallVector<Expr*, 2> args = {m_IndVarCountExpr, initDiff.getExpr_dx()};
  QualType cladArrayType =
      GetCladArrayOfType(utils::GetValueType(VD->getType()));
  TypeSourceInfo* TSI = m_Context.getTrivialTypeSourceInfo(cladArrayType, loc);
  Expr* constructorCallExpr =
      m_Sema.BuildCXXTypeConstructExpr(TSI, loc, args, loc, false).get();

  VarDecl* VDDerived =
      BuildVarDecl(GetCladArrayOfType(utils::GetValueType(VD->getType())),
                   "_d_vector_" + VD->getNameAsString(), constructorCallExpr,
                   false, nullptr, VarDecl::InitializationStyle::CallInit);

  m_Variables.emplace(VDClone, BuildDeclRef(VDDerived));
  return VarDeclDiff(VDClone, VDDerived);
}

} // namespace clad
