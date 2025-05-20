#include "JacobianModeVisitor.h"

#include "ConstantFolder.h"
#include "clad/Differentiator/CladUtils.h"
#include "clad/Differentiator/DerivativeBuilder.h"

#include "clang/Sema/Lookup.h"

#include "llvm/Support/SaveAndRestore.h"

using namespace clang;

namespace clad {
JacobianModeVisitor::JacobianModeVisitor(DerivativeBuilder& builder,
                                         const DiffRequest& request)
    : VectorPushForwardModeVisitor(builder, request) {}

DerivativeAndOverload JacobianModeVisitor::Derive() {
  const FunctionDecl* FD = m_DiffReq.Function;
  assert(m_DiffReq.Mode == DiffMode::jacobian);

  DiffParams args{};
  for (const DiffInputVarInfo& dParam : m_DiffReq.DVI)
    args.push_back(dParam.param);

  // Generate name for the derivative function.
  std::string derivedFnName = m_DiffReq.BaseFunctionName + "_jac";
  if (args.size() != FD->getNumParams()) {
    for (const ValueDecl* arg : args) {
      const auto* it = std::find(FD->param_begin(), FD->param_end(), arg);
      auto idx = std::distance(FD->param_begin(), it);
      derivedFnName += ('_' + std::to_string(idx));
    }
  }
  IdentifierInfo* II = &m_Context.Idents.get(derivedFnName);
  SourceLocation loc{m_DiffReq->getLocation()};
  DeclarationNameInfo name(II, loc);

  QualType vectorDiffFunctionType = GetDerivativeType();

  // Create the function declaration for the derivative.
  // FIXME: We should not use const_cast to get the decl context here.
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  auto* DC = const_cast<DeclContext*>(m_DiffReq->getDeclContext());
  m_Sema.CurContext = DC;
  DeclWithContext result = m_Builder.cloneFunction(
      m_DiffReq.Function, *this, DC, loc, name, vectorDiffFunctionType);
  FunctionDecl* vectorDiffFD = result.first;
  m_Derivative = vectorDiffFD;

  // Function declaration scope
  llvm::SaveAndRestore<DeclContext*> SaveContext(m_Sema.CurContext);
  llvm::SaveAndRestore<Scope*> SaveScope(getCurrentScope());
  beginScope(Scope::FunctionPrototypeScope | Scope::FunctionDeclarationScope |
             Scope::DeclScope);
  m_Sema.PushFunctionScope();
  m_Sema.PushDeclContext(getCurrentScope(), m_Derivative);

  // Create the body of the derivative.
  beginScope(Scope::FnScope | Scope::DeclScope);
  m_DerivativeFnScope = getCurrentScope();
  beginBlock();

  // Count the number of non-array independent variables requested for
  // differentiation.
  size_t nonArrayIndVarCount = 0;

  // Set the parameters for the derivative.
  llvm::SmallVector<ParmVarDecl*, 16> params;
  llvm::SmallVector<ParmVarDecl*, 16> derivedParams;
  llvm::SmallVector<DeclStmt*, 8> adjointDecls;

  auto origParams = FD->parameters();
  for (size_t i = 0, e = origParams.size(); i < e; ++i) {
    const ParmVarDecl* PVD = origParams[i];

    IdentifierInfo* PVDII = PVD->getIdentifier();
    auto* newPVD = CloneParmVarDecl(PVD, PVDII,
                                    /*pushOnScopeChains=*/true,
                                    /*cloneDefaultArg=*/false);
    params.push_back(newPVD);

    if (!utils::IsDifferentiableType(PVD->getType()))
      continue;
    auto derivedPVDName = "_d_vector_" + std::string(PVDII->getName());
    IdentifierInfo* derivedPVDII = CreateUniqueIdentifier(derivedPVDName);
    Expr* derivedExpr = nullptr;
    if (utils::isArrayOrPointerType(PVD->getType())) {
      ParmVarDecl* derivedPVD =
          utils::BuildParmVarDecl(m_Sema, m_Derivative, derivedPVDII,
                                  utils::GetParameterDerivativeType(
                                      m_Sema, m_DiffReq.Mode, PVD->getType()),
                                  PVD->getStorageClass());
      derivedParams.push_back(derivedPVD);
      derivedExpr =
          BuildOp(UO_Deref, BuildDeclRef(derivedPVD), PVD->getBeginLoc());
      derivedExpr = utils::BuildParenExpr(m_Sema, derivedExpr);
      Expr* getSize = BuildCallExprToMemFn(BuildDeclRef(derivedPVD),
                                           /*MemberFunctionName=*/"rows", {});
      llvm::StringRef PVDName = PVD->getName();
      if (!PVDName.contains("_clad_out_")) {
        if (!m_IndVarCountExpr)
          m_IndVarCountExpr = getSize;
        else
          m_IndVarCountExpr =
              BuildOp(BinaryOperatorKind::BO_Add, m_IndVarCountExpr, getSize);
      }
    } else if (PVD->getType()->isReferenceType()) {
      ParmVarDecl* derivedPVD =
          utils::BuildParmVarDecl(m_Sema, m_Derivative, derivedPVDII,
                                  utils::GetParameterDerivativeType(
                                      m_Sema, m_DiffReq.Mode, PVD->getType()),
                                  PVD->getStorageClass());
      derivedParams.push_back(derivedPVD);
      derivedExpr =
          BuildOp(UO_Deref, BuildDeclRef(derivedPVD), PVD->getBeginLoc());
      nonArrayIndVarCount += 1;
    } else {
      VarDecl* derivedPVD =
          BuildVarDecl(utils::GetParameterDerivativeType(m_Sema, m_DiffReq.Mode,
                                                         PVD->getType())
                           ->getPointeeType(),
                       derivedPVDII);
      adjointDecls.push_back(BuildDeclStmt(derivedPVD));
      derivedExpr = BuildDeclRef(derivedPVD);
      nonArrayIndVarCount += 1;
    }
    m_Variables[newPVD] = derivedExpr;
  }

  params.insert(params.end(), derivedParams.begin(), derivedParams.end());

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

  vectorDiffFD->setParams(
      clad_compat::makeArrayRef(params.data(), params.size()));
  vectorDiffFD->setBody(nullptr);

  // Instantiate a variable indepVarCount to store the total number of
  // independent variables requested.
  // size_t indepVarCount = m_IndVarCountExpr;
  auto* totalIndVars = BuildVarDecl(m_Context.UnsignedLongTy, "indepVarCount",
                                    m_IndVarCountExpr);
  addToCurrentBlock(BuildDeclStmt(totalIndVars));
  m_IndVarCountExpr = BuildDeclRef(totalIndVars);

  for (DeclStmt* decl : adjointDecls)
    addToCurrentBlock(decl);

  // Expression for maintaining the number of independent variables processed
  // till now present as array elements. This will be sum of sizes of all such
  // arrays.
  Expr* arrayIndVarCountExpr = nullptr;

  // Number of non-array independent variables processed till now.
  nonArrayIndVarCount = 0;

  // Current Index of independent variable in the param list of the function.
  size_t independentVarIndex = 0;

  size_t numParamsOriginalFn = m_DiffReq->getNumParams();
  for (size_t i = 0; i < numParamsOriginalFn; ++i) {
    bool is_array =
        utils::isArrayOrPointerType(m_DiffReq->getParamDecl(i)->getType());
    ParmVarDecl* param = params[i];
    Expr* paramDiff = m_Variables[param]->IgnoreParens();
    QualType dParamType = clad::utils::GetValueType(param->getType());
    // Desugaring the type is necessary to pass it to other templates
    dParamType = dParamType.getDesugaredType(m_Context);
    Expr* dVectorParam = nullptr;
    if (m_DiffReq.DVI.size() > independentVarIndex &&
        m_DiffReq.DVI[independentVarIndex].param ==
            m_DiffReq->getParamDecl(i)) {
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
        Expr* base = cast<UnaryOperator>(paramDiff)->getSubExpr();
        // Get size of the array.
        Expr* getSize = BuildCallExprToMemFn(Clone(base),
                                             /*MemberFunctionName=*/"rows", {});
        // Create an identity matrix for the parameter,
        // with number of rows equal to the size of the array,
        // and number of columns equal to the number of independent variables
        llvm::SmallVector<Expr*, 3> args = {getSize, m_IndVarCountExpr,
                                            offsetExpr};
        dVectorParam = BuildIdentityMatrixExpr(dParamType, args, loc);

        // Update the array independent expression.
        if (!arrayIndVarCountExpr)
          arrayIndVarCountExpr = getSize;
        else
          arrayIndVarCountExpr = BuildOp(BinaryOperatorKind::BO_Add,
                                         arrayIndVarCountExpr, getSize);
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
    if (utils::isArrayOrPointerType(param->getType()) ||
        param->getType()->isReferenceType()) {
      Expr* paramAssignment =
          BuildOp(BO_Assign, Clone(paramDiff), dVectorParam);
      addToCurrentBlock(paramAssignment);
    } else {
      auto* paramDecl = cast<VarDecl>(cast<DeclRefExpr>(paramDiff)->getDecl());
      SetDeclInit(paramDecl, dVectorParam);
    }
  }

  // Traverse the function body and generate the derivative.
  Stmt* BodyDiff = Visit(FD->getBody()).getStmt();
  for (Stmt* S : cast<CompoundStmt>(BodyDiff)->body())
    addToCurrentBlock(S);

  Stmt* vectorDiffBody = endBlock();
  m_Derivative->setBody(vectorDiffBody);
  endScope(); // Function body scope
  m_Sema.PopFunctionScopeInfo();
  m_Sema.PopDeclContext();
  endScope(); // Function decl scope
  // Create the overload declaration for the derivative.
  FunctionDecl* overloadFD = CreateDerivativeOverload();
  return DerivativeAndOverload{vectorDiffFD, overloadFD};
}

clang::Expr*
ComputeColIdx(clang::ASTContext& m_Context, clang::Sema& m_Sema,
              const std::set<std::pair<int, int>, utils::compare>& depset) {
  llvm::SmallVector<Expr*, 8> Elements;
  for (auto p : depset) {
    Expr* colIdxE = IntegerLiteral::Create(m_Context, llvm::APInt(32, p.first),
                                           m_Context.IntTy, noLoc);
    Elements.emplace_back(colIdxE);
  }
  return m_Sema.ActOnInitList(noLoc, Elements, noLoc).get();
}

clang::Expr*
ComputeRowIdx(clang::ASTContext& m_Context, clang::Sema& m_Sema,
              const std::set<std::pair<int, int>, utils::compare>& depset) {
  llvm::SmallVector<Expr*, 8> Elements;
  int nonZeroAbove = 0;
  int currentRow = -1;
  for (auto p : depset) {
    if (currentRow != p.second) {
      Expr* nonZeroAboveE = IntegerLiteral::Create(
          m_Context, llvm::APInt(32, nonZeroAbove), m_Context.IntTy, noLoc);
      Elements.emplace_back(nonZeroAboveE);
      currentRow += 1;
    }
    nonZeroAbove++;
  }
  Expr* nonZeroAboveE = IntegerLiteral::Create(
      m_Context, llvm::APInt(32, nonZeroAbove), m_Context.IntTy, noLoc);
  Elements.emplace_back(nonZeroAboveE);
  return m_Sema.ActOnInitList(noLoc, Elements, noLoc).get();
}

SparseJacobianModeVisitor::SparseJacobianModeVisitor(DerivativeBuilder& builder,
                                                     const DiffRequest& request)
    : VisitorBase(builder, request) {}

DerivativeAndOverload SparseJacobianModeVisitor::Derive() {
  const FunctionDecl* FD = m_DiffReq.Function;
  assert(m_DiffReq.Mode == DiffMode::jacobian && m_DiffReq.EnableSparsity);

  DiffParams args{};
  QualType outputType;

  for (const DiffInputVarInfo& dParam : m_DiffReq.DVI) {
    // rework
    if (llvm::StringRef(dParam.param->getName()).contains("_clad_out_"))
      outputType = dParam.param->getType()->getPointeeType();
    args.push_back(dParam.param);
  }
  // FIXME: Add proper directional derivative handling. Dependency analysis does
  // not account for parameters that are not in DVI, but it should.
  std::string derivedFnName = m_DiffReq.BaseFunctionName + "_sparse_jac";
  std::string jacName = m_DiffReq.BaseFunctionName + "_jac";

  if (args.size() != FD->getNumParams()) {
    for (const ValueDecl* arg : args) {
      const auto* it = std::find(FD->param_begin(), FD->param_end(), arg);
      auto idx = std::distance(FD->param_begin(), it);
      derivedFnName += ('_' + std::to_string(idx));
      jacName += ('_' + std::to_string(idx));
    }
  }

  IdentifierInfo* II = &m_Context.Idents.get(derivedFnName);
  SourceLocation loc{m_DiffReq->getLocation()};
  DeclarationNameInfo name(II, loc);

  static TemplateDecl* sparsityDecl = utils::LookupTemplateDeclInCladNamespace(
      m_Sema, /*ClassName=*/"sparsity_pattern");

  QualType sparsityQT =
      utils::InstantiateTemplate(m_Sema, sparsityDecl, {outputType});
  sparsityQT = m_Context.getPointerType(sparsityQT);
  QualType vectorDiffFunctionType = GetDerivativeType(sparsityQT);

  auto* jacDC = const_cast<DeclContext*>(m_DiffReq->getDeclContext());
  m_Sema.CurContext = jacDC;
  DeclWithContext result = m_Builder.cloneFunction(
      m_DiffReq.Function, *this, jacDC, loc, name, vectorDiffFunctionType);
  FunctionDecl* vectorSparseDiffFD = result.first;
  m_Derivative = vectorSparseDiffFD;

  clang::LookupResult jacLR =
      utils::LookupQualifiedName(jacName, m_Sema, jacDC);
  FunctionDecl* jacFD = dyn_cast<FunctionDecl>(jacLR.getRepresentativeDecl());

  CXXScopeSpec CSS;
  Expr* UnresolvedLookup =
      m_Sema.BuildDeclarationNameExpr(CSS, jacLR, /*ADL=*/true).get();

  llvm::SmallVector<Expr*, 16> callArgs;
  llvm::SmallVector<ParmVarDecl*, 16> paramsNew;

  unsigned indVarCount = 0;
  ParmVarDecl* outputPVD = nullptr;
  for (auto* PVD : jacFD->parameters()) {
    auto* newPVD = CloneParmVarDecl(PVD, PVD->getIdentifier(),
                                    /*pushOnScopeChains=*/false,
                                    /*cloneDefaultArg=*/false);
    // this is bad
    if (!utils::isArrayOrPointerType(PVD->getType()))
      ++indVarCount;
    if (llvm::StringRef(PVD->getName()).contains("_clad_out_"))
      outputPVD = newPVD;

    paramsNew.push_back(newPVD);
    callArgs.push_back(BuildDeclRef(cast<VarDecl>(newPVD)));
  }

  llvm::StringRef sName = "sparsity_patern";
  IdentifierInfo* sII = CreateUniqueIdentifier(sName);
  ParmVarDecl* sparsityPVD =
      utils::BuildParmVarDecl(m_Sema, m_Derivative, sII, sparsityQT);
  paramsNew.push_back(sparsityPVD);

  vectorSparseDiffFD->setParams(
      clad_compat::makeArrayRef(paramsNew.data(), paramsNew.size()));
  vectorSparseDiffFD->setBody(nullptr);

  llvm::SaveAndRestore<DeclContext*> SaveContext(m_Sema.CurContext);
  llvm::SaveAndRestore<Scope*> SaveScope(getCurrentScope());
  beginScope(Scope::FunctionPrototypeScope | Scope::FunctionDeclarationScope |
             Scope::DeclScope);
  m_Sema.PushFunctionScope();
  m_Sema.PushDeclContext(getCurrentScope(), m_Derivative);
  beginScope(Scope::FnScope | Scope::DeclScope);
  m_DerivativeFnScope = getCurrentScope();
  beginBlock();

  Expr* jacobianCall =
      m_Sema
          .ActOnCallExpr(getCurrentScope(), UnresolvedLookup, loc,
                         llvm::MutableArrayRef<Expr*>(callArgs),
                         m_DiffReq->getLocation())
          .get();

  llvm::SmallVector<Expr*, 8> Elements;
  for (auto idx : m_DiffReq.getDependencySet()) {
    Expr* idxIL = IntegerLiteral::Create(
        m_Context, llvm::APInt(32, idx.first + idx.second * indVarCount),
        m_Context.IntTy, noLoc);
    Elements.emplace_back(idxIL);
  }

  Expr* InitExprFull = m_Sema.ActOnInitList(loc, Elements, loc).get();

#if CLANG_VERSION_MAJOR >= 18
  QualType arrayTyInt = m_Context.getIncompleteArrayType(
      m_Context.IntTy, clang::ArraySizeModifier::Normal,
      /*IndexTypeQuals=*/0);
#else
  QualType arrayTyInt = m_Context.getIncompleteArrayType(
      m_Context.IntTy, clang::VariableArrayType::Normal,
      /*IndexTypeQuals=*/0);
#endif

  VarDecl* depSetDecl =
      BuildVarDecl(arrayTyInt, "dependency_set", InitExprFull);
  Expr* colIdxInitList =
      ComputeColIdx(m_Context, m_Sema, m_DiffReq.getDependencySet());
  Expr* rowIdxInitList =
      ComputeRowIdx(m_Context, m_Sema, m_DiffReq.getDependencySet());
  Expr* setColIdxE = BuildCallExprToMemFn(
      BuildDeclRef(dyn_cast<VarDecl>(sparsityPVD)),
      /*MemberFunctionName=*/"set_col_idx", {colIdxInitList});
  Expr* setRowIdxE = BuildCallExprToMemFn(
      BuildDeclRef(dyn_cast<VarDecl>(sparsityPVD)),
      /*MemberFunctionName=*/"set_row_idx", {rowIdxInitList});

  VarDecl* counterVD =
      BuildVarDecl(m_Context.IntTy, "i", getZeroInit(m_Context.IntTy));
  Expr* counterDRE = BuildDeclRef(counterVD);
  Expr* counterInc = BuildOp(UO_PreInc, counterDRE);
  Expr* forCond = BuildOp(
      BO_LT, counterDRE,
      IntegerLiteral::Create(
          m_Context,
          llvm::APInt(/*bitwidth=*/32, m_DiffReq.getDependencySet().size()),
          m_Context.IntTy, loc));

  Expr* dependencySetASE =
      m_Sema
          .ActOnArraySubscriptExpr(getCurrentScope(), BuildDeclRef(depSetDecl),
                                   loc, counterDRE, loc)
          .get();

  Expr* sparsityE = BuildOp(UO_Deref, BuildDeclRef(sparsityPVD));
  sparsityE = utils::BuildParenExpr(m_Sema, sparsityE);

  Expr* nnzValsASE = m_Sema
                         .ActOnArraySubscriptExpr(getCurrentScope(), sparsityE,
                                                  loc, counterDRE, loc)
                         .get();

  Expr* indVarExpr = IntegerLiteral::Create(
      m_Context, llvm::APInt(32, indVarCount), m_Context.IntTy, loc);
  Expr* matrixFirstIdx = BuildOp(BO_Div, dependencySetASE, indVarExpr);
  Expr* matrixSecondIdx = BuildOp(BO_Rem, dependencySetASE, indVarExpr);

  Expr* outputE = BuildParens(BuildOp(UO_Deref, BuildDeclRef(outputPVD)));
  outputE = utils::BuildParenExpr(m_Sema, outputE);

  Expr* outputFirstASE =
      m_Sema
          .ActOnArraySubscriptExpr(getCurrentScope(), outputE, loc,
                                   matrixFirstIdx, loc)
          .get();

  Expr* outputFullASE =
      m_Sema
          .ActOnArraySubscriptExpr(getCurrentScope(), outputFirstASE, loc,
                                   matrixSecondIdx, loc)
          .get();

  Expr* assignVals = BuildOp(BO_Assign, nnzValsASE, outputFullASE);

  Stmt* forAssignVals = new (m_Context)
      ForStmt(m_Context, /*Init=*/BuildDeclStmt(counterVD), forCond,
              /*CondVar=*/nullptr, counterInc, assignVals, loc, loc, loc);

  addToCurrentBlock(jacobianCall);
  addToCurrentBlock(BuildDeclStmt(depSetDecl));
  addToCurrentBlock(setColIdxE);
  addToCurrentBlock(setRowIdxE);
  addToCurrentBlock(forAssignVals);

  Stmt* vectorDiffBody = endBlock();
  m_Derivative->setBody(vectorDiffBody);
  endScope(); // Function body scope
  m_Sema.PopFunctionScopeInfo();
  m_Sema.PopDeclContext();
  endScope(); // Function decl scope
  FunctionDecl* overloadFD = CreateDerivativeOverload();
  return DerivativeAndOverload{vectorSparseDiffFD, overloadFD};
}

StmtDiff JacobianModeVisitor::VisitReturnStmt(const clang::ReturnStmt* RS) {
  // If there is no return value, we must not attempt to differentiate
  if (!RS->getRetValue())
    return nullptr;

  StmtDiff retValDiff = Visit(RS->getRetValue());
  // This can instantiate as part of the move or copy initialization and
  // needs a fake source location.
  SourceLocation fakeLoc = utils::GetValidSLoc(m_Sema);
  Stmt* returnStmt =
      m_Sema
          .ActOnReturnStmt(fakeLoc, retValDiff.getExpr_dx(), getCurrentScope())
          .get();
  return StmtDiff(returnStmt);
}
} // end namespace clad
