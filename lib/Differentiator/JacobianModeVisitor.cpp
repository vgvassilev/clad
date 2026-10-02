#include "JacobianModeVisitor.h"

#include "ConstantFolder.h"
#include "clad/Differentiator/CladUtils.h"
#include "clad/Differentiator/DerivativeBuilder.h"

#include "clang/AST/Decl.h"
#include "clang/AST/OperationKinds.h"

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

  // Save Sema state before cloneFunction mutates it.
  llvm::SaveAndRestore<DeclContext*> SaveContext(m_Sema.CurContext);
  llvm::SaveAndRestore<Scope*> SaveScope(getCurrentScope());
  // Create the function declaration for the derivative.
  // FIXME: We should not use const_cast to get the decl context here.
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  auto* DC = const_cast<DeclContext*>(m_DiffReq->getDeclContext());
  m_Sema.CurContext = DC;
  // `result` owns the namespace Scopes cloneFunction opens; its
  // destructor pops them before SaveScope restores.
  ClonedFunction result = m_Builder.cloneFunction(
      m_DiffReq.Function, *this, DC, loc, name, vectorDiffFunctionType);
  FunctionDecl* vectorDiffFD = result.fd;
  m_Derivative = vectorDiffFD;

  // Function declaration scope
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

  // Running sum of independent-variable counts, materialized into the
  // m_IndVarCountDecl variable below.
  Expr* indVarCountExpr = nullptr;

  // Set the parameters for the derivative.
  llvm::SmallVector<ParmVarDecl*, 16> params;
  llvm::SmallVector<ParmVarDecl*, 16> derivedParams;
  llvm::SmallVector<DeclStmt*, 8> adjointDecls;

  bool hasCustomDiffArgs = m_DiffReq.HasCustomDiffArgs;

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
    VarDecl* adjointDecl = nullptr;
    AdjointInfo::WrapKind wrap = AdjointInfo::Plain;

    auto diffIt = std::find_if(m_DiffReq.DVI.begin(), m_DiffReq.DVI.end(),
                               [PVD](const DiffInputVarInfo& info) {
                                 return info.param == PVD;
                               });
    bool isDiffParam = (diffIt != m_DiffReq.DVI.end());

    if (hasCustomDiffArgs) {
      if (isDiffParam) {
        if (utils::isArrayOrPointerType(PVD->getType())) {
          QualType valType = utils::GetNonConstValueType(PVD->getType());
          QualType matrixType = utils::GetCladMatrixOfType(m_Sema, valType);
          VarDecl* derivedPVD = BuildVarDecl(matrixType, derivedPVDII);
          adjointDecls.push_back(BuildDeclStmt(derivedPVD));
          adjointDecl = derivedPVD;
          wrap = AdjointInfo::Plain;

          size_t arrSize = 0;
          if (diffIt->TotalCapacity > 0)
            arrSize = diffIt->TotalCapacity;
          else if (diffIt->paramIndexInterval.isValid())
            arrSize = diffIt->paramIndexInterval.size();
          else {
            QualType pvdType = PVD->getType();
            if (const auto* DT = dyn_cast<DecayedType>(pvdType))
              pvdType = DT->getOriginalType();
            if (const auto* CAT = m_Context.getAsConstantArrayType(pvdType))
              arrSize = CAT->getSize().getZExtValue();
          }
          Expr* getSize = nullptr;
          if (arrSize > 0) {
            getSize = ConstantFolder::synthesizeLiteral(
                m_Context.UnsignedLongTy, m_Context, arrSize);
          } else {
            SourceLocation L = PVD->getLocation();
            utils::diag(m_Sema, DiagnosticsEngine::Error, L,
                        "cannot determine size of array parameter '%0'; specify bounds like '%0[0:N]'")
                << PVD->getName() << L;
            return {};
          }
          if (!indVarCountExpr)
            indVarCountExpr = getSize;
          else
            indVarCountExpr =
                BuildOp(BinaryOperatorKind::BO_Add, indVarCountExpr, getSize);
        } else {
          VarDecl* derivedPVD = BuildVarDecl(
              utils::GetParameterDerivativeType(m_Sema, m_DiffReq.Mode,
                                                PVD->getType())
                  ->getPointeeType(),
              derivedPVDII);
          adjointDecls.push_back(BuildDeclStmt(derivedPVD));
          adjointDecl = derivedPVD;
          wrap = AdjointInfo::Plain;
          nonArrayIndVarCount += 1;
        }
      } else {
        // Non-independent parameter
        if (utils::isArrayOrPointerType(PVD->getType())) {
          if (!utils::GetValueType(PVD->getType()).isConstQualified()) {
            ParmVarDecl* derivedPVD = utils::BuildParmVarDecl(
                m_Sema, m_Derivative, derivedPVDII,
                utils::GetParameterDerivativeType(m_Sema, m_DiffReq.Mode,
                                                  PVD->getType()),
                PVD->getStorageClass());
            derivedParams.push_back(derivedPVD);
            adjointDecl = derivedPVD;
            wrap = AdjointInfo::ParenDeref;
          }
        } else if (PVD->getType()->isReferenceType()) {
          if (!utils::GetValueType(PVD->getType()).isConstQualified()) {
            ParmVarDecl* derivedPVD = utils::BuildParmVarDecl(
                m_Sema, m_Derivative, derivedPVDII,
                utils::GetParameterDerivativeType(m_Sema, m_DiffReq.Mode,
                                                  PVD->getType()),
                PVD->getStorageClass());
            derivedParams.push_back(derivedPVD);
            adjointDecl = derivedPVD;
            wrap = AdjointInfo::Deref;
          }
        } else {
          VarDecl* derivedPVD = BuildVarDecl(
              utils::GetParameterDerivativeType(m_Sema, m_DiffReq.Mode,
                                                PVD->getType())
                  ->getPointeeType(),
              derivedPVDII);
          adjointDecls.push_back(BuildDeclStmt(derivedPVD));
          adjointDecl = derivedPVD;
          wrap = AdjointInfo::Plain;
        }
      }
    } else {
      if (utils::isArrayOrPointerType(PVD->getType())) {
        ParmVarDecl* derivedPVD = utils::BuildParmVarDecl(
            m_Sema, m_Derivative, derivedPVDII,
            utils::GetParameterDerivativeType(m_Sema, m_DiffReq.Mode,
                                              PVD->getType()),
            PVD->getStorageClass());
        derivedParams.push_back(derivedPVD);
        adjointDecl = derivedPVD;
        wrap = AdjointInfo::ParenDeref;
        Expr* getSize = BuildCallExprToMemFn(BuildDeclRef(derivedPVD),
                                             /*MemberFunctionName=*/"rows", {});
        llvm::StringRef PVDName = PVD->getName();
        if (!PVDName.contains("_clad_out_")) {
          if (!indVarCountExpr)
            indVarCountExpr = getSize;
          else
            indVarCountExpr =
                BuildOp(BinaryOperatorKind::BO_Add, indVarCountExpr, getSize);
        }
      } else if (PVD->getType()->isReferenceType()) {
        ParmVarDecl* derivedPVD = utils::BuildParmVarDecl(
            m_Sema, m_Derivative, derivedPVDII,
            utils::GetParameterDerivativeType(m_Sema, m_DiffReq.Mode,
                                              PVD->getType()),
            PVD->getStorageClass());
        derivedParams.push_back(derivedPVD);
        adjointDecl = derivedPVD;
        wrap = AdjointInfo::Deref;
        nonArrayIndVarCount += 1;
      } else {
        VarDecl* derivedPVD = BuildVarDecl(
            utils::GetParameterDerivativeType(m_Sema, m_DiffReq.Mode,
                                              PVD->getType())
                ->getPointeeType(),
            derivedPVDII);
        adjointDecls.push_back(BuildDeclStmt(derivedPVD));
        adjointDecl = derivedPVD;
        nonArrayIndVarCount += 1;
      }
    }
    m_Variables[newPVD] = {adjointDecl, wrap};
  }

  params.insert(params.end(), derivedParams.begin(), derivedParams.end());

  // Process the expression for the number independent variables.
  // This will be the sum of the sizes of all array parameters and the number
  // of non-array parameters.
  Expr* nonArrayIndVarCountExpr = ConstantFolder::synthesizeLiteral(
      m_Context.UnsignedLongTy, m_Context, nonArrayIndVarCount);
  if (!indVarCountExpr) {
    indVarCountExpr = nonArrayIndVarCountExpr;
  } else if (nonArrayIndVarCount != 0) {
    indVarCountExpr = BuildOp(BinaryOperatorKind::BO_Add, indVarCountExpr,
                              nonArrayIndVarCountExpr);
  }

  utils::SetParams(vectorDiffFD,
                   clad_compat::makeArrayRef(params.data(), params.size()));
  vectorDiffFD->setBody(nullptr);

  // Instantiate a variable indepVarCount to store the total number of
  // independent variables requested.
  // size_t indepVarCount = indVarCountExpr;
  auto* totalIndVars =
      BuildVarDecl(m_Context.UnsignedLongTy, "indepVarCount", indVarCountExpr);
  addToCurrentBlock(BuildDeclStmt(totalIndVars));
  m_IndVarCountDecl = totalIndVars;

  for (DeclStmt* decl : adjointDecls)
    addToCurrentBlock(decl);

  // Offset of the next independent variable into the vector of all of them.
  IndVarOffsetTracker offsetTracker(*this);

  // Current Index of independent variable in the param list of the function.
  size_t independentVarIndex = 0;

  size_t numParamsOriginalFn = m_DiffReq->getNumParams();
  for (size_t i = 0; i < numParamsOriginalFn; ++i) {
    ParmVarDecl* param = params[i];
    if (!m_Variables[param].Decl)
      continue;
    bool is_array =
        utils::isArrayOrPointerType(m_DiffReq->getParamDecl(i)->getType());
    QualType dParamType = clad::utils::GetNonConstValueType(param->getType());
    // Desugaring the type is necessary to pass it to other templates
    dParamType = dParamType.getDesugaredType(m_Context);
    Expr* dVectorParam = nullptr;
    if (m_DiffReq.DVI.size() > independentVarIndex &&
        m_DiffReq.DVI[independentVarIndex].param ==
            m_DiffReq->getParamDecl(i)) {
      Expr* offsetExpr = offsetTracker.buildOffset();

      if (is_array) {
        Expr* getSize = nullptr;
        if (hasCustomDiffArgs) {
          size_t arrSize = 0;
          if (m_DiffReq.DVI[independentVarIndex].TotalCapacity > 0)
            arrSize = m_DiffReq.DVI[independentVarIndex].TotalCapacity;
          else if (m_DiffReq.DVI[independentVarIndex].paramIndexInterval.isValid())
            arrSize = m_DiffReq.DVI[independentVarIndex].paramIndexInterval.size();
          else {
            QualType pvdType = param->getType();
            if (const auto* DT = dyn_cast<DecayedType>(pvdType))
              pvdType = DT->getOriginalType();
            if (const auto* CAT = m_Context.getAsConstantArrayType(pvdType))
              arrSize = CAT->getSize().getZExtValue();
          }
          if (arrSize > 0)
            getSize = ConstantFolder::synthesizeLiteral(
                m_Context.UnsignedLongTy, m_Context, arrSize);
        }
        if (!getSize) {
          Expr* base = BuildDeclRef(m_Variables[param].Decl);
          getSize = BuildCallExprToMemFn(base, "rows", {});
        }
        llvm::SmallVector<Expr*, 3> args = {getSize, buildIndVarCountRef(),
                                            offsetExpr};
        dVectorParam = BuildIdentityMatrixExpr(dParamType, args);

        offsetTracker.advanceByArray(getSize);
      } else {
        // Create a one hot vector for the parameter.
        llvm::SmallVector<Expr*, 2> args = {buildIndVarCountRef(), offsetExpr};
        dVectorParam =
            BuildCallExprToCladFunction("one_hot_vector", args, {dParamType});
        offsetTracker.advanceByScalar();
      }
      ++independentVarIndex;
    } else {
      // We cannot initialize derived variable for pointer types because
      // we do not know the correct size.
      if (is_array)
        continue;
      // This parameter is not an independent variable.
      // Initialize by all zeros.
      Expr* dCount = buildIndVarCountRef();
      dVectorParam =
          BuildCallExprToCladFunction("zero_vector", {dCount}, {dParamType});
    }

    if (m_Variables[param].Wrap != AdjointInfo::Plain) {
      // The store target is `*_d_p`; strip the parens the ParenDeref adjoint
      // carries for element access elsewhere.
      Expr* paramAssignment =
          BuildOp(BO_Assign, buildAdjoint(m_Variables[param])->IgnoreParens(),
                  dVectorParam);
      addToCurrentBlock(paramAssignment);
    } else {
      SetDeclInit(m_Variables[param].Decl, dVectorParam);
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

StmtDiff JacobianModeVisitor::VisitReturnStmt(const clang::ReturnStmt* RS) {
  // If there is no return value, we must not attempt to differentiate
  if (!RS->getRetValue())
    return nullptr;

  StmtDiff retValDiff = Visit(RS->getRetValue());
  Stmt* returnStmt = BuildReturnStmt(retValDiff.getExpr_dx());
  return StmtDiff(returnStmt);
}
} // end namespace clad
