#include "clad/Differentiator/VectorPushForwardModeVisitor.h"

#include "ConstantFolder.h"
#include "clad/Differentiator/CladUtils.h"

#include "llvm/Support/SaveAndRestore.h"

using namespace clang;

namespace clad {
VectorPushForwardModeVisitor::VectorPushForwardModeVisitor(
    DerivativeBuilder& builder, const DiffRequest& request)
    : VectorForwardModeVisitor(builder, request) {}

VectorPushForwardModeVisitor::~VectorPushForwardModeVisitor() = default;

void VectorPushForwardModeVisitor::ExecuteInsidePushforwardFunctionBlock() {
  // Extract the last parameter of the m_Derivative function.
  // This parameter will either be a clad array or a matrix.
  // If it's a clad array, use it's size, or if it's a clad matrix
  // use the size of 0th element of the matrix.
  ParmVarDecl* lastParam =
      m_Derivative->getParamDecl(m_Derivative->getNumParams() - 1);
  QualType lastParamType = utils::GetValueType(lastParam->getType());
  Expr* lastParamExpr = BuildDeclRef(lastParam);
  Expr* lastParamSizeExpr = nullptr;
  if (isCladArrayType(lastParamType)) {
    lastParamSizeExpr = BuildArrayRefSizeExpr(lastParamExpr);
  } else {
    auto* zero =
        ConstantFolder::synthesizeLiteral(m_Context.IntTy, m_Context, 0);
    Expr* arraySubscriptExpr =
        m_Sema
            .ActOnArraySubscriptExpr(getCurrentScope(), lastParamExpr,
                                     lastParamExpr->getExprLoc(), zero, noLoc)
            .get();
    lastParamSizeExpr = BuildArrayRefSizeExpr(arraySubscriptExpr);
  }

  // Create a variable to store the total number of independent variables.
  Expr* indVarCountExpr = lastParamSizeExpr;
  auto* totalIndVars =
      BuildVarDecl(m_Context.UnsignedLongTy, "indepVarCount", indVarCountExpr);
  addToCurrentBlock(BuildDeclStmt(totalIndVars));
  SetIndependentVarsExpr(BuildDeclRef(totalIndVars));

  BaseForwardModeVisitor::ExecuteInsidePushforwardFunctionBlock();
}

QualType
VectorPushForwardModeVisitor::GetParameterDerivativeType(QualType ParamType) {
  QualType valueType = utils::GetNonConstValueType(ParamType);
  QualType resType;
  if (utils::isArrayOrPointerType(ParamType)) {
    // If the parameter is a pointer or an array, then the derivative will be a
    // reference to the matrix.
    resType = GetCladMatrixOfType(valueType);
    resType = m_Context.getLValueReferenceType(resType);
  } else {
    // If the parameter is not a pointer or an array, then the derivative will
    // be a clad array.
    resType = GetCladArrayOfType(valueType);

    // Add const qualifier if the parameter is const.
    if (ParamType.getNonReferenceType().isConstQualified())
      resType.addConst();

    // Add reference qualifier if the parameter is a reference.
    if (ParamType->isReferenceType())
      resType = m_Context.getLValueReferenceType(resType);
  }
  return resType;
}

StmtDiff
VectorPushForwardModeVisitor::VisitReturnStmt(const clang::ReturnStmt* RS) {
  // If there is no return value, we must not attempt to differentiate
  if (!RS->getRetValue())
    return nullptr;

  StmtDiff retValDiff = Visit(RS->getRetValue());
  llvm::SmallVector<Expr*, 2> returnValues = {retValDiff.getExpr(),
                                              retValDiff.getExpr_dx()};
  // This can instantiate as part of the move or copy initialization and
  // needs a fake source location.
  SourceLocation fakeLoc = utils::GetValidSLoc(m_Sema);
  Expr* initList = m_Sema.ActOnInitList(fakeLoc, returnValues, noLoc).get();
  Stmt* returnStmt =
      m_Sema.ActOnReturnStmt(fakeLoc, initList, getCurrentScope()).get();
  return StmtDiff(returnStmt);
}

} // end namespace clad
