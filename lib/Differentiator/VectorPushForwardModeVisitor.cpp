#include "clad/Differentiator/VectorPushForwardModeVisitor.h"

#include "ConstantFolder.h"
#include "clad/Differentiator/CladUtils.h"
#include "clad/Differentiator/Compatibility.h"
#include "clad/Differentiator/DerivativeBuilder.h"

#include "clang/AST/Expr.h"
#include "clang/Basic/LLVM.h"

#include "llvm/ADT/SmallVector.h"
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
    Expr* arraySubscriptExpr = BuildArraySubscript(lastParamExpr, zero);
    lastParamSizeExpr = BuildArrayRefSizeExpr(arraySubscriptExpr);
  }

  // Create a variable to store the total number of independent variables.
  Expr* indVarCountExpr = lastParamSizeExpr;
  auto* totalIndVars =
      BuildVarDecl(m_Context.UnsignedLongTy, "indepVarCount", indVarCountExpr);
  addToCurrentBlock(BuildDeclStmt(totalIndVars));
  m_IndVarCountDecl = totalIndVars;

  BaseForwardModeVisitor::ExecuteInsidePushforwardFunctionBlock();
}

DerivativeAndOverload VectorPushForwardModeVisitor::Derive() {
  return BaseForwardModeVisitor::Derive();
}

StmtDiff
VectorPushForwardModeVisitor::VisitReturnStmt(const clang::ReturnStmt* RS) {
  // If there is no return value, we must not attempt to differentiate
  if (!RS->getRetValue())
    return nullptr;

  StmtDiff retValDiff = Visit(RS->getRetValue());
  Expr* retVal = retValDiff.getExpr();
  Expr* retValDx = retValDiff.getExpr_dx();
  QualType RetTy = m_DiffReq->getReturnType();
  // A pushforward of a void function returns void as well, while preserving any
  // tangent side effects produced by the return operand (e.g., comma).
  if (RetTy->isVoidType()) {
    if (retValDx && retValDx->HasSideEffects(m_Context))
      addToCurrentBlock(retValDx);
    return StmtDiff(BuildReturnStmt(retVal));
  }
  // As in scalar mode: the value member has the function's return type, and
  // a braced list does not narrow to it.
  if (!clad_compat::hasSameUnqualifiedType(m_Context, retVal->getType(),
                                           RetTy)) {
    auto* TSI = m_Context.getTrivialTypeSourceInfo(RetTy);
    retVal = m_Sema
                 .BuildCStyleCastExpr(RS->getBeginLoc(), TSI, RS->getEndLoc(),
                                      BuildParens(retVal))
                 .get();
  }
  bool needsNonConstRef = RetTy->isLValueReferenceType() &&
                          !RetTy.getNonReferenceType().isConstQualified();
  // The pushforward member is a vector of tangents; a constant zero would
  // size it rather than fill it, while a non-const reference return needs an
  // lvalue tangent directly.
  if (!needsNonConstRef)
    retValDx = asTangentVector(retValDx, RS->getRetValue()->getType());
  if (!retValDx || (needsNonConstRef && !retValDx->isLValue())) {
    retValDx = zeroReturnTangent(RS->getRetValue(), RetTy);
    if (!retValDx)
      return {};
  }
  llvm::SmallVector<Expr*, 2> returnValues = {retVal, retValDx};
  Expr* initList = BuildInitList(returnValues);
  Stmt* returnStmt = BuildReturnStmt(initList);
  return StmtDiff(returnStmt);
}

} // end namespace clad
