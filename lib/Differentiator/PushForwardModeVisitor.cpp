//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// version: $Id: ClangPlugin.cpp 7 2013-06-01 22:48:03Z v.g.vassilev@gmail.com $
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#include "clad/Differentiator/PushForwardModeVisitor.h"
#include "clad/Differentiator/BaseForwardModeVisitor.h"

#include "clad/Differentiator/CladUtils.h"
#include "clad/Differentiator/DerivativeBuilder.h"

#include "clang/AST/Stmt.h"

#include "llvm/Support/SaveAndRestore.h"

using namespace clang;

namespace clad {
PushForwardModeVisitor::PushForwardModeVisitor(DerivativeBuilder& builder,
                                               const DiffRequest& request)
    : BaseForwardModeVisitor(builder, request) {}

PushForwardModeVisitor::~PushForwardModeVisitor() = default;

StmtDiff PushForwardModeVisitor::VisitReturnStmt(const ReturnStmt* RS) {
  // If there is no return value, we must not attempt to differentiate
  if (!RS->getRetValue())
    return nullptr;

  StmtDiff retValDiff = Visit(RS->getRetValue());
  Expr* retVal = retValDiff.getExpr();
  Expr* retVal_dx = retValDiff.getExpr_dx();
  QualType RetTy = m_DiffReq->getReturnType();
  // A non-const reference return needs a tangent to refer to; a value without
  // a derivative has at most the literal 0, which nothing can bind.
  if (RetTy->isLValueReferenceType() &&
      !RetTy.getNonReferenceType().isConstQualified() &&
      (!retVal_dx || !retVal_dx->isLValue())) {
    SourceLocation L = RS->getRetValue()->getBeginLoc();
    diag(DiagnosticsEngine::Error, L,
         "cannot differentiate '%0': the returned reference refers to a value "
         "without a derivative")
        << m_DiffReq->getNameAsString() << L;
    return {};
  }
  if (!m_Context.hasSameUnqualifiedType(retVal->getType(), RetTy)) {
    // Check if implficit cast would work.
    // Add a cast to the return type.
    TypeSourceInfo* TSI = m_Context.getTrivialTypeSourceInfo(RetTy);
    retVal = m_Sema
                 .BuildCStyleCastExpr(RS->getBeginLoc(), TSI, RS->getEndLoc(),
                                      BuildParens(retVal))
                 .get();
    retVal_dx =
        m_Sema
            .BuildCStyleCastExpr(RS->getBeginLoc(), TSI, RS->getEndLoc(),
                                 BuildParens(retVal_dx))
            .get();
  }
  llvm::SmallVector<Expr*, 2> returnValues = {retVal, retVal_dx};
  Expr* initList = BuildInitList(returnValues);
  Stmt* returnStmt = BuildReturnStmt(initList);
  return StmtDiff(returnStmt);
}
} // end namespace clad
