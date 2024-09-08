//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// version: $Id: ClangPlugin.cpp 7 2013-06-01 22:48:03Z v.g.vassilev@gmail.com $
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#include "clad/Differentiator/PushForwardModeVisitor.h"
#include "clad/Differentiator/BaseForwardModeVisitor.h"

#include "clad/Differentiator/CladUtils.h"

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
  if (!m_Context.hasSameUnqualifiedType(retVal->getType(),
                                        m_DiffReq->getReturnType())) {
    // Check if implficit cast would work.
    // Add a cast to the return type.
    TypeSourceInfo* TSI =
        m_Context.getTrivialTypeSourceInfo(m_DiffReq->getReturnType());
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
  // This can instantiate as part of the move or copy initialization and
  // needs a fake source location.
  SourceLocation fakeLoc = utils::GetValidSLoc(m_Sema);
  Expr* initList = m_Sema.ActOnInitList(fakeLoc, returnValues, noLoc).get();
  Stmt* returnStmt =
      m_Sema.ActOnReturnStmt(fakeLoc, initList, getCurrentScope()).get();
  return StmtDiff(returnStmt);
}
} // end namespace clad
