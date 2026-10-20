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
#include "clang/AST/Type.h"

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
  // Both halves go into a ValueAndPushforward<T, T> whose T is the primal's
  // return type, and a braced initializer refuses a narrowing conversion, so
  // each has to arrive as that type. The value and the tangent are asked
  // separately: a tangent the numerical fallback computed is `precision`
  // whatever the primal returns, which is a narrowing for a float function
  // even though the value beside it needs no cast at all.
  QualType retTy = m_DiffReq->getReturnType();
  auto castToReturnType = [&](Expr* E) {
    if (!E || m_Context.hasSameUnqualifiedType(E->getType(), retTy))
      return E;
    // Check if implficit cast would work.
    // Add a cast to the return type.
    auto* TSI = m_Context.getTrivialTypeSourceInfo(retTy);
    return m_Sema
        .BuildCStyleCastExpr(RS->getBeginLoc(), TSI, RS->getEndLoc(),
                             BuildParens(E))
        .get();
  };
  Expr* retVal = castToReturnType(retValDiff.getExpr());
  Expr* retVal_dx = castToReturnType(retValDiff.getExpr_dx());
  llvm::SmallVector<Expr*, 2> returnValues = {retVal, retVal_dx};
  Expr* initList = BuildInitList(returnValues);
  Stmt* returnStmt = BuildReturnStmt(initList);
  return StmtDiff(returnStmt);
}
} // end namespace clad
