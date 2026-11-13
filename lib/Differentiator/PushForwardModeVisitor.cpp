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
  // A pushforward of a void function returns void as well. An early return
  // in it must still leave: the primal does not run what follows it, so
  // neither its value nor its tangent may be computed.
  if (!RS->getRetValue()) {
    if (RS == m_DiffReq.getTailReturn())
      return nullptr;
    return StmtDiff(BuildReturnStmt(nullptr));
  }

  StmtDiff retValDiff = Visit(RS->getRetValue());
  Expr* retVal = retValDiff.getExpr();
  Expr* retVal_dx = retValDiff.getExpr_dx();
  QualType RetTy = m_DiffReq->getReturnType();
  if (RetTy->isVoidType()) {
    if (retVal_dx && retVal_dx->HasSideEffects(m_Context))
      addToCurrentBlock(retVal_dx);
    return StmtDiff(BuildReturnStmt(retVal));
  }
  bool hadRetValDx = retVal_dx != nullptr;
  bool invalidNonConstRef = RetTy->isLValueReferenceType() &&
                            !RetTy.getNonReferenceType().isConstQualified() &&
                            (!retVal_dx || !retVal_dx->isLValue());
  if (!retVal_dx || invalidNonConstRef) {
    retVal_dx = zeroReturnTangent(RS->getRetValue(), RetTy);
    if (!retVal_dx)
      return {};
  }
  if (!m_Context.hasSameUnqualifiedType(retVal->getType(), RetTy)) {
    // Check if implicit cast would work.
    // Add a cast to the return type.
    TypeSourceInfo* TSI = m_Context.getTrivialTypeSourceInfo(RetTy);
    retVal = m_Sema
                 .BuildCStyleCastExpr(RS->getBeginLoc(), TSI, RS->getEndLoc(),
                                      BuildParens(retVal))
                 .get();
    if (hadRetValDx || !isa<InitListExpr>(retVal_dx))
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
