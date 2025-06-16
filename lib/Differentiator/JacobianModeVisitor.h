#ifndef CLAD_DIFFERENTIATOR_JACOBIANMODEVISITOR_H
#define CLAD_DIFFERENTIATOR_JACOBIANMODEVISITOR_H

#include "clad/Differentiator/DerivativeBuilder.h"
#include "clad/Differentiator/VectorPushForwardModeVisitor.h"

#include "clang/AST/Type.h"

namespace clad {
class JacobianModeVisitor : public VectorPushForwardModeVisitor {

public:
  JacobianModeVisitor(DerivativeBuilder& builder, const DiffRequest& request);

  DerivativeAndOverload Derive() override;

  StmtDiff VisitReturnStmt(const clang::ReturnStmt* RS) override;
};
} // end namespace clad

#endif // CLAD_DIFFERENTIATOR_JACOBIANMODEVISITOR_H
