#ifndef CLAD_DIFFERENTIATOR_JACOBIANMODEVISITOR_H
#define CLAD_DIFFERENTIATOR_JACOBIANMODEVISITOR_H

#include "clad/Differentiator/VectorPushForwardModeVisitor.h"

namespace clad {
class JacobianModeVisitor : public VectorPushForwardModeVisitor {

public:
  JacobianModeVisitor(DerivativeBuilder& builder, const DiffRequest& request);

  DerivativeAndOverload DeriveJacobian();

  clang::QualType getParamAdjointType(clang::QualType T);

  StmtDiff VisitReturnStmt(const clang::ReturnStmt* RS) override;
};
} // end namespace clad

#endif // CLAD_DIFFERENTIATOR_JACOBIANMODEVISITOR_H
