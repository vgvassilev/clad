#ifndef CLAD_DIFFERENTIATOR_JACOBIANMODEVISITOR_H
#define CLAD_DIFFERENTIATOR_JACOBIANMODEVISITOR_H

#include "clad/Differentiator/VectorPushForwardModeVisitor.h"

#include "clang/AST/Type.h"

namespace clad {
class JacobianModeVisitor : public VectorPushForwardModeVisitor {

public:
  JacobianModeVisitor(DerivativeBuilder& builder, const DiffRequest& request);

  DerivativeAndOverload DeriveJacobian();

  clang::QualType GetParameterDerivativeType(clang::QualType T) override;

  StmtDiff VisitReturnStmt(const clang::ReturnStmt* RS) override;
};
} // end namespace clad

#endif // CLAD_DIFFERENTIATOR_JACOBIANMODEVISITOR_H
