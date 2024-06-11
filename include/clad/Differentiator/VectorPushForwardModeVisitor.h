#ifndef CLAD_DIFFERENTIATOR_VECTORPUSHFORWARDMODEVISITOR_H
#define CLAD_DIFFERENTIATOR_VECTORPUSHFORWARDMODEVISITOR_H

#include "PushForwardModeVisitor.h"
#include "VectorForwardModeVisitor.h"

namespace clad {
class VectorPushForwardModeVisitor : public VectorForwardModeVisitor {

public:
  VectorPushForwardModeVisitor(DerivativeBuilder& builder,
                               const DiffRequest& request);
  ~VectorPushForwardModeVisitor() override;

  void ExecuteInsidePushforwardFunctionBlock() override;

  StmtDiff VisitReturnStmt(const clang::ReturnStmt* RS) override;
};
} // end namespace clad

#endif // CLAD_DIFFERENTIATOR_VECTORPUSHFORWARDMODEVISITOR_H
