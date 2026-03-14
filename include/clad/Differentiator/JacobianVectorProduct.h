#ifndef CLAD_DIFFERENTIATOR_JACOBIAN_VECTOR_PRODUCT_H
#define CLAD_DIFFERENTIATOR_JACOBIAN_VECTOR_PRODUCT_H

#include "JacobianModeVisitor.h"
#include "clad/Differentiator/DerivativeBuilder.h"
#include <clad/Differentiator/DiffPlanner.h>

using namespace clad;

class clad::JacobianVectorProductModeVisitor : public clad::JacobianModeVisitor {
  public:
  JacobianVectorProductModeVisitor(DerivativeBuilder& builder,
                                   const DiffRequest& request);

  DerivativeAndOverload Derive() override;
  private:
  DiffRequest m_ModifiedDiffRequest;
};

#endif
