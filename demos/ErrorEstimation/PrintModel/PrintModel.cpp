// For information on how to run this demo, please take a look at the README.
#include <iostream>

#include "PrintModel.h"
#include "clad/Differentiator/CladUtils.h"

// Here we use the BuildOp function provided by clad to build a multiplication
// expression that clad can generate code for.
clang::Expr* PrintModel::AssignError(clad::StmtDiff refExpr,
                                      const std::string& name) {
  // Next, build a llvm vector-like container to store the parameters
  // of the function call.
  llvm::SmallVector<clang::Expr*, 3> params{refExpr.getExpr_dx(),
                                            refExpr.getExpr(),
                                            clad::utils::CreateStringLiteral(
                                              m_Context, name)};
  // Finally, build a call to std::abs
  auto funcExpr = GetFunctionCall("getErrorVal", "clad", params);
  // Return the built error expression.
  return funcExpr;
}

// We need this statement to register our model with clad. Without this, clad
// will NOT be able to resolve to our overidden functions properly.
static clad::ErrorEstimationModelRegistry::Add<
    clad::EstimationPluginHelper<PrintModel>>
    CX("printModel", "Print model for error estimation in clad");
