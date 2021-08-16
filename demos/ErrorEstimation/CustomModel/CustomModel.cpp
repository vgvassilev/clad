// For information on how to run this demo, please take a look at the README.

#include "CustomModel.h"

// Here we use the BuildOp function provided by clad to build a multiplication
// expression that clad can generate code for.
clang::Expr* CustomModel::AssignError(clad::StmtDiff refExpr) {
  return BuildOp(clang::BO_Mul, refExpr.getExpr_dx(), refExpr.getExpr());
}

clang::Expr* CustomModel::SetError(clang::VarDecl* decl) { return nullptr; }

// We need this statement to register our model with clad. Without this, clad
// will NOT be able to resolve to our overidden functions properly.
static clad::plugin::ErrorEstimationModelRegistry::Add<
    clad::EstimationPluginHelper<CustomModel>>
    CX("customModel", "Custom model for error estimation in clad");
