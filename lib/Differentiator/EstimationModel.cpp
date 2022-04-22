#include "clad/Differentiator/EstimationModel.h"
#include "clad/Differentiator/DerivativeBuilder.h"

#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/OperationKinds.h"

#include "llvm/Support/Registry.h"
using namespace clang;

namespace clad {

  FPErrorEstimationModel::~FPErrorEstimationModel() {}

  Expr* FPErrorEstimationModel::IsVariableRegistered(const VarDecl* VD) {
    auto it = m_EstimateVar.find(VD);
    if (it != m_EstimateVar.end())
      return it->second;
    return nullptr;
  }

  void FPErrorEstimationModel::AddVarToEstimate(VarDecl* VD, Expr* VDRef) {
    m_EstimateVar.emplace(VD, VDRef);
  }

  // FIXME: Maybe this should be left to the user too.
  Expr* FPErrorEstimationModel::CalculateAggregateError() {
    Expr* addExpr = nullptr;
    // Loop over all the error variables and form the final error expression of
    // the form... _final_error = _delta_var + _delta_var1 +...
    for (auto var : m_EstimateVar) {
      // Errors through array subscript expressions are already captured
      // to avoid having long add expression at the end and to only add
      // the values to the final error that have a non zero delta.
      if (isArrayOrPointerType(var.first->getType()))
        continue;

      if (!addExpr) {
        addExpr = var.second;
        continue;
      }
      addExpr = BuildOp(BO_Add, addExpr, var.second);
    }
    // Return an expression that can be directly assigned to final error.
    return addExpr;
  }

  // Return nullptr here, this is interpreted as 0 internally.
  Expr* FPErrorEstimationModel::SetError(VarDecl* declStmt) { 
    return nullptr; 
  }

  Expr* TaylorApprox::AssignError(StmtDiff refExpr) {
    // Get the machine epsilon value.
    double val = std::numeric_limits<float>::epsilon();
    // Convert it into a floating point literal clang::Expr.
    auto epsExpr = FloatingLiteral::Create(m_Context, llvm::APFloat(val), true,
                                           m_Context.DoubleTy, noLoc);
    // Build the final operations.
    // Here, we first build a multiplication operation for the following:
    // refExpr * <--floating point literal (i.e. machine dependent constant)-->
    // Build another multiplication operation with above and the derivative
    // value.
    return BuildOp(BO_Mul, refExpr.getExpr_dx(),
                   BuildOp(BO_Mul, refExpr.getExpr(), epsExpr));
  }

} // namespace clad

// instantiate our error estimation model registry so that we can register
// custom models passed by users as a shared lib
LLVM_INSTANTIATE_REGISTRY(clad::ErrorEstimationModelRegistry)