#include "clad/Differentiator/EstimationModel.h"

#include "clad/Differentiator/CladUtils.h"
#include "clad/Differentiator/DerivativeBuilder.h"

#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/OperationKinds.h"
#include "clang/Sema/Lookup.h"

#include "llvm/Support/Registry.h"
using namespace clang;

namespace clad {

  FPErrorEstimationModel::~FPErrorEstimationModel() {}

  Expr* TaylorApprox::AssignError(StmtDiff refExpr,
                                  const std::string& varName) {
    // Get the machine epsilon value.
    double val = std::numeric_limits<float>::epsilon();
    // Convert it into a floating point literal clang::Expr.
    auto epsExpr = FloatingLiteral::Create(m_Context, llvm::APFloat(val), true,
                                           m_Context.DoubleTy, noLoc);
    // Here, we first build a multiplication operation for the following:
    // refExpr * <--floating point literal (i.e. machine dependent constant)-->
    // Build another multiplication operation with above and the derivative
    auto errExpr = BuildOp(BO_Mul, refExpr.getExpr_dx(),
                           BuildOp(BO_Mul, refExpr.getExpr(), epsExpr));
    // Next, build a llvm vector-like container to store the parameters
    // of the function call.
    llvm::SmallVector<Expr*, 1> params{errExpr};
    // Finally, build a call to std::abs
    auto absExpr = GetFunctionCall("abs", "std", params);
    // Return the built error expression.
    return absExpr;
  }

} // namespace clad

// instantiate our error estimation model registry so that we can register
// custom models passed by users as a shared lib
LLVM_INSTANTIATE_REGISTRY(clad::ErrorEstimationModelRegistry)