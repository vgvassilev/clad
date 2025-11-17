#include "clad/Differentiator/EstimationModel.h"

#include "clad/Differentiator/CladUtils.h"
#include "clad/Differentiator/DerivativeBuilder.h"
#include "clad/Differentiator/DiffPlanner.h"
#include "clad/Differentiator/VisitorBase.h"

#include "clang/AST/Decl.h"
#include "clang/AST/DeclarationName.h"
#include "clang/AST/Expr.h"
#include "clang/AST/OperationKinds.h"
#include "clang/Sema/DeclSpec.h"
#include "clang/Sema/Lookup.h"

#include "llvm/ADT/SmallVector.h"

#include <string>
using namespace clang;

namespace clad {

FPErrorEstimationModel::FPErrorEstimationModel(DerivativeBuilder& builder,
                                               const DiffRequest& request)
    : VisitorBase(builder, request) {
  LookupCustomErrorFunction();
}

  FPErrorEstimationModel::~FPErrorEstimationModel() {}

  void FPErrorEstimationModel::LookupCustomErrorFunction() {
    CXXScopeSpec SS;
    NamespaceDecl* cladNS =
        utils::LookupNSD(m_Sema, "clad", /*shouldExist=*/true);
    SS.Extend(m_Context, cladNS, noLoc, noLoc);
    IdentifierInfo* II = &m_Context.Idents.get("getErrorVal");
    DeclarationNameInfo DNInfo(DeclarationName(II),
                               utils::GetValidSLoc(m_Sema));
    LookupResult R(m_Sema, DNInfo, Sema::LookupOrdinaryName);
    m_Sema.LookupQualifiedName(R, cladNS);
    if (R.empty())
      return;
    m_CustomErrorFunction =
        m_Sema.BuildDeclarationNameExpr(SS, R, /*ADL*/ false).get();
  }

  Expr* FPErrorEstimationModel::AssignError(StmtDiff refExpr,
                                            const std::string& varName) {
    if (m_CustomErrorFunction) {
      llvm::SmallVector<clang::Expr*, 3> callParams{
          refExpr.getExpr_dx(), refExpr.getExpr(),
          clad::utils::CreateStringLiteral(m_Context, varName)};
      return m_Sema
          .ActOnCallExpr(getCurrentScope(), m_CustomErrorFunction, noLoc,
                         callParams, noLoc)
          .get();
    }
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