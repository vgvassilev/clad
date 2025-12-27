#include "clad/Differentiator/EstimationModel.h"

#include "clad/Differentiator/CladUtils.h"
#include "clad/Differentiator/DerivativeBuilder.h"
#include "clad/Differentiator/DiffPlanner.h"
#include "clad/Differentiator/VisitorBase.h"

#include "clang/AST/Decl.h"
#include "clang/AST/DeclarationName.h"
#include "clang/AST/Expr.h"
#include "clang/AST/OperationKinds.h"
#include "clang/AST/Type.h"
#include "clang/Sema/DeclSpec.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/TemplateDeduction.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallVector.h"

#include <limits>
#include <string>
using namespace clang;

namespace clad {

FPErrorEstimationModel::FPErrorEstimationModel(DerivativeBuilder& builder,
                                               const DiffRequest& request)
    : VisitorBase(builder, request) {
  LookupCustomErrorFunction();
}

void FPErrorEstimationModel::LookupCustomErrorFunction() {
  NamespaceDecl* cladNS =
      utils::LookupNSD(m_Sema, "clad", /*shouldExist=*/true);
  IdentifierInfo* II = &m_Context.Idents.get("getErrorVal");
  DeclarationNameInfo DNInfo(DeclarationName(II), utils::GetValidSLoc(m_Sema));
  LookupResult R(m_Sema, DNInfo, Sema::LookupOrdinaryName);
  m_Sema.LookupQualifiedName(R, cladNS);
  if (R.empty())
    return;

  FunctionProtoType::ExtProtoInfo EPI;
  QualType ConstCharPtr =
      m_Context.getPointerType(m_Context.getConstType(m_Context.CharTy));
  QualType DoubleTy = m_Context.DoubleTy;
  llvm::SmallVector<QualType, 3> FnTypes = {DoubleTy, DoubleTy, ConstCharPtr};
  QualType FnTy = m_Context.getFunctionType(DoubleTy, FnTypes, EPI);
  TemplateSpecCandidateSet FailedCandidates(utils::GetValidSLoc(m_Sema),
                                            /*ForTakingAddress=*/false);
  if (utils::MatchOverloadType(m_Sema, FnTy, R, FailedCandidates)) {
    // FIXME: MatchOverloadType returns an overload expr without the `clad::`
    // namespace specifier. Here, we rebuild manually.
    CXXScopeSpec SS;
    SS.Extend(m_Context, cladNS, noLoc, noLoc);
    m_CustomErrorFunction =
        m_Sema.BuildDeclarationNameExpr(SS, R, /*ADL=*/false).get();
    return;
  }

  // We did not match the found candidates. Warn and offer the user hints.
  auto errId = m_Sema.Diags.getCustomDiagID(
      DiagnosticsEngine::Error,
      "user-defined derivative error function was provided but not used; "
      "expected signature %0 does not match");
  m_Sema.Diag(m_DiffReq.Function->getLocation(), errId) << FnTy;
  FailedCandidates.NoteCandidates(m_Sema, utils::GetValidSLoc(m_Sema));
  utils::DiagnoseSignatureMismatch(m_Sema, FnTy, R);
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
  Expr* epsExpr = FloatingLiteral::Create(m_Context, llvm::APFloat(val), true,
                                          m_Context.DoubleTy, noLoc);
  // Here, we first build a multiplication operation for the following:
  // refExpr * <--floating point literal (i.e. machine dependent constant)-->
  // Build another multiplication operation with above and the derivative
  Expr* errExpr = BuildOp(BO_Mul, refExpr.getExpr_dx(),
                          BuildOp(BO_Mul, refExpr.getExpr(), epsExpr));
  // Next, build a llvm vector-like container to store the parameters
  // of the function call.
  llvm::SmallVector<Expr*, 1> params{errExpr};
  // Finally, build a call to std::abs
  Expr* absExpr = GetFunctionCall("abs", "std", params);
  // Return the built error expression.
  return absExpr;
}

} // namespace clad
