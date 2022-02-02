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

  Expr* FPErrorEstimationModel::getAsExpr(std::string expr) {
    return utils::CreateStringLiteral(m_Context, expr);
  }

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

  Expr* FPErrorEstimationModel::GetFunctionCall(
      std::string funcName, std::string nmspace,
      llvm::SmallVectorImpl<Expr*>& callArgs) {
    NamespaceDecl* NSD =
        utils::LookupNSD(m_Sema, nmspace, /*shouldExist=*/true);
    DeclContext* DC = NSD;
    CXXScopeSpec SS;
    SS.Extend(m_Context, NSD, noLoc, noLoc);

    IdentifierInfo* II = &m_Context.Idents.get(funcName);
    DeclarationName name(II);
    DeclarationNameInfo DNI(name, noLoc);
    LookupResult R(m_Sema, DNI, Sema::LookupOrdinaryName);

    if (DC)
      m_Sema.LookupQualifiedName(R, DC);
    Expr* UnresolvedLookup = nullptr;
    if (!R.empty())
      UnresolvedLookup =
          m_Sema.BuildDeclarationNameExpr(SS, R, /*ADL=*/false).get();
    llvm::MutableArrayRef<Expr*> MARargs =
        llvm::MutableArrayRef<Expr*>(callArgs);
    SourceLocation Loc;
    return m_Sema
        .ActOnCallExpr(getCurrentScope(), UnresolvedLookup, Loc, MARargs, Loc)
        .get();
  }

  Expr* TaylorApprox::AssignError(StmtDiff refExpr) {
    // Get the machine epsilon value.
    double val = std::numeric_limits<float>::epsilon();
    // Convert it into a floating point literal clang::Expr.
    auto epsExpr = FloatingLiteral::Create(m_Context, llvm::APFloat(val), true,
                                           m_Context.DoubleTy, noLoc);
    // Here, we first build a multiplication operation for the following:
    // refExpr * <--floating point literal (i.e. machine dependent constant)-->
    // Build another multiplication operation with above and the derivative
    // value.
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

  // Return the error expression of the variable we are visiting.
  void TaylorApprox::Print(std::string varName, StmtDiff refExpr, Expr* errExpr,
                           llvm::SmallVectorImpl<Expr*>& out) {
    // Return the following clang:expr:
    // Variable-name : variable-error
    out.push_back(getAsExpr(varName));
    out.push_back(getAsExpr(" : "));
    out.push_back(errExpr);
    out.push_back(getAsExpr("\n"));
  }

} // namespace clad

// instantiate our error estimation model registry so that we can register
// custom models passed by users as a shared lib
LLVM_INSTANTIATE_REGISTRY(clad::ErrorEstimationModelRegistry)