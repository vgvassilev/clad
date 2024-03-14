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