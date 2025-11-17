#ifndef CLAD_ESTIMATION_MODEL_H
#define CLAD_ESTIMATION_MODEL_H

#include "VisitorBase.h"

#include "clad/Differentiator/DiffPlanner.h"

#include <string>
#include <unordered_map>

namespace clang {
  class VarDecl;
  class Stmt;
  class Expr;
  class ASTContext;
} // namespace clang

namespace clad {
  class DerivativeBuilder;
  class ErrorEstimationHandler;
} // namespace clad

namespace clad {

  /// A base class for user defined error estimation models.
  class FPErrorEstimationModel : public VisitorBase {
  private:
    /// Map to keep track of the error estimate variables for each declaration
    /// reference.
    std::unordered_map<const clang::VarDecl*, clang::Expr*> m_EstimateVar;

    /// An Expr representing a custom `getErrorVal` function, if any.
    clang::Expr* m_CustomErrorFunction = nullptr;
    void LookupCustomErrorFunction();

  public:
    // FIXME: Add a proper parameter for the DiffRequest here.
    FPErrorEstimationModel(DerivativeBuilder& builder,
                           const DiffRequest& request);
    ~FPErrorEstimationModel() override;

    /// Clear the variable estimate map so that we can start afresh.
    void clearEstimationVariables() { m_EstimateVar.clear(); }

    // FIXME: This is a dummy override needed because Derive is abstract.
    DerivativeAndOverload Derive() override { return {}; }

    /// The function to build the error expression of a
    /// specific estimation model. The error expression is returned in the form
    /// of a clang::Expr.
    /// \param[in] refExpr The reference of the expression to which the error
    /// has to be assigned, this is a StmtDiff type hence one can use getExpr()
    /// to get the unmodified expression and getExpr_dx() to get the absolute
    /// derivative of the same.
    /// \param [in] name Name of the variable being analysed.
    ///
    /// \returns The error expression of the input value.
    // Return an expression of the following kind:
    // std::abs(dfdx * delta_x * Em)
    clang::Expr* AssignError(StmtDiff refExpr, const std::string& name);

    friend class ErrorEstimationHandler;
  };
} // namespace clad

#endif // CLAD_ESTIMATION_MODEL_H
