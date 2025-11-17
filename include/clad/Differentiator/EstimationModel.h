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
  protected:
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
    virtual clang::Expr* AssignError(StmtDiff refExpr, const std::string& name);

    friend class ErrorEstimationHandler;
  };

  /// A base class to build the error estimation registry over.
  class EstimationPlugin {
  public:
    virtual ~EstimationPlugin() {}
    /// Function that will return the instance of the user registered
    /// custom model.
    /// \param[in] builder A build instance to pass to the custom model
    /// constructor.
    /// \param[in] request The differentiation configuration passed to the
    /// custom model
    /// \returns A reference to the custom class wrapped in the
    /// FPErrorEstimationModel class.
    virtual std::unique_ptr<FPErrorEstimationModel>
    InstantiateCustomModel(DerivativeBuilder& builder,
                           const DiffRequest& request) = 0;
  };

  /// A class used to register custom plugins.
  ///
  /// \tparam The custom user class.
  template <typename CustomClass>
  class EstimationPluginHelper : public EstimationPlugin {
  public:
    /// Return an instance of the user defined custom class.
    ///
    /// \param[in] builder The current instance of derivative builder.
    std::unique_ptr<FPErrorEstimationModel>
    InstantiateCustomModel(DerivativeBuilder& builder,
                           const DiffRequest& request) override {
      return std::unique_ptr<FPErrorEstimationModel>(
          new CustomClass(builder, request));
    }
  };

  /// Register any custom error estimation model a user provides
  using ErrorEstimationModelRegistry = llvm::Registry<EstimationPlugin>;
} // namespace clad

#endif // CLAD_ESTIMATION_MODEL_H
