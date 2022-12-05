#ifndef CLAD_ESTIMATION_MODEL_H
#define CLAD_ESTIMATION_MODEL_H

#include "VisitorBase.h"

#include "llvm/Support/Registry.h"

#include <unordered_map>

namespace clang {
  class VarDecl;
  class Stmt;
  class Expr;
  class ASTContext;
} // namespace clang

namespace clad {
  class DerivativeBuilder;
} // namespace clad

namespace clad {

  /// A base class for user defined error estimation models.
  class FPErrorEstimationModel : public VisitorBase {
  protected:
    /// Map to keep track of the error estimate variables for each declaration
    /// reference.
    std::unordered_map<const clang::VarDecl*, clang::Expr*> m_EstimateVar;

  public:
    FPErrorEstimationModel(DerivativeBuilder& builder) : VisitorBase(builder) {}
    virtual ~FPErrorEstimationModel();

    /// Clear the variable estimate map so that we can start afresh.
    void clearEstimationVariables() { m_EstimateVar.clear(); }

    /// Check if a variable is registered for estimation.
    ///
    /// \param[in] VD The variable to check.
    ///
    /// \returns The delta expression of the variable if it is registered,
    /// nullptr otherwise.
    clang::Expr* IsVariableRegistered(const clang::VarDecl* VD);

    /// Track the variable declaration and utilize it in error
    /// estimation.
    ///
    /// \param[in] VD The declaration to track.
    void AddVarToEstimate(clang::VarDecl* VD, clang::Expr* VDRef);

    /// Helper to build a function call expression.
    ///
    /// \param[in] funcName The name of the function to build the expression
    /// for.
    /// \param[in] nmspace The name of the namespace for the function,
    /// currently does not support nested namespaces.
    /// \param[in] callArgs A vector of \c clang::Expr of all the parameters
    /// to the function call.
    ///
    /// \return The function call expression that can be used to emit into
    /// code.
    clang::Expr* GetFunctionCall(std::string funcName, std::string nmspace,
                                 llvm::SmallVectorImpl<clang::Expr*>& callArgs);

    /// User overridden function to return the error expression of a
    /// specific estimation model. The error expression is returned in the form
    /// of a clang::Expr, the user may use BuildOp() to build the final
    /// expression. An example of a possible override is:
    ///
    /// \n \code
    /// clang::Expr*
    /// AssignError(clad::StmtDiff* refExpr, const std::string& name) {
    ///   return BuildOp(BO_Mul, refExpr->getExpr_dx(), refExpr->getExpr());
    /// }
    /// \endcode
    /// \n The above returns the expression: drefExpr * refExpr. For more
    /// examples refer the TaylorApprox class.
    ///
    /// \param[in] refExpr The reference of the expression to which the error
    /// has to be assigned, this is a StmtDiff type hence one can use getExpr()
    /// to get the unmodified expression and getExpr_dx() to get the absolute
    /// derivative of the same.
    /// \param [in] name Name of the variable being analysed.
    ///
    /// \returns The error expression of the input value.
    virtual clang::Expr* AssignError(StmtDiff refExpr,
                                     const std::string& name) = 0;

    /// Initializes errors for '_delta_' statements.
    /// This function returns the initial error assignment. Similar to
    /// AssignError, however, this function is only called during declaration of
    /// variables. This function is separate from AssignError to keep
    /// implementation of different estimation models more flexible.
    ///
    /// The default definition is as follows:
    /// \n \code
    /// clang::Expr* SetError(clang::VarDecl* declStmt) {
    ///      return nullptr;
    /// }
    /// \endcode
    /// The above will return a 0 expression to be assigned to the '_delta_'
    /// declaration of input decl.
    ///
    /// \param[in] decl The declaration to which the error has to be assigned.
    ///
    /// \returns The error expression for declaration statements.
    virtual clang::Expr* SetError(clang::VarDecl* decl);

    /// Calculate aggregate error from m_EstimateVar.
    ///
    /// \returns the final error estimation statement.
    clang::Expr* CalculateAggregateError();

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
    /// \returns A reference to the custom class wrapped in the
    /// FPErrorEstimationModel class.
    virtual std::unique_ptr<FPErrorEstimationModel>
    InstantiateCustomModel(DerivativeBuilder& builder) = 0;
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
    InstantiateCustomModel(DerivativeBuilder& builder) override {
      return std::unique_ptr<FPErrorEstimationModel>(new CustomClass(builder));
    }
  };

  /// Example class for taylor series approximation based error estimation.
  class TaylorApprox : public FPErrorEstimationModel {
  public:
    TaylorApprox(DerivativeBuilder& builder)
        : FPErrorEstimationModel(builder) {}
    // Return an expression of the following kind:
    // std::abs(dfdx * delta_x * Em)
    clang::Expr* AssignError(StmtDiff refExpr,
                             const std::string& name) override;
  };

  /// Register any custom error estimation model a user provides
  using ErrorEstimationModelRegistry = llvm::Registry<EstimationPlugin>;
} // namespace clad

#endif // CLAD_ESTIMATION_MODEL_H
