#ifndef CLAD_ERROR_ESTIMATOR_H
#define CLAD_ERROR_ESTIMATOR_H

#include "EstimationModel.h"
#include "ReverseModeVisitor.h"

namespace clang {
  class Stmt;
  class Expr;
  class Decl;
  class VarDecl;
  class ParmVarDecl;
  class FunctionDecl;
  class CompoundStmt;
} // namespace clang

namespace clad {
  /// The estimation handler which interfaces with Clad's reverse mode visitors
  /// to fetch derivatives.
  class ErrorEstimationHandler : public ReverseModeVisitor {
    /// Keeps a track of the delta error expression we shouldn't emit.
    bool m_DoNotEmitDelta = false;
    /// Reference to the final error parameter in the augumented target
    /// function.
    clang::Expr* m_FinalError;
    /// Reference to the return error expression.
    clang::Expr* m_RetErrorExpr;
    /// An instance of the custom error estimation model to be used.
    FPErrorEstimationModel* m_EstModel; // We do not own this.
    /// A set of assignments resulting for declaration statments.
    VisitorBase::Stmts m_ForwardReplStmts;
    /// A vector to keep track of error statements for delayed emission.
    VisitorBase::Stmts m_ReverseErrorStmts;
    /// The index expression for emitting final errors for input param errors.
    clang::Expr* m_IdxExpr;
    /// A set of declRefExprs for parameter value replacements.
    std::unordered_map<const clang::VarDecl*, clang::Expr*> m_ParamRepls;

  public:
    ErrorEstimationHandler(DerivativeBuilder& builder)
        : ReverseModeVisitor(builder), m_DoNotEmitDelta(false),
          m_FinalError(nullptr), m_RetErrorExpr(nullptr), m_EstModel(nullptr),
          m_IdxExpr(nullptr) {}
    ~ErrorEstimationHandler() {}

    /// Function to set the error estimation model currently in use.
    ///
    /// \param[in] estModel The error estimation model, can be either
    /// an in-built one (TaylorApprox) or one provided by the user.
    void SetErrorEstimationModel(FPErrorEstimationModel* estModel);

    /// \param[in] finErrExpr The final error expression.
    void SetFinalErrorExpr(clang::Expr* finErrExpr) {
      m_FinalError = finErrExpr;
    }

    /// \returns The final error expression so far.
    clang::Expr* GetFinalErrorExpr() { return m_FinalError; }

    /// Function to build the final error statemnt of the function. This is the
    /// last statement of any target function in error estimation and
    /// aggregates the error in all the registered variables.
    void BuildFinalErrorStmt();

    /// Function to emit error statements into the derivative body.
    ///
    /// \param[in] var The variable whose error statement we want to emit.
    /// \param[in] deltaVar The "_delta_" expression of the variable 'var'.
    /// \param[in] errorExpr The error expression (LHS) of the variable 'var'.
    /// \param[in] isInsideLoop A flag to indicate if 'val' is inside a loop.
    void AddErrorStmtToBlock(clang::Expr* var, clang::Expr* deltaVar,
                             clang::Expr* errorExpr, bool isInsideLoop = false);

    /// Emit the error estimation related statements that were saved to be
    /// emitted at later points into specific blocks.
    ///
    /// \param[in] d The direction of the block in which to emit the
    /// statements.
    void EmitErrorEstimationStmts(direction d = forward);

    /// We should save the return value if it is an arithmetic expression,
    /// since we also need to take into account the error in that expression.
    ///
    /// \param[in] retExpr The return expression.
    /// \param[in] retDeclRefExpr The temporary value in which the return
    /// expression is stored.
    void SaveReturnExpr(clang::Expr* retExpr,
                        clang::DeclRefExpr* retDeclRefExpr);

    /// Emit the error for parameters of nested functions.
    ///
    /// \param[in] fnDecl The function declaration of the nested function.
    /// \param[in] CallArgs The orignal call arguments of the function call.
    /// \param[in] ArgResultDecls The differentiated call arguments.
    /// \param[in] numArgs The number of call args.
    void EmitNestedFunctionParamError(
        clang::FunctionDecl* fnDecl,
        llvm::SmallVectorImpl<clang::Expr*>& CallArgs,
        llvm::SmallVectorImpl<clang::VarDecl*>& ArgResultDecls, size_t numArgs);

    /// Save values of registered variables so that they can be replaced
    /// properly in case of re-assignments.
    ///
    /// \param[in] val The value to save.
    /// \param[in] isInsideLoop A flag to indicate if 'val' is inside a loop.
    ///
    /// \returns The saved variable and its derivative.
    StmtDiff SaveValue(clang::Expr* val, bool isInLoop = false);

    /// Save the orignal values of the input parameters in case of
    /// re-assignments.
    ///
    /// \param[in] paramRef The DeclRefExpr of the input parameter.
    void SaveParamValue(clang::DeclRefExpr* paramRef);

    /// Register variables to be used while accumulating error.
    /// Register variable declarations so that they may be used while
    /// calculating the final error estimates. Any unregistered variables will
    /// not be considered for the final estimation.
    ///
    /// \param[in] VD The variable declaration to be registered.
    /// \param[in] toCurrentScope Add the created "_delta_" variable declaration
    /// to the current scope instead of the global scope.
    ///
    /// \returns The Variable declaration of the '_delta_' prefixed variable.
    clang::Expr* RegisterVariable(clang::VarDecl* VD,
                                  bool toCurrentScope = false);

    /// Checks if a variable can be registered for error estimation.
    ///
    /// \param[in] VD The variable declaration to be registered.
    ///
    /// \returns True if the variable can be registered, false otherwise.
    bool CanRegisterVariable(clang::VarDecl* VD);

    /// Calculate aggregate error from m_EstimateVar.
    /// Builds the final error estimation statement.
    clang::Stmt* CalculateAggregateError();

    /// Get the underlying DeclRefExpr type it it exists.
    ///
    /// \param[in] expr The expression whose DeclRefExpr is requested.
    ///
    /// \returns The DeclRefExpr of input or null.
    clang::DeclRefExpr* GetUnderlyingDeclRefOrNull(clang::Expr* expr);

    /// Get the parameter replacement (if any).
    ///
    /// \param[in] VD The parameter variable declaration to get replacement
    /// for.
    ///
    /// \returns The underlying replaced Expr.
    clang::Expr* GetParamReplacement(const clang::ParmVarDecl* VD);

    /// An abstraction of the error estimation model's AssignError.
    ///
    /// \param[in] val The variable to get the error for.
    /// \param[in] valDiff The derivative of the variable 'var'
    ///
    /// \returns The error in the variable 'var'.
    clang::Expr* GetError(clang::Expr* var, clang::Expr* varDiff) {
      return m_EstModel->AssignError({var, varDiff});
    }

    /// An abstraction of the error estimation model's IsVariableRegistered.
    ///
    /// \param[in] VD The variable declaration to check the status of.
    ///
    /// \returns the reference to the respective '_delta_' expression if the
    /// variable is registered, null otherwise.
    clang::Expr* IsRegistered(clang::VarDecl* VD) {
      return m_EstModel->IsVariableRegistered(VD);
    }

    /// This function adds the final error and the other parameter errors to the
    /// forward block.
    ///
    /// \param[in] params A vector of the parameter decls.
    /// \param[in] numParams The number of orignal parameters.
    void EmitFinalErrorStmts(llvm::SmallVectorImpl<clang::ParmVarDecl*>& params,
                             int numParams);

    /// This function emits the error in unary operations.
    ///
    /// \param[in] var The variable to emit the error for.
    /// \param[in] isInsideLoop A flag to keep track of if we are inside a
    /// loop.
    void EmitUnaryOpErrorStmts(StmtDiff var, bool isInsideLoop);

    /// This function registers all LHS declRefExpr in binary operations.
    ///
    /// \param[in] LExpr The LHS of the operation.
    /// \param[in] RExpr The RHS of the operation.
    /// \param[in] isAssign A flag to know if the current operation is a simple
    /// assignment.
    ///
    /// \returns The delta value of the input 'var'.
    clang::Expr* RegisterBinaryOpLHS(clang::Expr* LExpr, clang::Expr* RExpr,
                                     bool isAssign);

    /// This function emits the error in a binary operation.
    ///
    /// \param[in] LExpr The LHS of the operation.
    /// \param[in] oldValue The derivative of the target function with respect
    /// to the LHS.
    /// \param[in] deltaVar The delta value of the LHS.
    /// \param[in] isInsideLoop A flag to keep track of if we are inside a
    /// loop.
    void EmitBinaryOpErrorStmts(clang::Expr* LExpr, clang::Expr* oldValue,
                                clang::Expr* deltaVar, bool isInsideLoop);

    /// This function emits the error in declaration statements.
    ///
    /// \param[in] VDDiff The variable declaration to calculate the error in.
    /// \param[in] isInsideLoop A flag to keep track of if we are inside a
    /// loop.
    void EmitDeclErrorStmts(VarDeclDiff VDDiff, bool isInsideLoop);
  };

} // namespace clad

#endif // CLAD_ERROR_ESTIMATOR_H
