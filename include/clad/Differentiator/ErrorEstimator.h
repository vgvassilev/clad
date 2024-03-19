#ifndef CLAD_ERROR_ESTIMATOR_H
#define CLAD_ERROR_ESTIMATOR_H

#include "EstimationModel.h"
#include "clad/Differentiator/ExternalRMVSource.h"
#include "clad/Differentiator/ReverseModeVisitorDirectionKinds.h"

#include "clang/AST/OperationKinds.h"

#include <stack>

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
class ErrorEstimationHandler : public ExternalRMVSource {
  // FIXME: Find a good way to modularize using declarations that are used in
  // multiple header files. 
  // `Stmts` is originally defined in `VisitorBase`.
  using Stmts = llvm::SmallVector<clang::Stmt*, 16>;
  /// Reference to the final error parameter in the augumented target
  /// function.
  clang::Expr* m_FinalError;
  /// Reference to the return error expression.
  clang::Expr* m_RetErrorExpr;
  /// An instance of the custom error estimation model to be used.
  FPErrorEstimationModel* m_EstModel; // We do not own this.
  /// A set of assignments resulting for declaration statments.
  Stmts m_ForwardReplStmts;
  /// A vector to keep track of error statements for delayed emission.
  Stmts m_ReverseErrorStmts;
  /// The index expression for emitting final errors for input param errors.
  clang::Expr* m_IdxExpr;
  /// A map from var decls to their size variables (e.g. `var_size`).
  std::unordered_map<const clang::VarDecl*, clang::Expr*> m_ArrSizes;
  /// An expression to match nested function call errors with their
  /// assignee (if any exists).
  clang::Expr* m_NestedFuncError = nullptr;

  std::stack<bool> m_ShouldEmit;
  ReverseModeVisitor* m_RMV;
  llvm::SmallVectorImpl<clang::QualType>* m_ParamTypes = nullptr;
  llvm::SmallVectorImpl<clang::ParmVarDecl*>* m_Params = nullptr;

public:
  using direction = rmv::direction;
  ErrorEstimationHandler()
      : m_FinalError(nullptr), m_RetErrorExpr(nullptr), m_EstModel(nullptr),
        m_IdxExpr(nullptr) {}
  ~ErrorEstimationHandler() override = default;

  /// Function to set the error estimation model currently in use.
  ///
  /// \param[in] estModel The error estimation model, can be either
  /// an in-built one (TaylorApprox) or one provided by the user.
  void SetErrorEstimationModel(FPErrorEstimationModel* estModel);

  /// \param[in] finErrExpr The final error expression.
  void SetFinalErrorExpr(clang::Expr* finErrExpr) { m_FinalError = finErrExpr; }

  /// Function to build the error statement corresponding
  /// to the function's return statement.
  void BuildReturnErrorStmt();

  /// Function to emit error statements into the derivative body.
  ///
  /// \param[in] errorExpr The error expression (LHS) of the variable.
  /// \param[in] addToTheFront A flag to decide whether the error stmts
  /// should be added to the beginning of the block or the current position.
  void AddErrorStmtToBlock(clang::Expr* errorExpr, bool addToTheFront = true);

  /// Emit the error estimation related statements that were saved to be
  /// emitted at later points into specific blocks.
  ///
  /// \param[in] d The direction of the block in which to emit the
  /// statements.
  void EmitErrorEstimationStmts(direction d = direction::forward);

  /// We should save the return value if it is an arithmetic expression,
  /// since we also need to take into account the error in that expression.
  ///
  /// \param[in] retExpr The return expression.
  /// \param[in] retDeclRefExpr The temporary value in which the return
  /// expression is stored.
  void SaveReturnExpr(clang::Expr* retExpr);

  /// Emit the error for parameters of nested functions.
  ///
  /// \param[in] fnDecl The function declaration of the nested function.
  /// \param[in] CallArgs The orignal call arguments of the function call.
  /// \param[in] ArgResultDecls The differentiated call arguments.
  /// \param[in] numArgs The number of call args.
  void
  EmitNestedFunctionParamError(clang::FunctionDecl* fnDecl,
                               llvm::SmallVectorImpl<clang::Expr*>& CallArgs,
                               llvm::SmallVectorImpl<clang::Expr*>& ArgResult,
                               size_t numArgs);

  /// Checks if a variable should be considered in error estimation.
  ///
  /// \param[in] VD The variable declaration.
  ///
  /// \returns true if the variable should be considered, false otherwise.
  bool ShouldEstimateErrorFor(clang::VarDecl* VD);

  /// Get the underlying DeclRefExpr type it it exists.
  ///
  /// \param[in] expr The expression whose DeclRefExpr is requested.
  ///
  /// \returns The DeclRefExpr of input or null.
  clang::DeclRefExpr* GetUnderlyingDeclRefOrNull(clang::Expr* expr);

  /// An abstraction of the error estimation model's AssignError.
  ///
  /// \param[in] val The variable to get the error for.
  /// \param[in] valDiff The derivative of the variable 'var'
  /// \param[in] varName Name of the variable to get the error for.
  ///
  /// \returns The error in the variable 'var'.
  clang::Expr* GetError(clang::Expr* var, clang::Expr* varDiff,
                        const std::string& varName) {
    return m_EstModel->AssignError({var, varDiff}, varName);
  }

  /// This function adds the final error and the other parameter errors to the
  /// forward block.
  ///
  /// \param[in] params A vector of the parameter decls.
  /// \param[in] numParams The number of orignal parameters.
  void EmitFinalErrorStmts(llvm::SmallVectorImpl<clang::ParmVarDecl*>& params,
                           unsigned numParams);

  /// This function emits the error in unary operations.
  ///
  /// \param[in] var The variable to emit the error for.
  /// \param[in] isInsideLoop A flag to keep track of if we are inside a
  /// loop.
  void EmitUnaryOpErrorStmts(StmtDiff var, bool isInsideLoop);

  /// This function emits the error in a binary operation.
  ///
  /// \param[in] LExpr The LHS of the operation.
  /// \param[in] oldValue The derivative of the target function with respect
  /// to the LHS.
  /// \param[in] deltaVar The delta value of the LHS.
  /// \param[in] isInsideLoop A flag to keep track of if we are inside a
  /// loop.
  void EmitBinaryOpErrorStmts(clang::Expr* LExpr, clang::Expr* oldValue);

  /// This function emits the error in declaration statements.
  ///
  /// \param[in] VDDiff The variable declaration to calculate the error in.
  /// \param[in] isInsideLoop A flag to keep track of if we are inside a
  /// loop.
  void EmitDeclErrorStmts(VarDeclDiff VDDiff, bool isInsideLoop);

  /// This function returns the size expression for a given variable
  /// (`var.size()` for clad::array/clad::array_ref
  /// or `var_size` for array/pointer types)
  clang::Expr* getSizeExpr(const clang::VarDecl* VD);

  void InitialiseRMV(ReverseModeVisitor& RMV) override;
  void ForgetRMV() override;
  void ActBeforeCreatingDerivedFnParamTypes(unsigned&) override;
  void ActAfterCreatingDerivedFnParamTypes(
      llvm::SmallVectorImpl<clang::QualType>& paramTypes) override;
  void ActAfterCreatingDerivedFnParams(
      llvm::SmallVectorImpl<clang::ParmVarDecl*>& params) override;
  void ActBeforeCreatingDerivedFnBodyScope() override;
  void ActOnEndOfDerivedFnBody() override;
  void ActBeforeDifferentiatingStmtInVisitCompoundStmt() override;
  void ActAfterProcessingStmtInVisitCompoundStmt() override;
  void
  ActAfterProcessingArraySubscriptExpr(const clang::Expr* revArrSub) override;
  void ActBeforeDifferentiatingSingleStmtBranchInVisitIfStmt() override;
  void ActBeforeFinalizingVisitBranchSingleStmtInIfVisitStmt() override;
  void ActBeforeDifferentiatingLoopInitStmt() override;
  void ActBeforeDifferentiatingSingleStmtLoopBody() override;
  void ActAfterProcessingSingleStmtBodyInVisitForLoop() override;
  void ActBeforeFinalizingVisitReturnStmt(StmtDiff& retExprDiff) override;
  void ActBeforeFinalizingPostIncDecOp(StmtDiff& diff) override;
  void ActBeforeFinalizingVisitCallExpr(
      const clang::CallExpr*& CE, clang::Expr*& fnDecl,
      llvm::SmallVectorImpl<clang::Expr*>& derivedCallArgs,
      llvm::SmallVectorImpl<clang::Expr*>& ArgResult, bool asGrad) override;
  void ActBeforeFinalizingAssignOp(clang::Expr*&, clang::Expr*&, clang::Expr*&,
                                   clang::BinaryOperator::Opcode&) override;
  void ActBeforeFinalizingDifferentiateSingleStmt(const direction& d) override;
  void ActBeforeFinalizingDifferentiateSingleExpr(const direction& d) override;
  void ActBeforeDifferentiatingCallExpr(
      llvm::SmallVectorImpl<clang::Expr*>& pullbackArgs,
      llvm::SmallVectorImpl<clang::Stmt*>& ArgDecls, bool hasAssignee) override;
  void ActBeforeFinalizingVisitDeclStmt(
      llvm::SmallVectorImpl<clang::Decl*>& decls,
      llvm::SmallVectorImpl<clang::Decl*>& declsDiff) override;
};
} // namespace clad

#endif // CLAD_ERROR_ESTIMATOR_H
