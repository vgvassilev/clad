#ifndef CLAD_EXTERNAL_RMV_SOURCE_H
#define CLAD_EXTERNAL_RMV_SOURCE_H

#include "clad/Differentiator/ParseDiffArgsTypes.h"
#include "clad/Differentiator/ReverseModeVisitorDirectionKinds.h"
#include "clad/Differentiator/ReverseModeVisitor.h"

#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/OperationKinds.h"
#include "clang/AST/Type.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace clad {

struct DiffRequest;
class StmtDiff;
class VarDeclDiff;

using direction = rmv::direction;

// FIXME: We should find the common denominator of functionality for all
// visitors (Forward and Reverse) and use this as a common way to listen to
// what they do. Therefore, ideally, in the future, this class should not rely
// on `ReverseModeVisitor`
/// An abstract interface that should be implemented by external sources
/// that provide additional behaviour, in the form of callbacks at crucial
/// locations, to the reverse mode visitor.
///
/// External sources should be attached to the reverse mode visitor object by
/// using `ReverseModeVisitor::AddExternalSource`.
class ExternalRMVSource {
public:
  ExternalRMVSource() = default;

  virtual ~ExternalRMVSource() = default;

  /// Initialise the external source with the ReverseModeVisitor object.
  virtual void InitialiseRMV(ReverseModeVisitor& RMV) {}

  /// Informs the external source that the associated `ReverseModeVisitor`
  /// object is no longer available.
  virtual void ForgetRMV(){};

  /// This is called at the beginning of the `ReverseModeVisitor::Derive`
  /// function.
  virtual void ActOnStartOfDerive() {}

  /// This is called at the end of the `ReverseModeVisitor::Derive`
  /// function.
  virtual void ActOnEndOfDerive() {}

  /// This is called just after differentiation arguments are parsed
  /// in `ReverseModeVisitor::Derive`.
  ///
  ///\param[in] request differentiation request
  ///\param[in] args differentiation args
  virtual void ActAfterParsingDiffArgs(const DiffRequest& request,
                                       DiffParams& args) {}

  /// This is called just before creating derived function parameter types.
  virtual void ActBeforeCreatingDerivedFnParamTypes(unsigned& numExtraParam) {}

  /// This is called just after creating derived function parameter types.
  ///
  /// \param paramTypes sequence container containing derived function
  /// parameter types.
  virtual void ActAfterCreatingDerivedFnParamTypes(
      llvm::SmallVectorImpl<clang::QualType>& paramTypes) {}
  virtual void ActAfterCreatingDerivedFnParams(
      llvm::SmallVectorImpl<clang::ParmVarDecl*>& params) {}

  /// This is called just before the scope is created for the derived
  /// function.
  virtual void ActBeforeCreatingDerivedFnScope() {}

  /// This is called just after the scope for the derived function is
  /// created.
  virtual void ActAfterCreatingDerivedFnScope() {}

  /// This is called just before the scope is created for the derived
  /// function body.
  virtual void ActBeforeCreatingDerivedFnBodyScope() {}

  /// This is called at the beginning of the derived function body.
  /// \param request differentiation request
  virtual void ActOnStartOfDerivedFnBody(const DiffRequest& request) {}

  /// This is called at the end of the derived function body.
  virtual void ActOnEndOfDerivedFnBody() {}

  /// This is called just before differentiating each statement in the
  /// `VisitCompoundStmt` method.
  virtual void ActBeforeDifferentiatingStmtInVisitCompoundStmt() {}

  /// This is called after differentiating and processing each statement in
  /// the `ViistCompoundStmt` method.
  virtual void ActAfterProcessingStmtInVisitCompoundStmt() {}

  /// This is called just before differentiating if-branch body statement
  /// that is not contained in a compound statement.
  virtual void ActBeforeDifferentiatingSingleStmtBranchInVisitIfStmt() {}

  /// This is called just before finalising processing of Single statement
  /// branch in `VisitBranch` lambda in
  virtual void ActBeforeFinalisingVisitBranchSingleStmtInIfVisitStmt() {}

  /// This is called just before differentiating init statement of loops.
  virtual void ActBeforeDifferentiatingLoopInitStmt() {}

  /// This is called just before differentiating loop body that is not
  /// contained in the compound statement.
  virtual void ActBeforeDifferentiatingSingleStmtLoopBody() {}

  /// This is called just after processing single statement for loop body.
  virtual void ActAfterProcessingSingleStmtBodyInVisitForLoop() {}

  /// This is called just before finalising `VisitReturnStmt`.
  virtual void ActBeforeFinalisingVisitReturnStmt(StmtDiff& retExprDiff) {}

  /// This ic called just before finalising `VisitCallExpr`.
  ///
  /// \param CE call expression that is being visited.
  /// \param CallArgs
  /// \param ArgResultDecls
  virtual void ActBeforeFinalizingVisitCallExpr(
      const clang::CallExpr*& CE, clang::Expr*& OverloadedDerivedFn,
      llvm::SmallVectorImpl<clang::Expr*>& derivedCallArgs,
      llvm::SmallVectorImpl<clang::VarDecl*>& ArgResultDecls, bool asGrad) {}

  /// This is called just before finalising processing of post and pre
  /// increment and decrement operations.
  virtual void ActBeforeFinalisingPostIncDecOp(StmtDiff& diff){};

  /// This is called just after cloning of LHS assignment operation.
  virtual void ActAfterCloningLHSOfAssignOp(clang::Expr*&, clang::Expr*&,
                                            clang::BinaryOperatorKind& opCode) {
  }

  /// This is called just after finaising processing of assignment operator.
  virtual void ActBeforeFinalisingAssignOp(clang::Expr*&, clang::Expr*&){};

  /// This is called at that beginning of
  /// `ReverseModeVisitor::DifferentiateSingleStmt`.
  virtual void ActOnStartOfDifferentiateSingleStmt(){};

  /// This is called just before finalising
  /// `ReverseModeVisitor::DifferentiateSingleStmt`.
  virtual void ActBeforeFinalizingDifferentiateSingleStmt(const direction& d) {}

  /// This is called just before finalising
  /// `ReverseModeVisitor::DifferentiateSingleExpr`.
  virtual void ActBeforeFinalizingDifferentiateSingleExpr(const direction& d) {}

  virtual void ActBeforeDifferentiatingCallExpr(
      llvm::SmallVectorImpl<clang::Expr*>& pullbackArgs,
      llvm::SmallVectorImpl<clang::DeclStmt*>& ArgDecls, bool hasAssignee) {}

  virtual void ActBeforeFinalizingVisitDeclStmt(
      llvm::SmallVectorImpl<clang::Decl*>& decls,
      llvm::SmallVectorImpl<clang::Decl*>& declsDiff) {}
};
} // namespace clad
#endif