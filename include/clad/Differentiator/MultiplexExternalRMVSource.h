#ifndef MULTIPLEX_EXTERNAL_RMV_SOURCE_H
#define MULTIPLEX_EXTERNAL_RMV_SOURCE_H

#include "clad/Differentiator/ExternalRMVSource.h"

#include "llvm/ADT/SmallVector.h"

namespace clad {
struct DiffRequest;

// is `ExternalRMVSourceMultiplexer` a better name for the class?
/// Manages multiple external RMV sources.
class MultiplexExternalRMVSource : public ExternalRMVSource {
private:
  llvm::SmallVector<ExternalRMVSource*, 4> m_Sources;

public:
  MultiplexExternalRMVSource() = default;
  virtual ~MultiplexExternalRMVSource();
  /// Adds `source` to the sequence of external RMV sources managed by this
  /// multiplexer.
  void AddSource(ExternalRMVSource& source);
  void InitialiseRMV(ReverseModeVisitor& RMV) override;
  void ForgetRMV() override;

  void ActOnStartOfDerive() override;
  void ActOnEndOfDerive() override;
  void ActAfterParsingDiffArgs(const DiffRequest& request,
                               DiffParams& args) override;
  void ActBeforeCreatingDerivedFnParamTypes(unsigned& numExtraParams) override;
  void ActAfterCreatingDerivedFnParamTypes(
      llvm::SmallVectorImpl<clang::QualType>& paramTypes) override;
  void ActAfterCreatingDerivedFnParams(
      llvm::SmallVectorImpl<clang::ParmVarDecl*>& params) override;
  void ActBeforeCreatingDerivedFnScope() override;
  void ActAfterCreatingDerivedFnScope() override;
  void ActBeforeCreatingDerivedFnBodyScope() override;
  void ActOnStartOfDerivedFnBody(const DiffRequest& request) override;
  void ActOnEndOfDerivedFnBody() override;
  void ActBeforeDifferentiatingStmtInVisitCompoundStmt() override;
  void ActAfterProcessingStmtInVisitCompoundStmt() override;
  void ActBeforeDifferentiatingSingleStmtBranchInVisitIfStmt() override;
  void ActBeforeFinalisingVisitBranchSingleStmtInIfVisitStmt() override;
  void ActBeforeDifferentiatingLoopInitStmt() override;
  void ActBeforeDifferentiatingSingleStmtLoopBody() override;
  void ActAfterProcessingSingleStmtBodyInVisitForLoop() override;
  void ActBeforeFinalisingVisitReturnStmt(StmtDiff& retExprDiff) override;
  void ActBeforeFinalizingVisitCallExpr(
      const clang::CallExpr*& CE, clang::Expr*& OverloadedDerivedFn,
      llvm::SmallVectorImpl<clang::Expr*>& derivedCallArgs,
      llvm::SmallVectorImpl<clang::VarDecl*>& ArgResultDecls,
      bool asGrad) override;
  void ActBeforeFinalisingPostIncDecOp(StmtDiff& diff) override;
  void ActAfterCloningLHSOfAssignOp(clang::Expr*&, clang::Expr*&,
                                    clang::BinaryOperatorKind& opCode) override;
  void ActBeforeFinalisingAssignOp(clang::Expr*&, clang::Expr*&) override;
  void ActOnStartOfDifferentiateSingleStmt() override;
  void ActBeforeFinalizingDifferentiateSingleStmt(const direction& d) override;
  void ActBeforeFinalizingDifferentiateSingleExpr(const direction& d) override;
  void ActBeforeDifferentiatingCallExpr(
      llvm::SmallVectorImpl<clang::Expr*>& pullbackArgs,
      llvm::SmallVectorImpl<clang::DeclStmt*>& ArgDecls,
      bool hasAssignee) override;
  void ActBeforeFinalizingVisitDeclStmt(
      llvm::SmallVectorImpl<clang::Decl*>& decls,
      llvm::SmallVectorImpl<clang::Decl*>& declsDiff) override;
};
} // namespace clad

#endif
