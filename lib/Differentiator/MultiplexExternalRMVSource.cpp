#include "clad/Differentiator/MultiplexExternalRMVSource.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
namespace clad {
// void MultiplexExternalRMVSource::MultiplexExternalRMVSource() {}
void MultiplexExternalRMVSource::AddSource(ExternalRMVSource& source) {
  m_Sources.push_back(&source);
}

void MultiplexExternalRMVSource::InitialiseRMV(ReverseModeVisitor& RMV) {
  for (auto source : m_Sources) {
    source->InitialiseRMV(RMV);
  }
}

void MultiplexExternalRMVSource::ForgetRMV() {
  for (auto source : m_Sources)
    source->ForgetRMV();
}

void MultiplexExternalRMVSource::ActOnStartOfDerive() {
  for (auto source : m_Sources) {
    source->ActOnStartOfDerive();
  }
}

void MultiplexExternalRMVSource::ActOnEndOfDerive() {
  for (auto source : m_Sources) {
    source->ActOnEndOfDerive();
  }
}

void MultiplexExternalRMVSource::ActAfterParsingDiffArgs(
    const DiffRequest& request, DiffParams& args) {
  for (auto source : m_Sources) {
    source->ActAfterParsingDiffArgs(request, args);
  }
}

void MultiplexExternalRMVSource::ActBeforeCreatingDerivedFnParamTypes(
    unsigned& numExtraParams) {
  for (auto source : m_Sources) {
    source->ActBeforeCreatingDerivedFnParamTypes(numExtraParams);
  }
};
void MultiplexExternalRMVSource::ActAfterCreatingDerivedFnParamTypes(
    llvm::SmallVectorImpl<clang::QualType>& paramTypes) {
  for (auto source : m_Sources) {
    source->ActAfterCreatingDerivedFnParamTypes(paramTypes);
  }
}
void MultiplexExternalRMVSource::ActAfterCreatingDerivedFnParams(
    llvm::SmallVectorImpl<clang::ParmVarDecl*>& params) {
  // llvm::errs() << "Reaching multiplexer ActAfterCreatingDerivedFnParams\n";
  for (auto source : m_Sources) {
    source->ActAfterCreatingDerivedFnParams(params);
  }
}

void MultiplexExternalRMVSource::ActBeforeCreatingDerivedFnScope() {
  for (auto source : m_Sources) {
    source->ActBeforeCreatingDerivedFnScope();
  }
}

void MultiplexExternalRMVSource::ActAfterCreatingDerivedFnScope() {
  for (auto source : m_Sources) {
    source->ActAfterCreatingDerivedFnScope();
  }
}

void MultiplexExternalRMVSource::ActBeforeCreatingDerivedFnBodyScope() {
  for (auto source : m_Sources) {
    source->ActBeforeCreatingDerivedFnBodyScope();
  }
}

void MultiplexExternalRMVSource::ActOnStartOfDerivedFnBody(
    const DiffRequest& request) {
  for (auto source : m_Sources) {
    source->ActOnStartOfDerivedFnBody(request);
  }
}

void MultiplexExternalRMVSource::ActOnEndOfDerivedFnBody() {
  for (auto source : m_Sources) {
    source->ActOnEndOfDerivedFnBody();
  }
}

void MultiplexExternalRMVSource::
    ActBeforeDifferentiatingStmtInVisitCompoundStmt() {
  for (auto source : m_Sources) {
    source->ActBeforeDifferentiatingStmtInVisitCompoundStmt();
  }
}

void MultiplexExternalRMVSource::ActAfterProcessingStmtInVisitCompoundStmt() {
  for (auto source : m_Sources) {
    source->ActAfterProcessingStmtInVisitCompoundStmt();
  }
}

void MultiplexExternalRMVSource::
    ActBeforeDifferentiatingSingleStmtBranchInVisitIfStmt() {
  for (auto source : m_Sources) {
    source->ActBeforeDifferentiatingSingleStmtBranchInVisitIfStmt();
  }
}

void MultiplexExternalRMVSource::
    ActBeforeFinalisingVisitBranchSingleStmtInIfVisitStmt() {
  for (auto source : m_Sources) {
    source->ActBeforeFinalisingVisitBranchSingleStmtInIfVisitStmt();
  }
}

void MultiplexExternalRMVSource::ActBeforeDifferentiatingLoopInitStmt() {
  for (auto source : m_Sources) {
    source->ActBeforeDifferentiatingLoopInitStmt();
  }
}

void MultiplexExternalRMVSource::ActBeforeDifferentiatingSingleStmtLoopBody() {
  for (auto source : m_Sources) {
    source->ActBeforeDifferentiatingSingleStmtLoopBody();
  }
}

void MultiplexExternalRMVSource::
    ActAfterProcessingSingleStmtBodyInVisitForLoop() {
  for (auto source : m_Sources) {
    source->ActAfterProcessingSingleStmtBodyInVisitForLoop();
  }
}

void MultiplexExternalRMVSource::ActBeforeFinalisingVisitReturnStmt(
    StmtDiff& ExprDiff, clang::Expr*& retDeclRefExpr) {
  for (auto source : m_Sources) {
    source->ActBeforeFinalisingVisitReturnStmt(ExprDiff, retDeclRefExpr);
  }
}

void MultiplexExternalRMVSource::ActBeforeFinalizingVisitCallExpr(
    const clang::CallExpr*& CE, clang::Expr*& OverloadedDerivedFn,
    llvm::SmallVectorImpl<clang::Expr*>& CallArgs,
    llvm::SmallVectorImpl<clang::VarDecl*>& ArgResultDecls, bool asGrad) {
  for (auto source : m_Sources) {
    source->ActBeforeFinalizingVisitCallExpr(CE, OverloadedDerivedFn, CallArgs,
                                             ArgResultDecls, asGrad);
  }
}

void MultiplexExternalRMVSource::ActBeforeFinalisingPostIncDecOp(
    StmtDiff& diff) {
  for (auto source : m_Sources) {
    source->ActBeforeFinalisingPostIncDecOp(diff);
  }
}
void MultiplexExternalRMVSource::ActAfterCloningLHSOfAssignOp(
    clang::Expr*& LCloned, clang::Expr*& R, clang::BinaryOperatorKind& opCode) {
  for (auto source : m_Sources) {
    source->ActAfterCloningLHSOfAssignOp(LCloned, R, opCode);
  }
}

void MultiplexExternalRMVSource::ActBeforeFinalisingAssignOp(
    clang::Expr*& LCloned, clang::Expr*& oldValue) {
  for (auto source : m_Sources) {
    source->ActBeforeFinalisingAssignOp(LCloned, oldValue);
  }
}

void MultiplexExternalRMVSource::ActOnStartOfDifferentiateSingleStmt() {
  for (auto source : m_Sources) {
    source->ActOnStartOfDifferentiateSingleStmt();
  }
}

void MultiplexExternalRMVSource::ActBeforeFinalizingDifferentiateSingleStmt(
    const direction& d) {
  for (auto source : m_Sources) {
    source->ActBeforeFinalizingDifferentiateSingleStmt(d);
  }
}

void MultiplexExternalRMVSource::ActBeforeFinalizingDifferentiateSingleExpr(
    const direction& d) {
  for (auto source : m_Sources) {
    source->ActBeforeFinalizingDifferentiateSingleExpr(d);
  }
}

void MultiplexExternalRMVSource::ActBeforeFinalizingVisitDeclStmt(
    llvm::SmallVectorImpl<clang::Decl*>& decls,
    llvm::SmallVectorImpl<clang::Decl*>& declsDiff) {
  for (auto source : m_Sources) {
    source->ActBeforeFinalizingVisitDeclStmt(decls, declsDiff);
  }
}
} // namespace clad