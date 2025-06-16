#ifndef CLAD_DIFFERENTIATOR_REVERSEMODEFORWPASSVISITOR_H
#define CLAD_DIFFERENTIATOR_REVERSEMODEFORWPASSVISITOR_H

#include "clad/Differentiator/DerivativeBuilder.h"
#include "clad/Differentiator/ParseDiffArgsTypes.h"
#include "clad/Differentiator/ReverseModeVisitor.h"
#include "clad/Differentiator/VisitorBase.h"

#include "clang/AST/StmtVisitor.h"
#include "clang/Sema/Sema.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace clad {
class ReverseModeForwPassVisitor : public ReverseModeVisitor {
private:
  Stmts m_Globals;

  llvm::SmallVector<clang::ParmVarDecl*, 8> BuildParams(DiffParams& diffParams);

public:
  ReverseModeForwPassVisitor(DerivativeBuilder& builder,
                             const DiffRequest& request);
  DerivativeAndOverload Derive() override;

  // These overrides are a workaround to prevent RMFPV from generating
  // reverse sweep derivative stmts and store/restore stmts,
  // which are not used in reverse_forw functions
  clang::Expr* dfdx() override { return nullptr; }
  StmtDiff StoreAndRestore(clang::Expr* E, llvm::StringRef prefix = "_t",
                           bool moveToTape = false) override {
    return {};
  }

  StmtDiff ProcessSingleStmt(const clang::Stmt* S);
  StmtDiff VisitCompoundStmt(const clang::CompoundStmt* CS) override;
  StmtDiff VisitDeclRefExpr(const clang::DeclRefExpr* DRE) override;
  StmtDiff VisitReturnStmt(const clang::ReturnStmt* RS) override;
  StmtDiff VisitUnaryOperator(const clang::UnaryOperator* UnOp) override;
};
} // namespace clad

#endif
