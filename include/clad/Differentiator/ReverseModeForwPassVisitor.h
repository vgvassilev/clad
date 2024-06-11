#ifndef CLAD_DIFFERENTIATOR_REVERSEMODEFORWPASSVISITOR_H
#define CLAD_DIFFERENTIATOR_REVERSEMODEFORWPASSVISITOR_H

#include "clad/Differentiator/ParseDiffArgsTypes.h"
#include "clad/Differentiator/ReverseModeVisitor.h"

#include "clang/AST/StmtVisitor.h"
#include "clang/Sema/Sema.h"

#include "llvm/ADT/SmallVector.h"

namespace clad {
class ReverseModeForwPassVisitor : public ReverseModeVisitor {
private:
  Stmts m_Globals;

  llvm::SmallVector<clang::QualType, 8>
  ComputeParamTypes(const DiffParams& diffParams);
  clang::QualType ComputeReturnType();
  llvm::SmallVector<clang::ParmVarDecl*, 8> BuildParams(DiffParams& diffParams);
  clang::QualType GetParameterDerivativeType(clang::QualType yType,
                                             clang::QualType xType);

public:
  ReverseModeForwPassVisitor(DerivativeBuilder& builder,
                             const DiffRequest& request);
  DerivativeAndOverload Derive(const clang::FunctionDecl* FD,
                               const DiffRequest& request);

  StmtDiff ProcessSingleStmt(const clang::Stmt* S);
  StmtDiff VisitCompoundStmt(const clang::CompoundStmt* CS) override;
  StmtDiff VisitDeclRefExpr(const clang::DeclRefExpr* DRE) override;
  StmtDiff VisitReturnStmt(const clang::ReturnStmt* RS) override;
  StmtDiff VisitUnaryOperator(const clang::UnaryOperator* UnOp) override;
};
} // namespace clad

#endif
