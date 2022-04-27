#ifndef CLAD_TRANSFORM_SOURCE_FN_VISITOR_H
#define CLAD_TRANSFORM_SOURCE_FN_VISITOR_H

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
  ReverseModeForwPassVisitor(DerivativeBuilder& builder);
  OverloadedDeclWithContext Derive(const clang::FunctionDecl* FD,
                                   const DiffRequest& request);

  StmtDiff ProcessSingleStmt(const clang::Stmt* S);

  StmtDiff VisitStmt(const clang::Stmt* S) override;
  StmtDiff VisitCompoundStmt(const clang::CompoundStmt* CS) override;
  StmtDiff VisitDeclRefExpr(const clang::DeclRefExpr* DRE) override;
  // StmtDiff VisitDeclStmt(const clang::DeclStmt* DS);
  StmtDiff VisitReturnStmt(const clang::ReturnStmt* RS) override;
};
} // namespace clad

#endif