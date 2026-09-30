#include "clad/Differentiator/ASTIntegrity.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Stmt.h"
#include "clang/Tooling/Tooling.h"
#include "gtest/gtest.h"

using namespace clang;
using namespace clad;

TEST(ASTIntegrityTest, FindSharedNode) {
  // Build a basic AST to get a valid ASTContext.
  auto AST = tooling::buildASTFromCode("void f() {}");
  ASTContext& Ctx = AST->getASTContext();

  // Create an integer literal to share.
  auto* SharedLit = IntegerLiteral::Create(Ctx, llvm::APInt(32, 1), Ctx.IntTy, SourceLocation());

  Stmt* Stmts1[] = {SharedLit};
  auto* Comp1 = CompoundStmt::Create(Ctx, Stmts1, SourceLocation(), SourceLocation());

  Stmt* Stmts2[] = {SharedLit, Comp1};
  auto* Root = CompoundStmt::Create(Ctx, Stmts2, SourceLocation(), SourceLocation());

  const Stmt* Shared = findSharedNode(Root);
  EXPECT_EQ(Shared, SharedLit);

  auto* Lit2 = IntegerLiteral::Create(Ctx, llvm::APInt(32, 2), Ctx.IntTy, SourceLocation());
  Stmt* Stmts3[] = {Lit2, Comp1};
  auto* ProperRoot = CompoundStmt::Create(Ctx, Stmts3, SourceLocation(), SourceLocation());

  EXPECT_EQ(findSharedNode(ProperRoot), nullptr);
}
