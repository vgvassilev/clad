//--------------------------------------------------------------------*- C++ -//
// clad - the C++ Clang-based Automatic Differentiator
//
// A constant folding tool, working on AST level
//
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//----------------------------------------------------------------------------//

#ifndef CLAD_CONSTANT_FOLDER_H
#define CLAD_CONSTANT_FOLDER_H

#include "clang/AST/StmtVisitor.h"

#include <cstdint>

namespace clang {
  class ASTContext;
  class BinaryOperator;
  class Expr;
  class ParenExpr;
  class QualType;
}

namespace clad {
  class ConstantFolder:
    public clang::StmtVisitor<ConstantFolder, clang::Expr*> {
  private:
    clang::ASTContext& m_Context;

  public:
    explicit ConstantFolder(clang::ASTContext& C) : m_Context(C) {}
    clang::Expr* fold(clang::Expr* E);
    clang::Expr* VisitExpr(clang::Expr* E);
    clang::Expr* VisitBinaryOperator(clang::BinaryOperator* BinOp);
    clang::Expr* VisitParenExpr(clang::ParenExpr* PE);
    static clang::Expr* synthesizeLiteral(clang::QualType, clang::ASTContext &C,
                                          uint64_t val);
    /// True when E is a constant zero that evaluates without side effects.
    /// This is the predicate the identity rules use, exposed so a caller can
    /// tell in advance which terms the fold is going to drop.
    static bool evaluatesToZero(clang::Expr* E, clang::ASTContext& C);

  private:
    clang::Expr* trivialFold(clang::Expr* E);
  };
} // end namespace clad
#endif // CLAD_CONSTANT_FOLDER_H
