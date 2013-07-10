//--------------------------------------------------------------------*- C++ -*-
// AutoDiff - the C++ Clang-based Automatic Differentiator
// version: $Id: ClangPlugin.cpp 7 2013-06-01 22:48:03Z v.g.vassilev@gmail.com $
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#ifndef AUTODIFF_DERIVATIVE_BUILDER_H
#define AUTODIFF_DERIVATIVE_BUILDER_H

#include "clang/AST/StmtVisitor.h"

#include "llvm/ADT/OwningPtr.h"

namespace clang {
  class ASTContext;
  class CXXOperatorCallExpr;
  class DeclRefExpr;
  class FunctionDecl;
  class Stmt;
}
namespace autodiff {
  namespace utils {
    class StmtClone;
  }
}

namespace autodiff {
  class NodeContext {
  public:
  private:
    clang::Stmt* m_Stmt;
  private:
    NodeContext() {};
  public:
    NodeContext(clang::Stmt* s) : m_Stmt(s) {}
    clang::Stmt* getStmt() { return m_Stmt; }
    const clang::Stmt* getStmt() const { return m_Stmt; }
    clang::Expr* getExpr() {
      assert(llvm::isa<clang::Expr>(m_Stmt) && "Must be an expression.");
      return llvm::cast<clang::Expr>(m_Stmt);
    }
    const clang::Expr* getExpr() const {
      assert(llvm::isa<clang::Expr>(m_Stmt) && "Must be an expression.");
      return llvm::cast<clang::Expr>(m_Stmt);
    }
  };
  
  class DerivativeBuilder
  : public clang::StmtVisitor<DerivativeBuilder, NodeContext> {
    
  private:
    clang::ASTContext* m_Context;
    llvm::OwningPtr<utils::StmtClone> m_NodeCloner;
    
  public:
    DerivativeBuilder();
    ~DerivativeBuilder();
    
    ///\brief Produces the first derivative of a given function.
    ///
    ///\param[in] FD - the function that will be differentiated.
    ///
    ///\returns The differentiated function.
    ///
    const clang::FunctionDecl* Derive(clang::FunctionDecl* FD);
    
    NodeContext VisitStmt(clang::Stmt* S);
    NodeContext VisitCompoundStmt(clang::CompoundStmt* CS);
    NodeContext VisitReturnStmt(clang::ReturnStmt* RS);
    NodeContext VisitBinaryOperator(clang::BinaryOperator* BinOp);
    NodeContext VisitCXXOperatorCallExpr(clang::CXXOperatorCallExpr* OpCall);
    NodeContext VisitDeclRefExpr(clang::DeclRefExpr* DRE);
    NodeContext VisitParenExpr(clang::ParenExpr* PE);
    NodeContext VisitIntegerLiteral(clang::IntegerLiteral* IL);
  };
  
} // end namespace autodiff

#endif // AUTODIFF_DERIVATIVE_BUILDER_H
