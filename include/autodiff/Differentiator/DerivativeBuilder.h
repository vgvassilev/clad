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
  class NamespaceDecl;
  class Scope;
  class Sema;
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
    clang::Sema& m_Sema;
    clang::ASTContext& m_Context;
    clang::FunctionDecl* derivedFD;
    clang::ValueDecl* independentVar;
    llvm::OwningPtr<clang::Scope> m_CurScope;
    llvm::OwningPtr<utils::StmtClone> m_NodeCloner;
    clang::NamespaceDecl* m_BuiltinDerivativesNSD;
    clang::Expr* updateReferencesOf(clang::Expr* InSubtree);
    clang::Expr* findOverloadedDefinition(clang::DeclarationNameInfo DNI,
                            llvm::SmallVector<clang::Expr*, 4> CallArgs);
    bool overloadExists(clang::Expr* UnresolvedLookup,
                            llvm::MutableArrayRef<clang::Expr*> ARargs);

  public:
    DerivativeBuilder(clang::Sema& S);
    ~DerivativeBuilder();

    ///\brief Produces the first derivative of a given function.
    ///
    ///\param[in] FD - the function that will be differentiated.
    ///
    ///\returns The differentiated function.
    ///
    clang::FunctionDecl* Derive(clang::FunctionDecl* FD,
                                      clang::ValueDecl* argVar);    
    NodeContext VisitStmt(clang::Stmt* S);
    NodeContext VisitCompoundStmt(clang::CompoundStmt* CS);
    NodeContext VisitReturnStmt(clang::ReturnStmt* RS);
    NodeContext VisitBinaryOperator(clang::BinaryOperator* BinOp);
    NodeContext VisitCXXOperatorCallExpr(clang::CXXOperatorCallExpr* OpCall);
    NodeContext VisitDeclRefExpr(clang::DeclRefExpr* DRE);
    NodeContext VisitParenExpr(clang::ParenExpr* PE);
    NodeContext VisitIntegerLiteral(clang::IntegerLiteral* IL);
    NodeContext VisitCallExpr(clang::CallExpr* CE);
    NodeContext VisitDeclStmt(clang::DeclStmt* DS);
  };
  
} // end namespace autodiff

#endif // AUTODIFF_DERIVATIVE_BUILDER_H
