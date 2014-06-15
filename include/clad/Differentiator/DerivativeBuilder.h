//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// version: $Id: ClangPlugin.cpp 7 2013-06-01 22:48:03Z v.g.vassilev@gmail.com $
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#ifndef CLAD_DERIVATIVE_BUILDER_H
#define CLAD_DERIVATIVE_BUILDER_H

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
namespace clad {
  namespace utils {
    class StmtClone;
  }
}

namespace clad {
  class NodeContext {
  private:
    typedef llvm::SmallVector<clang::Stmt*, 2> Statements;
    Statements m_Stmts;
    NodeContext() {};
  public:
    NodeContext(clang::Stmt* s) { m_Stmts.push_back(s); }
    
    NodeContext(clang::Stmt* s0, clang::Stmt* s1) {
      m_Stmts.push_back(s0);
      m_Stmts.push_back(s1);
    }
    
    //NodeContext(llvm::ArrayRef) : m_Stmt(s) {}
    
    bool isSingleStmt() const { return m_Stmts.size() == 1; }
    
    clang::Stmt* getStmt() {
      assert(isSingleStmt() && "Cannot get multiple stmts.");
      return m_Stmts.front();
    }
    
    const clang::Stmt* getStmt() const { return getStmt(); }
    
    const Statements& getStmts() const { return m_Stmts; }
    
    clang::CompoundStmt* wrapInCompoundStmt(clang::ASTContext& C) const;
        
    clang::Expr* getExpr() {
      assert(llvm::isa<clang::Expr>(getStmt()) && "Must be an expression.");
      return llvm::cast<clang::Expr>(getStmt());
    }

    const clang::Expr* getExpr() const { return getExpr(); }

    template<typename T> T* getAs() {
      if (clang::Expr* E = llvm::dyn_cast<clang::Expr>(getStmt()))
        return llvm::cast<T>(E);
      return llvm::cast<T>(getStmt());
    }
  };

  class DerivativeBuilder
    : public clang::StmtVisitor<DerivativeBuilder, NodeContext> {
    
  private:
    clang::Sema& m_Sema;
    clang::ASTContext& m_Context;
    clang::ValueDecl* m_IndependentVar;
    llvm::OwningPtr<clang::Scope> m_CurScope;
    llvm::OwningPtr<utils::StmtClone> m_NodeCloner;
    clang::NamespaceDecl* m_BuiltinDerivativesNSD;
    bool m_DerivativeInFlight;
    void updateReferencesOf(clang::Stmt* InSubtree);
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
    NodeContext VisitIfStmt(clang::IfStmt* If);
    NodeContext VisitReturnStmt(clang::ReturnStmt* RS);
    NodeContext VisitUnaryOperator(clang::UnaryOperator* UnOp);
    NodeContext VisitBinaryOperator(clang::BinaryOperator* BinOp);
    NodeContext VisitCXXOperatorCallExpr(clang::CXXOperatorCallExpr* OpCall);
    NodeContext VisitDeclRefExpr(clang::DeclRefExpr* DRE);
    NodeContext VisitParenExpr(clang::ParenExpr* PE);
    NodeContext VisitIntegerLiteral(clang::IntegerLiteral* IL);
    NodeContext VisitCallExpr(clang::CallExpr* CE);
    NodeContext VisitDeclStmt(clang::DeclStmt* DS);
    NodeContext VisitImplicitCastExpr(clang::ImplicitCastExpr* ICE);
  };
  
} // end namespace clad

#endif // CLAD_DERIVATIVE_BUILDER_H
