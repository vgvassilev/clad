//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// version: $Id: ClangPlugin.cpp 7 2013-06-01 22:48:03Z v.g.vassilev@gmail.com $
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#ifndef CLAD_DERIVATIVE_BUILDER_H
#define CLAD_DERIVATIVE_BUILDER_H

#include "clang/AST/RecursiveASTVisitor.h"
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
  class FunctionDeclInfo;
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

  class DiffPlan;
  class DerivativeBuilder
    : public clang::ConstStmtVisitor<DerivativeBuilder, NodeContext> {
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
    clang::FunctionDecl* Derive(FunctionDeclInfo& FDI, DiffPlan* plan);
    NodeContext VisitStmt(const clang::Stmt* S);
    NodeContext VisitCompoundStmt(const clang::CompoundStmt* CS);
    NodeContext VisitIfStmt(const clang::IfStmt* If);
    NodeContext VisitReturnStmt(const clang::ReturnStmt* RS);
    NodeContext VisitUnaryOperator(const clang::UnaryOperator* UnOp);
    NodeContext VisitBinaryOperator(const clang::BinaryOperator* BinOp);
    NodeContext VisitCXXOperatorCallExpr(const clang::CXXOperatorCallExpr* OpCall);
    NodeContext VisitDeclRefExpr(const clang::DeclRefExpr* DRE);
    NodeContext VisitParenExpr(const clang::ParenExpr* PE);
    NodeContext VisitIntegerLiteral(const clang::IntegerLiteral* IL);
    NodeContext VisitFloatingLiteral(const clang::FloatingLiteral* FL);
    NodeContext VisitCallExpr(const clang::CallExpr* CE);
    NodeContext VisitDeclStmt(const clang::DeclStmt* DS);
    NodeContext VisitImplicitCastExpr(const clang::ImplicitCastExpr* ICE);
  };
} // end namespace clad

#endif // CLAD_DERIVATIVE_BUILDER_H
