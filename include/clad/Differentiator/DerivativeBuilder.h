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
  class DerivativeBuilder;

  ///\brief A pair function, independent variable.
  ///
  class FunctionDeclInfo {
  private:
    clang::FunctionDecl* m_FD;
    clang::ParmVarDecl* m_PVD;
  public:
    FunctionDeclInfo(clang::FunctionDecl* FD, clang::ParmVarDecl* PVD);
    clang::FunctionDecl* getFD() const { return m_FD; }
    clang::ParmVarDecl* getPVD() const { return m_PVD; }
    bool isValid() const { return m_FD && m_PVD; }
    LLVM_DUMP_METHOD void dump() const;
  };

  ///\brief The list of the dependent functions which also need differentiation
  /// because they are called by the function we are asked to differentitate.
  ///
  class DiffPlan {
  private:
    typedef llvm::SmallVector<FunctionDeclInfo, 16> Functions;
    Functions m_Functions;
    clang::CallExpr* m_CallToUpdate;
  public:
    DiffPlan() : m_CallToUpdate(0) { }
    typedef Functions::iterator iterator;
    typedef Functions::const_iterator const_iterator;
    void push_back(FunctionDeclInfo FDI) { m_Functions.push_back(FDI); }
    iterator begin() { return m_Functions.begin(); }
    iterator end() { return m_Functions.end(); }
    const_iterator begin() const { return m_Functions.begin(); }
    const_iterator end() const { return m_Functions.end(); }
    size_t size() const { return m_Functions.size(); }
    void setCallToUpdate(clang::CallExpr* CE) { m_CallToUpdate = CE; }
    void updateCall(clang::FunctionDecl* FD, clang::Sema& SemaRef);
    LLVM_DUMP_METHOD void dump();
  };

  typedef llvm::SmallVector<DiffPlan, 16> DiffPlans;

  class DiffCollector: public clang::RecursiveASTVisitor<DiffCollector> {
  private:
    ///\brief The diff step-by-step plan for differentiation.
    ///
    DiffPlans& m_DiffPlans;

    ///\brief If set it means that we need to find the called functions and
    /// add them for implicit diff.
    ///
    FunctionDeclInfo* m_TopMostFDI;

    clang::Sema& m_Sema;

    DiffPlan& getCurrentPlan() { return m_DiffPlans.back(); }

    ///\brief Tries to find the independent variable of explicitly diffed
    /// functions.
    ///
    clang::ParmVarDecl* getIndependentArg(clang::Expr* argExpr,
                                          clang::FunctionDecl* FD);
  public:
    DiffCollector(clang::DeclGroupRef DGR, DiffPlans& plans, clang::Sema& S);
    bool VisitCallExpr(clang::CallExpr* E);
  };
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
    clang::FunctionDecl* Derive(clang::FunctionDecl* FD,
                                      clang::ValueDecl* argVar);
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
