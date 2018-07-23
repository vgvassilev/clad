//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// version: $Id: ClangPlugin.cpp 7 2013-06-01 22:48:03Z v.g.vassilev@gmail.com $
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#ifndef CLAD_DERIVATIVE_BUILDER_H
#define CLAD_DERIVATIVE_BUILDER_H

#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/StmtVisitor.h"

#include <stack>
#include <unordered_map>

namespace clang {
  class ASTContext;
  class CXXOperatorCallExpr;
  class DeclRefExpr;
  class FunctionDecl;
  class MemberExpr;
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
    
    //FIXME: warning: all paths through this function will call itself
    const clang::Stmt* getStmt() const {
      return const_cast<NodeContext*>(this)->getStmt();
    }
    
    const Statements& getStmts() const { return m_Stmts; }
    
    clang::CompoundStmt* wrapInCompoundStmt(clang::ASTContext& C) const;
        
    clang::Expr* getExpr() {
      assert(llvm::isa<clang::Expr>(getStmt()) && "Must be an expression.");
      return llvm::cast<clang::Expr>(getStmt());
    }

    //FIXME: warning: all paths through this function will call itself
    const clang::Expr* getExpr() const {
      return const_cast<NodeContext*>(this)->getExpr();
    }

    template<typename T> T* getAs() {
      if (clang::Expr* E = llvm::dyn_cast<clang::Expr>(getStmt()))
        return llvm::cast<T>(E);
      return llvm::cast<T>(getStmt());
    }
  };

  class DiffPlan;
  /// The main builder class which then uses either ForwardModeVisitor or 
  /// ReverseModeVisitor based on the required mode.
  class DerivativeBuilder {
  private:
    friend class VisitorBase;
    friend class ForwardModeVisitor;
    friend class ReverseModeVisitor;

    clang::Sema& m_Sema;
    clang::ASTContext& m_Context;
    std::unique_ptr<clang::Scope> m_CurScope;
    std::unique_ptr<utils::StmtClone> m_NodeCloner;
    clang::NamespaceDecl* m_BuiltinDerivativesNSD;

    /// Updates references in newly cloned statements.
    void updateReferencesOf(clang::Stmt* InSubtree);
    /// Clones a statement
    clang::Stmt* Clone(const clang::Stmt* S);
    /// A shorthand to simplify cloning of expressions.
    clang::Expr* Clone(const clang::Expr* E);
    /// A shorthand to simplify syntax for creation of new expressions.
    /// Uses m_Sema.BuildUnOp internally.
    clang::Expr* BuildOp(clang::UnaryOperatorKind OpCode, clang::Expr* E);
    /// Uses m_Sema.BuildBin internally.
    clang::Expr* BuildOp(
      clang::BinaryOperatorKind OpCode,
      clang::Expr* L,
      clang::Expr* R);

    clang::Expr* findOverloadedDefinition(clang::DeclarationNameInfo DNI,
                            llvm::SmallVectorImpl<clang::Expr*>& CallArgs);
    bool overloadExists(clang::Expr* UnresolvedLookup,
                            llvm::MutableArrayRef<clang::Expr*> ARargs);

  public:
    DerivativeBuilder(clang::Sema& S);
    ~DerivativeBuilder();

    ///\brief Produces the derivative of a given function
    /// according to a given plan.
    ///
    ///\param[in] FD - the function that will be differentiated.
    ///
    ///\returns The differentiated function.
    ///
    clang::FunctionDecl* Derive(FunctionDeclInfo& FDI, const DiffPlan & plan);
  };

  /// A base class for all common functionality for visitors
  class VisitorBase {
  protected:
    VisitorBase(DerivativeBuilder& builder) :
      m_Builder(builder),
      m_Sema(builder.m_Sema),
      m_Context(builder.m_Context),
      m_CurScope(builder.m_CurScope),
      m_DerivativeInFlight(false) {}

    DerivativeBuilder& m_Builder;
    clang::Sema& m_Sema;
    clang::ASTContext& m_Context;
    std::unique_ptr<clang::Scope> & m_CurScope;
    bool m_DerivativeInFlight;
    /// The Derivative function that is being generated.
    clang::FunctionDecl* m_Derivative;
    /// The function that is currently differentiated.
    clang::FunctionDecl* m_Function;

    clang::Expr* BuildOp(clang::UnaryOperatorKind OpCode, clang::Expr* E) {
      return m_Builder.BuildOp(OpCode, E);
    }

    clang::Expr* BuildOp(
      clang::BinaryOperatorKind OpCode,
      clang::Expr* L,
      clang::Expr* R) {
      return m_Builder.BuildOp(OpCode, L, R);
    }

    clang::CompoundStmt* MakeCompoundStmt(
      const llvm::SmallVector<clang::Stmt*, 16> & Stmts);
  };
    
  /// A visitor for processing the function code in forward mode.
  /// Used to compute derivatives by clad::differentiate.
  class ForwardModeVisitor
    : public clang::ConstStmtVisitor<ForwardModeVisitor, NodeContext>,
      public VisitorBase {
  private:
    clang::ValueDecl* m_IndependentVar;
    unsigned m_DerivativeOrder;
    unsigned m_ArgIndex;

  public:
    ForwardModeVisitor(DerivativeBuilder& builder);
    ~ForwardModeVisitor();

    ///\brief Produces the first derivative of a given function.
    ///
    ///\param[in] FD - the function that will be differentiated.
    ///
    ///\returns The differentiated function.
    ///
    clang::FunctionDecl* Derive(FunctionDeclInfo& FDI, const DiffPlan& plan);

    NodeContext Clone(const clang::Stmt* S);
    NodeContext VisitStmt(const clang::Stmt* S);
    NodeContext VisitCompoundStmt(const clang::CompoundStmt* CS);
    NodeContext VisitIfStmt(const clang::IfStmt* If);
    NodeContext VisitReturnStmt(const clang::ReturnStmt* RS);
    NodeContext VisitUnaryOperator(const clang::UnaryOperator* UnOp);
    NodeContext VisitBinaryOperator(const clang::BinaryOperator* BinOp);
    NodeContext VisitCXXOperatorCallExpr(
      const clang::CXXOperatorCallExpr* OpCall);
    NodeContext VisitDeclRefExpr(const clang::DeclRefExpr* DRE);
    NodeContext VisitParenExpr(const clang::ParenExpr* PE);
    NodeContext VisitMemberExpr(const clang::MemberExpr* ME);
    NodeContext VisitIntegerLiteral(const clang::IntegerLiteral* IL);
    NodeContext VisitFloatingLiteral(const clang::FloatingLiteral* FL);
    NodeContext VisitCallExpr(const clang::CallExpr* CE);
    NodeContext VisitDeclStmt(const clang::DeclStmt* DS);
    NodeContext VisitImplicitCastExpr(const clang::ImplicitCastExpr* ICE);
    NodeContext VisitConditionalOperator(const clang::ConditionalOperator* CO);
  };

  /// A visitor for processing the function code in reverse mode.
  /// Used to compute derivatives by clad::gradient.
  class ReverseModeVisitor
    : public clang::ConstStmtVisitor<ReverseModeVisitor, void>,
      public VisitorBase {
  private:

    using Stmts = llvm::SmallVector<clang::Stmt*, 16>;
 
    /// Stack is used to pass the arguments (dfdx) to further nodes
    /// in the Visit method.
    std::stack<clang::Expr*> m_Stack;
    clang::Expr* dfdx () {
        return m_Stack.top ();
    }
    void Visit(const clang::Stmt* stmt, clang::Expr* expr) {
        m_Stack.push(expr);
        clang::ConstStmtVisitor<ReverseModeVisitor, void>::Visit(stmt);
        m_Stack.pop();
    }
 
    /// A stack of all the blocks where the statements of the gradient function
    /// are stored (e.g., function body, if statement blocks).
    std::stack<Stmts> m_Blocks;
    /// Get the latest block of code (i.e. place for statements output).
    Stmts & currentBlock() {
      return m_Blocks.top();
    }
    /// Create new block.
    Stmts & startBlock() {
      m_Blocks.push({});
      return m_Blocks.top();
    }
    /// Remove the block from the stack, wrap it in CompoundStmt and return it.
    clang::CompoundStmt* finishBlock() {
      auto CS = MakeCompoundStmt(currentBlock());
      m_Blocks.pop();
      return CS;
    }
 
    //// A reference to the output parameter of the gradient function.
    clang::Expr* m_Result;
    // Shorthands that delegate their functionality to DerviativeBuilder.
    // Used to simplify the code.
    clang::Stmt* Clone(const clang::Stmt* S);
    clang::Expr* Clone(const clang::Expr* E);

  public:
    ReverseModeVisitor(DerivativeBuilder& builder);
    ~ReverseModeVisitor();

    ///\brief Produces the gradient of a given function.
    ///
    ///\param[in] FD - the function that will be differentiated.
    ///
    ///\returns The gradient of the function
    ///
    clang::FunctionDecl* Derive(FunctionDeclInfo & FDI, const DiffPlan& plan);
    void VisitCompoundStmt(const clang::CompoundStmt* CS);
    void VisitIfStmt(const clang::IfStmt* If);
    void VisitReturnStmt(const clang::ReturnStmt* RS);
    void VisitUnaryOperator(const clang::UnaryOperator* UnOp);
    void VisitBinaryOperator(const clang::BinaryOperator* BinOp);
    void VisitDeclStmt(const clang::DeclStmt* DS);
    void VisitMemberExpr(const clang::MemberExpr* ME);
    void VisitDeclRefExpr(const clang::DeclRefExpr* DRE);
    void VisitParenExpr(const clang::ParenExpr* PE);
    void VisitIntegerLiteral(const clang::IntegerLiteral* IL);
    void VisitFloatingLiteral(const clang::FloatingLiteral* FL);
    void VisitCallExpr(const clang::CallExpr* CE);
    void VisitImplicitCastExpr(const clang::ImplicitCastExpr* ICE);
    void VisitConditionalOperator(const clang::ConditionalOperator* CO);
  };
} // end namespace clad

#endif // CLAD_DERIVATIVE_BUILDER_H
