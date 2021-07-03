//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// version: $Id: ClangPlugin.cpp 7 2013-06-01 22:48:03Z v.g.vassilev@gmail.com $
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#ifndef CLAD_REVERSE_MODE_VISITOR_H
#define CLAD_REVERSE_MODE_VISITOR_H

#include "Compatibility.h"
#include "VisitorBase.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Sema/Sema.h"

#include <array>
#include <stack>
#include <unordered_map>

namespace clad {
  /// A visitor for processing the function code in reverse mode.
  /// Used to compute derivatives by clad::gradient.
  class ReverseModeVisitor
      : public clang::ConstStmtVisitor<ReverseModeVisitor, StmtDiff>,
        public VisitorBase {
  private:
    llvm::SmallVector<const clang::VarDecl*, 16> m_IndependentVars;
    /// In addition to a sequence of forward-accumulated Stmts (m_Blocks), in
    /// the reverse mode we also accumulate Stmts for the reverse pass which
    /// will be executed on return.
    std::vector<Stmts> m_Reverse;
    /// Stack is used to pass the arguments (dfdx) to further nodes
    /// in the Visit method.
    std::stack<clang::Expr*> m_Stack;
    /// A sequence of DeclStmts containing "tape" variable declarations
    /// that will be put immediately in the beginning of derivative function
    /// block.
    Stmts m_Globals;
    //// A reference to the output parameter of the gradient function.
    clang::Expr* m_Result;
    /// A flag indicating if the Stmt we are currently visiting is inside loop.
    bool isInsideLoop = false;
    /// Output variable of vector-valued function
    std::string outputArrayStr;
    unsigned outputArrayCursor = 0;
    unsigned numParams = 0;
    bool isVectorValued = false;

    const char* funcPostfix() const {
      if (isVectorValued)
        return "_jac";
      else
        return "_grad";
    }

    /// Removes the local as well as non-local const qualifiers from a QualType
    /// and returns a new type.
    static clang::QualType
    getNonConstType(clang::QualType T, clang::ASTContext& C, clang::Sema& S) {
      if (T->isPointerType()) {
        clang::Qualifiers quals(T->getPointeeType().getQualifiers());
        quals.removeConst();
        clang::QualType newType = S.BuildQualifiedType(
            T->getPointeeType().getUnqualifiedType(), noLoc, quals);
        return C.getPointerType(newType);
      } else {
        clang::Qualifiers quals(T.getQualifiers());
        quals.removeConst();
        return S.BuildQualifiedType(T.getUnqualifiedType(), noLoc, quals);
      }
    }

  public:
    clang::Expr* dfdx() {
      if (m_Stack.empty())
        return nullptr;
      return m_Stack.top();
    }
    StmtDiff Visit(const clang::Stmt* stmt, clang::Expr* dfdS = nullptr) {
      // No need to push the same expr multiple times.
      bool push = !(!m_Stack.empty() && (dfdS == dfdx()));
      if (push)
        m_Stack.push(dfdS);
      auto result =
          clang::ConstStmtVisitor<ReverseModeVisitor, StmtDiff>::Visit(stmt);
      if (push)
        m_Stack.pop();
      return result;
    }

    /// An enum to operate between forward and reverse passes.
    enum direction { forward, reverse };
    /// Get the latest block of code (i.e. place for statements output).
    Stmts& getCurrentBlock(direction d = forward) {
      if (d == forward)
        return m_Blocks.back();
      else
        return m_Reverse.back();
    }
    /// Create new block.
    Stmts& beginBlock(direction d = forward) {
      if (d == forward)
        m_Blocks.push_back({});
      else
        m_Reverse.push_back({});
      return getCurrentBlock(d);
    }
    /// Remove the block from the stack, wrap it in CompoundStmt and return it.
    clang::CompoundStmt* endBlock(direction d = forward) {
      if (d == forward) {
        auto CS = MakeCompoundStmt(getCurrentBlock(forward));
        m_Blocks.pop_back();
        return CS;
      } else {
        auto CS = MakeCompoundStmt(getCurrentBlock(reverse));
        std::reverse(CS->body_begin(), CS->body_end());
        m_Reverse.pop_back();
        return CS;
      }
    }
    /// Output a statement to the current block. If Stmt is null or is an unused
    /// expression, it is not output and false is returned.
    bool addToCurrentBlock(clang::Stmt* S, direction d = forward) {
      return addToBlock(S, getCurrentBlock(d));
    }

    /// Stores the result of an expression in a temporary variable (of the same
    /// type as is the result of the expression) and returns a reference to it.
    /// If force decl creation is true, this will allways create a temporary
    /// variable declaration. Otherwise, temporary variable is created only
    /// if E requires evaluation (e.g. there is no point to store literals or
    /// direct references in intermediate variables)
    clang::Expr* StoreAndRef(clang::Expr* E, direction d = forward,
                             llvm::StringRef prefix = "_t",
                             bool forceDeclCreation = false,
                             clang::VarDecl::InitializationStyle IS =
                                 clang::VarDecl::InitializationStyle::CInit) {
      assert(E && "cannot infer type from null expression");
      return StoreAndRef(E, getNonConstType(E->getType(), m_Context, m_Sema), d,
                         prefix, forceDeclCreation, IS);
    }

    /// An overload allowing to specify the type for the variable.
    clang::Expr* StoreAndRef(clang::Expr* E, clang::QualType Type,
                             direction d = forward,
                             llvm::StringRef prefix = "_t",
                             bool forceDeclCreation = false,
                             clang::VarDecl::InitializationStyle IS =
                                 clang::VarDecl::InitializationStyle::CInit) {
      // Name reverse temporaries as "_r" instead of "_t".
      if ((d == reverse) && (prefix == "_t"))
        prefix = "_r";
      return VisitorBase::StoreAndRef(E, Type, getCurrentBlock(d), prefix,
                                      forceDeclCreation, IS);
    }

    /// For an expr E, decides if it is useful to store it in a global temporary
    /// variable and replace E's further usage by a reference to that variable
    /// to avoid recomputiation.
    bool UsefulToStoreGlobal(clang::Expr* E);
    clang::VarDecl* GlobalStoreImpl(clang::QualType Type,
                                    llvm::StringRef prefix);
    /// Creates a (global in the function scope) variable declaration, puts
    /// it into m_Globals block (to be inserted into the beginning of fn's
    /// body). Returns reference R to the created declaration. If E is not null,
    /// puts an additional assignment statement (R = E) in the forward block.
    /// Alternatively, if isInsideLoop is true, stores E in a stack. Returns
    /// StmtDiff, where .getExpr() is intended to be used in forward pass and
    /// .getExpr_dx() in the reverse pass. Two expressions can be different in
    /// some cases, e.g. clad::push/pop inside loops.
    StmtDiff GlobalStoreAndRef(clang::Expr* E,
                               clang::QualType Type,
                               llvm::StringRef prefix = "_t",
                               bool force = false);
    StmtDiff GlobalStoreAndRef(clang::Expr* E,
                               llvm::StringRef prefix = "_t",
                               bool force = false);

    //// A type returned by DelayedGlobalStoreAndRef
    /// .Result is a reference to the created (yet uninitialized) global
    /// variable. When the expression is finally visited and rebuilt, .Finalize
    /// must be called with new rebuilt expression, to initialize the global
    /// variable. Alternatively, expression may be not worth storing in a global
    /// varialbe and is  easy to clone (e.g. it is a constant literal). Then
    /// .Result is cloned E, .isConstant is true and .Finalize does nothing.
    struct DelayedStoreResult {
      ReverseModeVisitor& V;
      StmtDiff Result;
      bool isConstant;
      bool isInsideLoop;
      void Finalize(clang::Expr* New);
    };

    /// Sometimes (e.g. when visiting multiplication/division operator), we
    /// need to allocate global variable for an expression (e.g. for RHS) before
    /// we visit that expression for efficiency reasons, since we may use that
    /// global variable for visiting another expression (e.g. LHS) instead of
    /// cloning LHS. The global variable will be assigned with the actual
    /// expression only later, after the expression is visited and rebuilt.
    /// This is what DelayedGlobalStoreAndRef does. E is expected to be the
    /// original (uncloned) expression.
    DelayedStoreResult DelayedGlobalStoreAndRef(clang::Expr* E,
                                                llvm::StringRef prefix = "_t");

    struct CladTapeResult {
      ReverseModeVisitor& V;
      clang::Expr* Push;
      clang::Expr* Pop;
      clang::Expr* Ref;
      /// A request to get expr accessing last element in the tape
      /// (clad::back(Ref)). Since it is required only rarely, it is built on
      /// demand in the method.
      clang::Expr* Last();
    };

    /// If E is supposed to be stored in a tape, will create a global
    /// declaration of tape of corresponding type and return a result struct
    /// with reference to the tape and constructed calls to push/pop methods.
    CladTapeResult MakeCladTapeFor(clang::Expr* E);

  public:
    ReverseModeVisitor(DerivativeBuilder& builder);
    ~ReverseModeVisitor();

    ///\brief Produces the gradient of a given function.
    ///
    ///\param[in] FD - the function that will be differentiated.
    ///
    ///\returns The gradient of the function, potentially created enclosing
    /// context and if generated, its overload.
    ///
    /// We name the gradient of f as 'f_grad'.
    /// If the gradient of the same function is requested several times
    /// with different parameters, but same parameter types, every such request
    /// will create f_grad function with the same signature, which will be
    /// ambiguous. E.g.
    ///   double f(double x, double y, double z) { ... }
    ///   clad::gradient(f, "x, y");
    ///   clad::gradient(f, "x, z");
    /// will create 2 definitions for f_grad with the same signature.
    ///
    /// Improved naming scheme is required. Hence, we append the indices to of
    /// the requested parameters to 'f_grad', i.e. in the previous example "x,
    /// y" will give 'f_grad_0_1' and "x, z" will give 'f_grad_0_2'.
    OverloadedDeclWithContext Derive(const clang::FunctionDecl* FD,
                                     const DiffRequest& request);
    StmtDiff VisitArraySubscriptExpr(const clang::ArraySubscriptExpr* ASE);
    StmtDiff VisitBinaryOperator(const clang::BinaryOperator* BinOp);
    StmtDiff VisitCallExpr(const clang::CallExpr* CE);
    StmtDiff VisitCompoundStmt(const clang::CompoundStmt* CS);
    StmtDiff VisitConditionalOperator(const clang::ConditionalOperator* CO);
    StmtDiff VisitCXXBoolLiteralExpr(const clang::CXXBoolLiteralExpr* BL);
    StmtDiff VisitCXXDefaultArgExpr(const clang::CXXDefaultArgExpr* DE);
    StmtDiff VisitDeclRefExpr(const clang::DeclRefExpr* DRE);
    StmtDiff VisitDeclStmt(const clang::DeclStmt* DS);
    StmtDiff VisitFloatingLiteral(const clang::FloatingLiteral* FL);
    StmtDiff VisitForStmt(const clang::ForStmt* FS);
    StmtDiff VisitIfStmt(const clang::IfStmt* If);
    StmtDiff VisitImplicitCastExpr(const clang::ImplicitCastExpr* ICE);
    StmtDiff VisitInitListExpr(const clang::InitListExpr* ILE);
    StmtDiff VisitIntegerLiteral(const clang::IntegerLiteral* IL);
    StmtDiff VisitMemberExpr(const clang::MemberExpr* ME);
    StmtDiff VisitParenExpr(const clang::ParenExpr* PE);
    StmtDiff VisitReturnStmt(const clang::ReturnStmt* RS);
    StmtDiff VisitStmt(const clang::Stmt* S);
    StmtDiff VisitUnaryOperator(const clang::UnaryOperator* UnOp);
    StmtDiff VisitExprWithCleanups(const clang::ExprWithCleanups* EWC);
    /// Decl is not Stmt, so it cannot be visited directly.
    VarDeclDiff DifferentiateVarDecl(const clang::VarDecl* VD);
    /// A helper method to differentiate a single Stmt in the reverse mode.
    /// Internally, calls Visit(S, expr). Its result is wrapped into a
    /// CompoundStmt (if several statements are created) and proper Stmt
    /// order is maintained.
    StmtDiff DifferentiateSingleStmt(const clang::Stmt* S,
                                     clang::Expr* dfdS = nullptr);
    /// A helper method used to keep substatements created by Visit(E, expr) in
    /// separate forward/reverse blocks instead of putting them into current
    /// blocks. First result is a StmtDiff of forward/reverse blocks with
    /// additionally created Stmts, second is a direct result of call to Visit.
    std::pair<StmtDiff, StmtDiff>
    DifferentiateSingleExpr(const clang::Expr* E, clang::Expr* dfdE = nullptr);
    /// Shorthand for warning on differentiation of unsupported operators
    void unsupportedOpWarn(clang::SourceLocation loc,
                           llvm::ArrayRef<llvm::StringRef> args = {}) {
      diag(clang::DiagnosticsEngine::Warning,
           loc,
           "attempt to differentiate unsupported operator, ignored.",
           args);
    }
    /// Builds an overload for the gradient function that has derived params for
    /// all the arguments of the requested function and it calls the original
    /// gradient function internally
    clang::FunctionDecl* CreateGradientOverload(
        llvm::SmallVectorImpl<clang::QualType>& GradientParamTypes,
        llvm::SmallVectorImpl<clang::ParmVarDecl*>& GradientParams,
        clang::DeclarationNameInfo& GradientName,
        clang::FunctionDecl* GradientFD);
  };
} // end namespace clad

#endif // CLAD_REVERSE_MODE_VISITOR_H