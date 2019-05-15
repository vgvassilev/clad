//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// version: $Id: ClangPlugin.cpp 7 2013-06-01 22:48:03Z v.g.vassilev@gmail.com $
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#ifndef CLAD_DERIVATIVE_BUILDER_H
#define CLAD_DERIVATIVE_BUILDER_H

#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Sema/Sema.h"

#include <array>
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
  class DiffRequest;
  namespace plugin {
    class CladPlugin;
    clang::FunctionDecl* ProcessDiffRequest(CladPlugin& P, DiffRequest& request);
  }
}

namespace clad {
  /// A pair of FunctionDecl and potential enclosing context, e.g. a function
  // in nested namespaces
  using DeclWithContext = std::pair<clang::FunctionDecl*, clang::Decl*>;

  using DiffParams = llvm::SmallVector<const clang::VarDecl*, 16>;

  static clang::SourceLocation noLoc{};
  /// The main builder class which then uses either ForwardModeVisitor or
  /// ReverseModeVisitor based on the required mode.
  class DerivativeBuilder {
  private:
    friend class VisitorBase;
    friend class ForwardModeVisitor;
    friend class ReverseModeVisitor;

    clang::Sema& m_Sema;
    plugin::CladPlugin& m_CladPlugin;
    clang::ASTContext& m_Context;
    std::unique_ptr<utils::StmtClone> m_NodeCloner;
    clang::NamespaceDecl* m_BuiltinDerivativesNSD;

    clang::Expr* findOverloadedDefinition(clang::DeclarationNameInfo DNI,
                            llvm::SmallVectorImpl<clang::Expr*>& CallArgs);
    bool overloadExists(clang::Expr* UnresolvedLookup,
                            llvm::MutableArrayRef<clang::Expr*> ARargs);
    /// Shorthand to issues a warning or error.
    template <std::size_t N>
    void diag(clang::DiagnosticsEngine::Level level, // Warning or Error
              clang::SourceLocation loc,
              const char (&format)[N],
              llvm::ArrayRef<llvm::StringRef> args = {}) {
      unsigned diagID = m_Sema.Diags.getCustomDiagID(level, format);
      clang::Sema::SemaDiagnosticBuilder stream = m_Sema.Diag(loc, diagID);
      for (auto arg : args)
        stream << arg;
    }
  public:
    DerivativeBuilder(clang::Sema& S, plugin::CladPlugin& P);
    ~DerivativeBuilder();

    ///\brief Produces the derivative of a given function
    /// according to a given plan.
    ///
    ///\param[in] FD - the function that will be differentiated.
    ///
    ///\returns The differentiated function and potentially created enclosing
    /// context.
    ///
    DeclWithContext Derive(const clang::FunctionDecl* FD,
                           const DiffRequest & request);
  };

  /// A base class for all common functionality for visitors
  class VisitorBase {
  protected:
    VisitorBase(DerivativeBuilder& builder) :
      m_Builder(builder),
      m_Sema(builder.m_Sema),
      m_CladPlugin(builder.m_CladPlugin),
      m_Context(builder.m_Context),
      m_CurScope(m_Sema.TUScope),
      m_DerivativeFnScope(nullptr),
      m_DerivativeInFlight(false),
      m_Derivative(nullptr),
      m_Function(nullptr) {}

    using Stmts = llvm::SmallVector<clang::Stmt*, 16>;

    DerivativeBuilder& m_Builder;
    clang::Sema& m_Sema;
    plugin::CladPlugin& m_CladPlugin;
    clang::ASTContext& m_Context;
    /// Current Scope at the point of visiting.
    clang::Scope* m_CurScope;
    /// Pointer to the topmost Scope in the created derivative function.
    clang::Scope* m_DerivativeFnScope;
    bool m_DerivativeInFlight;
    /// The Derivative function that is being generated.
    clang::FunctionDecl* m_Derivative;
    /// The function that is currently differentiated.
    const clang::FunctionDecl* m_Function;

    /// Map used to keep track of variable declarations and match them
    /// with their derivatives.
    std::unordered_map<const clang::VarDecl*, clang::Expr*> m_Variables;
    /// Map contains variable declarations replacements. If the original
    /// function contains a declaration which name collides with something
    /// already created inside derivative's body, the declaration is replaced
    /// with a new one.
    /// See the example inside ForwardModeVisitor::VisitDeclStmt.
    std::unordered_map<const clang::VarDecl*, clang::VarDecl*> m_DeclReplacements;
    /// A stack of all the blocks where the statements of the gradient function
    /// are stored (e.g., function body, if statement blocks).
    std::vector<Stmts> m_Blocks;
  public:
    template <typename Range>
    clang::CompoundStmt* MakeCompoundStmt(const Range & Stmts) {
      auto Stmts_ref = llvm::makeArrayRef(Stmts.data(), Stmts.size());
      return new (m_Context) clang::CompoundStmt(m_Context,
                                                 Stmts_ref,
                                                 noLoc,
                                                 noLoc);
    }

    /// Get the latest block of code (i.e. place for statements output).
    Stmts& getCurrentBlock() {
      return m_Blocks.back();
    }
    /// Create new block.
    Stmts& beginBlock() {
      m_Blocks.push_back({});
      return m_Blocks.back();
    }
    /// Remove the block from the stack, wrap it in CompoundStmt and return it.
    clang::CompoundStmt* endBlock() {
      auto CS = MakeCompoundStmt(getCurrentBlock());
      m_Blocks.pop_back();
      return CS;
    }

    // Check if result of the expression is unused.
    bool isUnusedResult(const clang::Expr* E);
    /// Output a statement to the current block. If Stmt is null or is an unused
    /// expression, it is not output and false is returned. 
    bool addToCurrentBlock(clang::Stmt* S);
    bool addToBlock(clang::Stmt* S, Stmts& block);

    /// Get a current scope.
    clang::Scope* getCurrentScope() {
      return m_CurScope;
    }
    /// Enters a new scope.
    void beginScope(unsigned ScopeFlags) {
      // FIXME: since Sema::CurScope is private, we cannot access it and have
      // to use separate member variable m_CurScope. The only options to set
      // CurScope of Sema seemt to be through Parser or ContextAndScopeRAII.
      m_CurScope = new clang::Scope(getCurrentScope(), ScopeFlags, m_Sema.Diags);
    }
    void endScope() {
      // This will remove all the decls in the scope from the IdResolver. 
      m_Sema.ActOnPopScope(noLoc, m_CurScope);
      auto oldScope = m_CurScope;
      m_CurScope = oldScope->getParent();
      delete oldScope;
    }

    /// A shorthand to simplify syntax for creation of new expressions.
    /// Uses m_Sema.BuildUnOp internally.
    clang::Expr* BuildOp(clang::UnaryOperatorKind OpCode, clang::Expr* E);
    /// Uses m_Sema.BuildBin internally.
    clang::Expr* BuildOp(clang::BinaryOperatorKind OpCode,
                         clang::Expr* L,
                         clang::Expr* R);

    clang::Expr* BuildParens(clang::Expr* E);

    /// Builds variable declaration to be used inside the derivative body
    clang::VarDecl* BuildVarDecl(clang::QualType Type,
                                 clang::IdentifierInfo* Identifier,
                                 clang::Expr* Init = nullptr,
                                 bool DirectInit = false,
                                 clang::TypeSourceInfo* TSI = nullptr);

    clang::VarDecl* BuildVarDecl(clang::QualType Type,
                                 llvm::StringRef prefix = "_t",
                                 clang::Expr* Init = nullptr,
                                 bool DirectInit = false,
                                 clang::TypeSourceInfo* TSI = nullptr);
    /// Creates a namespace declaration and enters its context. All subsequent
    /// Stmts are built inside that namespace, until m_Sema.PopDeclContextIsUsed.
    clang::NamespaceDecl* BuildNamespaceDecl(clang::IdentifierInfo* II,
                                             bool isInline);
    /// Rebuild a sequence of nested namespaces ending with DC.
    clang::NamespaceDecl* RebuildEnclosingNamespaces(clang::DeclContext* DC);
    /// Wraps a declaration in DeclStmt.
    clang::DeclStmt* BuildDeclStmt(clang::Decl* D);
    clang::DeclStmt* BuildDeclStmt(llvm::MutableArrayRef<clang::Decl*> DS);

    /// Builds a DeclRefExpr to a given Decl.
    clang::DeclRefExpr* BuildDeclRef(clang::DeclaratorDecl* D);

    /// Stores the result of an expression in a temporary variable (of the same
    /// type as is the result of the expression) and returns a reference to it.
    /// If force decl creation is true, this will allways create a temporary
    /// variable declaration. Otherwise, temporary variable is created only 
    /// if E requires evaluation (e.g. there is no point to store literals or
    /// direct references in intermediate variables)
    clang::Expr* StoreAndRef(clang::Expr* E, Stmts& block,
                             llvm::StringRef prefix = "_t",
                             bool forceDeclCreation = false);
    /// A shorthand to store directly to the current block.
    clang::Expr* StoreAndRef(clang::Expr* E,
                             llvm::StringRef prefix = "_t",
                             bool forceDeclCreation = false);
    /// An overload allowing to specify the type for the variable.
    clang::Expr* StoreAndRef(clang::Expr* E, clang::QualType Type, Stmts& block,
                             llvm::StringRef prefix = "_t",
                             bool forceDeclCreation = false);
    /// A flag for silencing warnings/errors output by diag function.
    bool silenceDiags = false;
    /// Shorthand to issues a warning or error.
    template <std::size_t N>
    void diag(clang::DiagnosticsEngine::Level level, // Warning or Error
              clang::SourceLocation loc,
              const char (&format)[N],
              llvm::ArrayRef<llvm::StringRef> args = {}) {
      if (!silenceDiags)
        m_Builder.diag(level, loc, format, args);
    }

    /// Creates unique identifier of the form "_nameBase<number>" that is
    /// guaranteed not to collide with anything in the current scope.
    clang::IdentifierInfo* CreateUniqueIdentifier(llvm::StringRef nameBase);
    std::unordered_map<std::string, std::size_t> m_idCtr;

    /// Updates references in newly cloned statements.
    void updateReferencesOf(clang::Stmt* InSubtree);
    /// Clones a statement
    clang::Stmt* Clone(const clang::Stmt* S);
    /// A shorthand to simplify cloning of expressions.
    clang::Expr* Clone(const clang::Expr* E);
    /// Parses the argument expression for the clad::differentiate/clad::gradient
    /// call. The argument is used to specify independent parameter(s) for
    /// differentiation. There are three valid options for the argument expression:
    ///   1) A string literal, containing comma-separated names of function's parameters,
    ///      as defined in function's defintion. The function will be differentiated
    ///      w.r.t. all the specified parameters.
    ///   2) A numeric literal. The function will be differentiated w.r.t. to the
    ///      parameter corresponding to literal's value index.
    ///   3) If no argument is provided, a default argument is used. The function
    ///      will be differentiated w.r.t. to its every parameter.
    DiffParams parseDiffArgs(const clang::Expr* diffArgs,
                             const clang::FunctionDecl* FD);

    /// Get an expression used to zero-initialize given type.
    /// Returns 0 for scalar types, otherwise {}.
    clang::Expr* getZeroInit(clang::QualType T);

    /// Split an array subscript expression into a pair of base expr and 
    /// a vector of all indices.
    std::pair<const clang::Expr*, llvm::SmallVector<const clang::Expr*, 4>>
    SplitArraySubscript(const clang::Expr* ASE);

    /// Build an array subscript expression with a given base expression and
    /// a sequence of indices.
    clang::Expr* BuildArraySubscript(clang::Expr* Base,
                                     const llvm::SmallVectorImpl<clang::Expr*> & IS);
    /// Find namespace clad declaration.
    clang::NamespaceDecl* GetCladNamespace();
    /// Find declaration of clad::tape templated type.
    clang::TemplateDecl* GetCladTapeDecl();
    /// Perform a lookup into clad namespace for an entity with given name.
    clang::LookupResult LookupCladTapeMethod(llvm::StringRef name);
    /// Perform lookup into clad namespace for push/pop/back. Returns 
    /// LookupResult, which is will be resolved later (which is handy since they
    /// are templates).
    clang::LookupResult& GetCladTapePush();
    clang::LookupResult& GetCladTapePop();
    clang::LookupResult& GetCladTapeBack();
    /// Instantiate clad::tape<T> type.
    clang::QualType GetCladTapeOfType(clang::QualType T);
  };
  /// A class that represents the result of Visit of ForwardModeVisitor.
  /// Stmt() allows to access the original (cloned) Stmt and Stmt_dx() allows
  /// to access its derivative (if exists, otherwise null). If Visit produces
  /// other (intermediate) statements, they are output to the current block.
  class StmtDiff {
  private:
    std::array<clang::Stmt*, 2> data;
  public:
    StmtDiff(clang::Stmt* orig = nullptr,
             clang::Stmt* diff = nullptr) {
      data[1] = orig;
      data[0] = diff;
    }

    clang::Stmt* getStmt() { return data[1]; }
    clang::Stmt* getStmt_dx() { return data[0]; }
    clang::Expr* getExpr() {
      return llvm::cast_or_null<clang::Expr>(getStmt());
    }
    clang::Expr* getExpr_dx() {
      return llvm::cast_or_null<clang::Expr>(getStmt_dx());
    }
    // Stmt_dx goes first!
    std::array<clang::Stmt*, 2>& getBothStmts() {
      return data;
    }
  };

  class VarDeclDiff {
  private:
    std::array<clang::VarDecl*, 2> data;
  public:
    VarDeclDiff(clang::VarDecl* orig = nullptr,
             clang::VarDecl* diff = nullptr) {
      data[1] = orig;
      data[0] = diff;
    }

    clang::VarDecl* getDecl() { return data[1]; }
    clang::VarDecl* getDecl_dx() { return data[0]; }
    // Decl_dx goes first!
    std::array<clang::VarDecl*, 2>& getBothDecls() {
      return data;
    }
  };

  /// A visitor for processing the function code in forward mode.
  /// Used to compute derivatives by clad::differentiate.
  class ForwardModeVisitor
    : public clang::ConstStmtVisitor<ForwardModeVisitor, StmtDiff>,
      public VisitorBase {
  private:
    const clang::VarDecl* m_IndependentVar = nullptr;
    unsigned m_DerivativeOrder = ~0;
    unsigned m_ArgIndex = ~0;

  public:
    ForwardModeVisitor(DerivativeBuilder& builder);
    ~ForwardModeVisitor();

    ///\brief Produces the first derivative of a given function.
    ///
    ///\param[in] FD - the function that will be differentiated.
    ///
    ///\returns The differentiated and potentially created enclosing
    /// context.
    ///
    DeclWithContext Derive(const clang::FunctionDecl* FD,
                           const DiffRequest& request);
    StmtDiff VisitArraySubscriptExpr(const clang::ArraySubscriptExpr* ASE);
    StmtDiff VisitBinaryOperator(const clang::BinaryOperator* BinOp);
    StmtDiff VisitCallExpr(const clang::CallExpr* CE);
    StmtDiff VisitCompoundStmt(const clang::CompoundStmt* CS);
    StmtDiff VisitConditionalOperator(const clang::ConditionalOperator* CO);
    StmtDiff VisitCXXBoolLiteralExpr(const clang::CXXBoolLiteralExpr* BL);
    StmtDiff VisitCXXDefaultArgExpr(const clang::CXXDefaultArgExpr* DE);
    StmtDiff VisitCXXOperatorCallExpr(const clang::CXXOperatorCallExpr* OpCall);
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
    // Decl is not Stmt, so it cannot be visited directly.
    VarDeclDiff DifferentiateVarDecl(const clang::VarDecl* VD);
  };

  /// A visitor for processing the function code in reverse mode.
  /// Used to compute derivatives by clad::gradient.
  class ReverseModeVisitor
    : public clang::ConstStmtVisitor<ReverseModeVisitor, StmtDiff>,
      public VisitorBase {
  private:
    llvm::SmallVector<clang::VarDecl*, 16> m_IndependentVars;
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
  public:
    clang::Expr* dfdx () {
      if (m_Stack.empty())
        return nullptr;
      return m_Stack.top();
    }
    StmtDiff Visit(const clang::Stmt* stmt, clang::Expr* dfdS = nullptr) {
      // No need to push the same expr multiple times.
      bool push = !(!m_Stack.empty() && (dfdS == dfdx()));
      if (push)
        m_Stack.push(dfdS);
      auto result = clang::ConstStmtVisitor<ReverseModeVisitor, StmtDiff>::Visit(stmt);
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
                             bool forceDeclCreation = false) {
      assert(E && "cannot infer type from null expression");
      return StoreAndRef(E, E->getType(), d, prefix, forceDeclCreation);
    }
    
    /// An overload allowing to specify the type for the variable.
    clang::Expr* StoreAndRef(clang::Expr* E, clang::QualType Type,
                             direction d = forward,
                             llvm::StringRef prefix = "_t",
                             bool forceDeclCreation = false) {
      // Name reverse temporaries as "_r" instead of "_t".
      if ((d == reverse) && (prefix == "_t"))
        prefix = "_r";
      return VisitorBase::StoreAndRef(E, Type, getCurrentBlock(d), prefix,
                                      forceDeclCreation);
    }

    /// For an expr E, decides if it is useful to store it in a global temporary
    /// variable and replace E's further usage by a reference to that variable to 
    /// avoid recomputiation.
    bool UsefulToStoreGlobal(clang::Expr* E);
    clang::VarDecl* GlobalStoreImpl(clang::QualType Type, llvm::StringRef prefix);
    /// Creates a (global in the function scope) variable declaration, puts
    /// it into m_Globals block (to be inserted into the beginning of fn's
    /// body). Returns reference R to the created declaration. If E is not null,
    /// puts an additional assignment statement (R = E) in the forward block.
    /// Alternatively, if isInsideLoop is true, stores E in a stack. Returns
    /// StmtDiff, where .getExpr() is intended to be used in forward pass and
    /// .getExpr_dx() in the reverse pass. Two expressions can be different in
    /// some cases, e.g. clad::push/pop inside loops.
    StmtDiff GlobalStoreAndRef(clang::Expr* E, clang::QualType Type,
                               llvm::StringRef prefix = "_t", bool force = false);
    StmtDiff GlobalStoreAndRef(clang::Expr* E, llvm::StringRef prefix = "_t",
                               bool force = false);

    //// A type returned by DelayedGlobalStoreAndRef
    /// .Result is a reference to the created (yet uninitialized) global variable.
    /// When the expression is finally visited and rebuilt, .Finalize must be
    /// called with new rebuilt expression, to initialize the global variable.
    /// Alternatively, expression may be not worth storing in a global varialbe 
    /// and is  easy to clone (e.g. it is a constant literal). Then .Result is 
    /// cloned E, .isConstant is true and .Finalize does nothing.
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

    /// If E is supposed to be stored in a tape, will create a global declaration
    /// of tape of corresponding type and return a result struct with reference
    /// to the tape and constructed calls to push/pop methods.
    CladTapeResult MakeCladTapeFor(clang::Expr* E);

  public:
    ReverseModeVisitor(DerivativeBuilder& builder);
    ~ReverseModeVisitor();

    ///\brief Produces the gradient of a given function.
    ///
    ///\param[in] FD - the function that will be differentiated.
    ///
    ///\returns The gradient of the function and potentially created enclosing
    /// context.
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
    /// Improved naming scheme is required. Hence, we append the indices to of the
    /// requested parameters to 'f_grad', i.e. in the previous example "x, y" will
    /// give 'f_grad_0_1' and "x, z" will give 'f_grad_0_2'.
    DeclWithContext Derive(const clang::FunctionDecl* FD,
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

  };
} // end namespace clad

#endif // CLAD_DERIVATIVE_BUILDER_H
