//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// version: $Id: ClangPlugin.cpp 7 2013-06-01 22:48:03Z v.g.vassilev@gmail.com $
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#ifndef CLAD_REVERSE_MODE_VISITOR_H
#define CLAD_REVERSE_MODE_VISITOR_H

#include "clad/Differentiator/Compatibility.h"
#include "clad/Differentiator/VisitorBase.h"
#include "clad/Differentiator/ReverseModeVisitorDirectionKinds.h"
#include "clad/Differentiator/ParseDiffArgsTypes.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Sema/Sema.h"

#include <array>
#include <memory>
#include <stack>
#include <unordered_map>

namespace clad {
  class ErrorEstimationHandler;
  class ExternalRMVSource;
  class MultiplexExternalRMVSource;

  /// A visitor for processing the function code in reverse mode.
  /// Used to compute derivatives by clad::gradient.
  class ReverseModeVisitor
      : public clang::ConstStmtVisitor<ReverseModeVisitor, StmtDiff>,
        public VisitorBase {
  
  private:
    // FIXME: We should remove friend-dependency of the plugin classes here.
    // For this we will need to separate out AST related functions in
    // a separate namespace, as well as add getters/setters function of
    // several private/protected members of the visitor classes.
    friend class ErrorEstimationHandler;
    llvm::SmallVector<const clang::ValueDecl*, 16> m_IndependentVars;
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
    /// Stores the pop index values for arrays in reverse mode.This is required
    /// to maintain the correct statement order when the current block has
    /// delayed emission i.e. assignment LHS.
    Stmts m_PopIdxValues;
    std::vector<Stmts> m_LoopBlock;
    unsigned outputArrayCursor = 0;
    unsigned numParams = 0;
    bool isVectorValued = false;
    bool use_enzyme = false;
    // FIXME: Should we make this an object instead of a pointer?
    // Downside of making it an object: We will need to include
    // 'MultiplexExternalRMVSource.h' file
    MultiplexExternalRMVSource* m_ExternalSource = nullptr;
    clang::Expr* m_Pullback = nullptr;
    const char* funcPostfix() const {
      if (isVectorValued)
        return "_jac";
      else if (use_enzyme)
        return "_grad_enzyme";
      else
        return "_grad";
    }

    /// Removes the local const qualifiers from a QualType and returns a new
    /// type.
    static clang::QualType
    getNonConstType(clang::QualType T, clang::ASTContext& C, clang::Sema& S) {
        clang::Qualifiers quals(T.getQualifiers());
        quals.removeConst();
        return S.BuildQualifiedType(T.getUnqualifiedType(), noLoc, quals);
    }
    // Function to Differentiate with Clad as Backend
    void DifferentiateWithClad();

    // Function to Differentiate with Enzyme as Backend
    void DifferentiateWithEnzyme();

  public:
    using direction = rmv::direction;
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

    /// This visit method explicitly sets `dfdx` to `nullptr` for this visit.
    ///
    /// This method is helpful when we need derivative of some expression but we
    /// do not want `_d_expression += dfdx` statments to be (automatically)
    /// added.
    ///
    /// FIXME: Think of a better way for handling this situation. Maybe we
    /// should improve the overall dfdx design and approach. One other way of
    /// designing `VisitWithExplicitNoDfDx` in a more general way is
    /// to develop a function that takes an expression E and returns the
    /// corresponding derivative without any side effects. The difference
    /// between this function and the current `VisitWithExplicitNoDfDx` will be
    /// 1) better intent through the function name 2) We will also get
    /// derivatives of expressions other than `DeclRefExpr` and `MemberExpr`.
    StmtDiff VisitWithExplicitNoDfDx(const clang::Stmt* stmt) {
      m_Stack.push(nullptr);
      auto result =
          clang::ConstStmtVisitor<ReverseModeVisitor, StmtDiff>::Visit(stmt);
      m_Stack.pop();
      return result;
    }

    /// Get the latest block of code (i.e. place for statements output).
    Stmts& getCurrentBlock(direction d = direction::forward) {
      if (d == direction::forward)
        return m_Blocks.back();
      else
        return m_Reverse.back();
    }
    /// Create new block.
    Stmts& beginBlock(direction d = direction::forward) {
      if (d == direction::forward)
        m_Blocks.push_back({});
      else
        m_Reverse.push_back({});
      return getCurrentBlock(d);
    }
    /// Remove the block from the stack, wrap it in CompoundStmt and return it.
    clang::CompoundStmt* endBlock(direction d = direction::forward) {
      if (d == direction::forward) {
        auto CS = MakeCompoundStmt(getCurrentBlock(direction::forward));
        m_Blocks.pop_back();
        return CS;
      } else {
        auto CS = MakeCompoundStmt(getCurrentBlock(direction::reverse));
        std::reverse(CS->body_begin(), CS->body_end());
        m_Reverse.pop_back();
        return CS;
      }
    }
    /// Output a statement to the current block. If Stmt is null or is an unused
    /// expression, it is not output and false is returned.
    bool addToCurrentBlock(clang::Stmt* S, direction d = direction::forward) {
      return addToBlock(S, getCurrentBlock(d));
    }

    /// Adds a given statement to the global block.
    ///
    /// \param[in] S The statement to add to the block.
    ///
    /// \returns True if the statement was added to the block, false otherwise.
    bool AddToGlobalBlock(clang::Stmt* S) { return addToBlock(S, m_Globals); }

    /// Stores the result of an expression in a temporary variable (of the same
    /// type as is the result of the expression) and returns a reference to it.
    /// If force decl creation is true, this will allways create a temporary
    /// variable declaration. Otherwise, temporary variable is created only
    /// if E requires evaluation (e.g. there is no point to store literals or
    /// direct references in intermediate variables)
    clang::Expr* StoreAndRef(clang::Expr* E, direction d = direction::forward,
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
                             direction d = direction::forward,
                             llvm::StringRef prefix = "_t",
                             bool forceDeclCreation = false,
                             clang::VarDecl::InitializationStyle IS =
                                 clang::VarDecl::InitializationStyle::CInit) {
      // Name reverse temporaries as "_r" instead of "_t".
      if ((d == direction::reverse) && (prefix == "_t"))
        prefix = "_r";
      return VisitorBase::StoreAndRef(E, Type, getCurrentBlock(d), prefix,
                                      forceDeclCreation, IS);
    }

    /// For an expr E, decides if it is useful to store it in a global temporary
    /// variable and replace E's further usage by a reference to that variable
    /// to avoid recomputiation.
    bool UsefulToStoreGlobal(clang::Expr* E);

    /// Builds a variable declaration and stores it in the function
    /// global scope.
    ///
    /// \param[in] Type The type of variable declaration to build.
    ///
    /// \param[in] prefix The prefix (if any) to the declration name.
    ///
    /// \param[in] init The variable declaration initializer.
    ///
    /// \returns A variable declaration that is already added to the
    /// global scope.
    clang::VarDecl* GlobalStoreImpl(clang::QualType Type,
                                    llvm::StringRef prefix,
                                    clang::Expr* init = nullptr);
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

    /// Make a clad::tape to store variables.
    /// If E is supposed to be stored in a tape, will create a global
    /// declaration of tape of corresponding type and return a result struct
    /// with reference to the tape and constructed calls to push/pop methods.
    ///
    /// \param[in] E The expression to build the tape for.
    ///
    /// \param[in] prefix The prefix value for the name of the tape.
    ///
    /// \returns A struct containg necessary call expressions for the built
    /// tape
    CladTapeResult MakeCladTapeFor(clang::Expr* E,
                                   llvm::StringRef prefix = "_t");

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
    DerivativeAndOverload Derive(const clang::FunctionDecl* FD,
                                 const DiffRequest& request);
    DerivativeAndOverload DerivePullback(const clang::FunctionDecl* FD,
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
    StmtDiff VisitWhileStmt(const clang::WhileStmt* WS);
    StmtDiff VisitDoStmt(const clang::DoStmt* DS);
    StmtDiff VisitContinueStmt(const clang::ContinueStmt* CS);
    StmtDiff VisitBreakStmt(const clang::BreakStmt* BS);
    StmtDiff VisitCXXThisExpr(const clang::CXXThisExpr* CTE);
    StmtDiff VisitCXXConstructExpr(const clang::CXXConstructExpr* CE);
    StmtDiff
    VisitMaterializeTemporaryExpr(const clang::MaterializeTemporaryExpr* MTE);
    StmtDiff VisitCXXStaticCastExpr(const clang::CXXStaticCastExpr* SCE);
    VarDeclDiff DifferentiateVarDecl(const clang::VarDecl* VD);

    /// A helper method to differentiate a single Stmt in the reverse mode.
    /// Internally, calls Visit(S, expr). Its result is wrapped into a
    /// CompoundStmt (if several statements are created) and proper Stmt
    /// order is maintained.
    ///
    /// \param[in] S The statement to differentiate.
    ///
    /// \param[in] dfdS The expression to propogate to Visit
    ///
    /// \returns The orignal (cloned) and differentiated forms of S
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
    clang::FunctionDecl* CreateGradientOverload();

    /// Returns the type that should be used to represent the derivative of a
    /// variable of type `yType` with respect to a parameter variable of type
    /// `xType`.
    ///
    /// FIXME: Parameter derivative type rules are different from the derivative
    /// type rules for local variables. We should remove this inconsistency.
    /// See the following issue for more details:
    /// https://github.com/vgvassilev/clad/issues/385
    clang::QualType GetParameterDerivativeType(clang::QualType yType,
                                               clang::QualType xType);

    /// Allows to easily create and manage a counter for counting the number of
    /// executed iterations of a loop. 
    ///
    /// It is required to save the number of executed iterations to use the
    /// same number of iterations in the reverse pass.
    /// If we are currently inside a loop, then a clad tape object is created
    /// to be used as the counter; otherwise, a temporary global variable (in
    /// function scope) is created to be used as the counter.
    class LoopCounter {
      clang::Expr *m_Ref = nullptr;
      clang::Expr *m_Pop = nullptr;
      clang::Expr *m_Push = nullptr;
      ReverseModeVisitor& m_RMV;

    public:
      LoopCounter(ReverseModeVisitor& RMV);
      /// Returns `clad::push(_t, 0UL)` expression if clad tape is used
      /// for counter; otherwise, returns nullptr.
      clang::Expr* getPush() const { return m_Push; }

      /// Returns `clad::pop(_t)` expression if clad tape is used for 
      /// for counter; otherwise, returns nullptr.
      clang::Expr* getPop() const { return m_Pop; }

      /// Returns reference to the last object of the clad tape if clad tape 
      /// is used as the counter; otherwise returns reference to the counter
      /// variable.
      clang::Expr* getRef() const { return m_Ref; }

      /// Returns counter post-increment expression (`counter++`).
      clang::Expr* getCounterIncrement() {
        return m_RMV.BuildOp(clang::UnaryOperatorKind::UO_PostInc, m_Ref);
      }

      /// Returns counter post-decrement expression (`counter--`)
      clang::Expr* getCounterDecrement() {
        return m_RMV.BuildOp(clang::UnaryOperatorKind::UO_PostDec, m_Ref);
      }

      /// Returns `ConditionResult` object for the counter.
      clang::Sema::ConditionResult getCounterConditionResult() {
        return m_RMV.m_Sema.ActOnCondition(m_RMV.m_CurScope, noLoc, m_Ref,
                                           clang::Sema::ConditionKind::Boolean);
      }
    };

    /// Helper function to differentiate a loop body.
    ///
    ///\param[in] body body of the loop
    ///\param[in] loopCounter associated `LoopCounter` object of the loop.
    ///\param[in] condVarDiff derived statements of the condition
    /// variable, if any.
    ///\param[in] forLoopIncDiff derived statements of the `for` loop
    /// increment statement, if any.
    ///\param[in] isForLoop should be true if we are differentiating a `for`
    /// loop body; otherwise false.
    ///\returns {forward pass statements, reverse pass statements} for the loop
    /// body.
    StmtDiff DifferentiateLoopBody(const clang::Stmt* body,
                                   LoopCounter& loopCounter,
                                   clang::Stmt* condVarDifff = nullptr,
                                   clang::Stmt* forLoopIncDiff = nullptr,
                                   bool isForLoop = false);

    /// This class modifies forward and reverse blocks of the loop
    /// body so that `break` and `continue` statements are correctly
    /// handled. `break` and `continue` statements are handled by 
    /// enclosing entire reverse block loop body in a switch statement
    /// and only executing the statements, with the help of case labels,
    /// that were executed in the associated forward iteration. This is 
    /// determined by keeping track of which `break`/`continue` statement 
    /// was hit in which iteration and that in turn helps to determine which
    /// case label should be selected.
    ///
    /// Class usage:
    ///
    /// ```cpp
    /// auto activeBreakContStmtHandler = PushBreakContStmtHandler();
    /// activeBreakContHandler->BeginCFSwitchStmtScope();
    /// ....
    /// Differentiate loop body, and save results in StmtDiff BodyDiff
    /// ...
    /// activeBreakContHandler->EndCFSwitchStmtScope();
    /// activeBreakContHandler->UpdateForwAndRevBlocks(bodyDiff);
    /// PopBreakContStmtHandler();
    /// ```
    class BreakContStmtHandler {
      /// Keeps track of all the created switch cases. It is required
      /// because we need to register all the switch cases later with the
      /// switch statement that will be used to manage the control flow in
      /// the reverse block.
      llvm::SmallVector<clang::SwitchCase*, 4> m_SwitchCases;

      /// `m_ControlFlowTape` tape keeps track of which `break`/`continue`
      /// statement was hit in which iteration.
      /// \note `m_ControlFlowTape` is only initialized if the body contains
      /// `continue` or `break` statement.
      std::unique_ptr<CladTapeResult> m_ControlFlowTape;
      
      /// Each `break` and `continue` statement is assigned a unique number,
      /// starting from 1, that is used as the case label corresponding to that `break`/`continue`
      /// statement. `m_CaseCounter` stores the value that was used for last
      /// `break`/`continue` statement.
      std::size_t m_CaseCounter = 0;

      ReverseModeVisitor& m_RMV;

      /// Builds and returns a literal expression of type `std::size_t` with
      /// `value` as value.
      clang::Expr* CreateSizeTLiteralExpr(std::size_t value);

      /// Initialise the `m_ControlFlowTape`.
      /// \note `m_ControlFlowTape` is not initialised in the constructor
      /// because it is only initialised if it is required. It is only required
      /// if body contains `break` or `continue` statement.
      void InitializeCFTape();

      /// Builds and returns `clad::push(tapeRef, value)` expression.
      clang::Expr* CreateCFTapePushExpr(std::size_t value);

    public:
      BreakContStmtHandler(ReverseModeVisitor& RMV) : m_RMV(RMV) {}

      /// Begins control flow switch statement scope.
      /// Control flow switch statement is used to refer to the
      /// switch statement that manages the control flow of the reverse
      /// block.
      void BeginCFSwitchStmtScope() const;

      /// Ends control flow switch statement scope.
      void EndCFSwitchStmtScope() const;

      /// Builds and returns a switch case statement that corresponds
      /// to a `break` or `continue` statement and is registered in the
      /// control flow switch statement.
      clang::CaseStmt* GetNextCFCaseStmt();

      /// Builds and returns `clad::push(TapeRef, m_CurrentCounter)` 
      /// expression, where `TapeRef` and `m_CurrentCounter` are replaced
      /// by their actual values respectively.
      clang::Stmt* CreateCFTapePushExprToCurrentCase();

      /// Does final modifications on forward and reverse blocks
      /// so that `break` and `continue` statements are handled
      /// accurately.
      void UpdateForwAndRevBlocks(StmtDiff& bodyDiff);
    };
    // Keeps track of active control flow switch statements.
    llvm::SmallVector<BreakContStmtHandler, 4> m_BreakContStmtHandlers;

    BreakContStmtHandler* GetActiveBreakContStmtHandler() {
      return &m_BreakContStmtHandlers.back();
    }
    BreakContStmtHandler* PushBreakContStmtHandler() {
      m_BreakContStmtHandlers.emplace_back(*this);
      return &m_BreakContStmtHandlers.back();
    }
    void PopBreakContStmtHandler() {
      m_BreakContStmtHandlers.pop_back();
    }
                                   
    /// Registers an external RMV source.
    ///
    /// Multiple external RMV source can be registered by calling this function
    /// multiple times.
    ///\paramp[in] source An external RMV source
    void AddExternalSource(ExternalRMVSource& source);

    /// Computes and returns the sequence of derived function parameter types.
    ///
    /// Information about the original function and the differentiation mode
    /// are taken from the data member variables. In particular, `m_Function`,
    /// `m_Mode` data members should be correctly set before using this
    /// function.
    llvm::SmallVector<clang::QualType, 8> ComputeParamTypes(const DiffParams& diffParams);

    /// Builds and returns the sequence of derived function parameters.
    ///
    /// Information about the original function, derived function, derived
    /// function parameter types and the differentiation mode are implicitly
    /// taken from the data member variables. In particular, `m_Function`,
    /// `m_Mode` and `m_Derivative` should be correctly set before using this
    /// function.
    llvm::SmallVector<clang::ParmVarDecl*, 8>
    BuildParams(DiffParams& diffParams);

    clang::QualType ComputeAdjointType(clang::QualType T);
    clang::QualType ComputeParamType(clang::QualType T);
  };
} // end namespace clad

#endif // CLAD_REVERSE_MODE_VISITOR_H
