//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// version: $Id: ClangPlugin.cpp 7 2013-06-01 22:48:03Z v.g.vassilev@gmail.com $
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#ifndef CLAD_VISITOR_BASE_H
#define CLAD_VISITOR_BASE_H

namespace clad {
  class DerivativeBuilder;
}

#include "Compatibility.h"
#include "DerivativeBuilder.h"
#include "clad/Differentiator/DiffMode.h"

#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Sema/ParsedAttr.h"
#include "clang/Sema/Sema.h"

#include <array>
#include <stack>
#include <unordered_map>

namespace clad {
  /// A class that represents the result of Visit of ForwardModeVisitor.
  /// Stmt() allows to access the original (cloned) Stmt and Stmt_dx() allows
  /// to access its derivative (if exists, otherwise null). If Visit produces
  /// other (intermediate) statements, they are output to the current block.
  class StmtDiff {
  private:
    std::array<clang::Stmt*, 2> data;
    clang::Stmt* m_DerivativeForForwSweep;
  public:
    StmtDiff(clang::Stmt* orig = nullptr, clang::Stmt* diff = nullptr,
             clang::Stmt* forwSweepDiff = nullptr)
        : m_DerivativeForForwSweep(forwSweepDiff) {
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
    
    void updateStmt(clang::Stmt* S) { data[1] = S; }
    void updateStmtDx(clang::Stmt* S) { data[0] = S; }
    // Stmt_dx goes first!
    std::array<clang::Stmt*, 2>& getBothStmts() { return data; }

    clang::Stmt* getForwSweepStmt_dx() { return m_DerivativeForForwSweep; }

    clang::Expr* getForwSweepExpr_dx() {
      return llvm::cast_or_null<clang::Expr>(m_DerivativeForForwSweep);
    }

    void setForwSweepStmt_dx(clang::Stmt* S) { m_DerivativeForForwSweep = S; }
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
    std::array<clang::VarDecl*, 2>& getBothDecls() { return data; }
  };

  /// A base class for all common functionality for visitors
  class VisitorBase {
  protected:
    VisitorBase(DerivativeBuilder& builder)
        : m_Builder(builder), m_Sema(builder.m_Sema),
          m_CladPlugin(builder.m_CladPlugin), m_Context(builder.m_Context),
          m_CurScope(m_Sema.TUScope), m_DerivativeFnScope(nullptr),
          m_DerivativeInFlight(false), m_Derivative(nullptr),
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
    DiffMode m_Mode;
    /// Map used to keep track of variable declarations and match them
    /// with their derivatives.
    std::unordered_map<const clang::ValueDecl*, clang::Expr*> m_Variables;
    /// Map contains variable declarations replacements. If the original
    /// function contains a declaration which name collides with something
    /// already created inside derivative's body, the declaration is replaced
    /// with a new one.
    /// See the example inside ForwardModeVisitor::VisitDeclStmt.
    std::unordered_map<const clang::VarDecl*, clang::VarDecl*>
        m_DeclReplacements;
    /// A stack of all the blocks where the statements of the gradient function
    /// are stored (e.g., function body, if statement blocks).
    std::vector<Stmts> m_Blocks;
    /// Stores output variables for vector-valued functions
    VectorOutputs m_VectorOutput;
    /// The functor type that is currently being differentiated, if any.
    const clang::CXXRecordDecl* m_Functor = nullptr;
    /// Stores derivative expression of the implicit `this` pointer.
    ///
    /// In the forward mode, `this` pointer derivative expression is of pointer
    /// type. In the reverse mode, `this` pointer derivative expression is of
    /// object type.
    // FIXME: Fix this inconsistency, by making `this` pointer derivative
    // expression to be of object type in the reverse mode as well.
    clang::Expr* m_ThisExprDerivative = nullptr;
    /// A function used to wrap result of visiting E in a lambda. Returns a call
    /// to the built lambda. Func is a functor that will be invoked inside
    /// lambda scope and block. Statements inside lambda are expected to be
    /// added by addToCurrentBlock from func invocation.
    template <typename F>
    static clang::Expr* wrapInLambda(VisitorBase& V, clang::Sema& S,
                                     const clang::Expr* E, F&& func) {
      // FIXME: Here we use some of the things that are used from Parser, it
      // seems to be the easiest way to create lambda.
      clang::LambdaIntroducer Intro;
      Intro.Default = clang::LCD_ByRef;
      // FIXME: Using noLoc here results in assert failure. Any other valid
      // SourceLocation seems to work fine.
      Intro.Range.setBegin(E->getBeginLoc());
      Intro.Range.setEnd(E->getEndLoc());
      clang::AttributeFactory AttrFactory;
      const clang::DeclSpec DS(AttrFactory);
      clang::Declarator D(DS,
                          CLAD_COMPAT_CLANG15_Declarator_DeclarationAttrs_ExtraParam
                          CLAD_COMPAT_CLANG12_Declarator_LambdaExpr);
      S.PushLambdaScope();
      V.beginScope(clang::Scope::BlockScope | clang::Scope::FnScope |
                   clang::Scope::DeclScope);
      S.ActOnStartOfLambdaDefinition(Intro, D, V.getCurrentScope());
      V.beginBlock();
      func();
      clang::CompoundStmt* body = V.endBlock();
      clang::Expr* lambda =
          S.ActOnLambdaExpr(noLoc, body, V.getCurrentScope()).get();
      V.endScope();
      return S.ActOnCallExpr(V.getCurrentScope(), lambda, noLoc, {}, noLoc)
          .get();
    }
    /// For a qualtype QT returns if it's type is Array or Pointer Type
    static bool isArrayOrPointerType(const clang::QualType QT) {
      return QT->isArrayType() || QT->isPointerType();
    }

    clang::CompoundStmt* MakeCompoundStmt(const Stmts& Stmts);

    /// Get the latest block of code (i.e. place for statements output).
    Stmts& getCurrentBlock() { return m_Blocks.back(); }
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
    clang::Scope* getCurrentScope() { return m_CurScope; }
    /// Enters a new scope.
    void beginScope(unsigned ScopeFlags) {
      // FIXME: since Sema::CurScope is private, we cannot access it and have
      // to use separate member variable m_CurScope. The only options to set
      // CurScope of Sema seemt to be through Parser or ContextAndScopeRAII.
      m_CurScope =
          new clang::Scope(getCurrentScope(), ScopeFlags, m_Sema.Diags);
    }
    void endScope() {
      // This will remove all the decls in the scope from the IdResolver.
      m_Sema.ActOnPopScope(noLoc, m_CurScope);
      auto oldScope = m_CurScope;
      m_CurScope = oldScope->getParent();
      delete oldScope;
    }

    /// A shorthand to simplify syntax for creation of new expressions.
    /// This function uses m_Sema.BuildUnOp internally to build unary
    /// operations. Typical usage of this function looks like the following:
    /// \n \code
    /// auto postIncExpr = BuildOp(UO_PostInc, expr);
    /// auto assignExpr = BuildOp(BO_Assign, AsgnExpr, postIncExpr);
    /// addToCurrentBlock(assignExpr);
    /// \endcode
    /// \n The above will build the following expression:
    /// \n \code
    /// Asgn  = exp++;
    /// \endcode
    /// \param[in] OpCode The code for the unary operation to be built.
    /// \param[in] E The expression to build the unary operation with.
    /// \returns An expression of the newly built unary operation or null if the
    /// operand in null.
    clang::Expr* BuildOp(clang::UnaryOperatorKind OpCode, clang::Expr* E,
                         clang::SourceLocation OpLoc = noLoc);
    /// A shorthand to simplify syntax for creation of new expressions.
    /// This function uses m_Sema.BuildBin internally to build binary
    /// operations. A typical usage of this function looks like the following:
    /// \n \code
    /// auto mulExpr = BuildOp(BO_Mul, LExpr, RExpr);
    /// auto assignExpr = BuildOp(BO_Assign, AsgnExpr, mulExpr);
    /// addToCurrentBlock(assignExpr);
    /// \endcode
    /// \n The above will build the following expression:
    /// \n \code
    /// Asgn  = L * R;
    /// \endcode
    /// \param[in] OpCode The code for the binary operation to be built.
    /// \param[in] L The LHS expression to build the binary operation with.
    /// \param[in] R The RHS expression to build the binary operation with.
    /// \returns An expression of the newly built binary operation or null if
    /// either LHS or RHS is null.
    clang::Expr* BuildOp(clang::BinaryOperatorKind OpCode, clang::Expr* L,
                         clang::Expr* R, clang::SourceLocation OpLoc = noLoc);

    clang::Expr* BuildParens(clang::Expr* E);

    /// Builds variable declaration to be used inside the derivative
    /// body.
    /// \param[in] Type The type of variable declaration to build.
    /// \param[in] Identifier The identifier information for the variable
    /// declaration.
    /// \param[in] Init The initalization expression to assign to the variable
    ///  declaration.
    /// \param[in] DirectInit A check for if the initialization expression is a
    /// C style initalization.
    /// \param[in] TSI The type source information of the variable declaration.
    /// \returns The newly built variable declaration.
    clang::VarDecl*
    BuildVarDecl(clang::QualType Type, clang::IdentifierInfo* Identifier,
                 clang::Expr* Init = nullptr, bool DirectInit = false,
                 clang::TypeSourceInfo* TSI = nullptr,
                 clang::VarDecl::InitializationStyle IS =
                     clang::VarDecl::InitializationStyle::CInit);
    /// Builds variable declaration to be used inside the derivative
    /// body.
    /// \param[in] Type The type of variable declaration to build.
    /// \param[in] prefix The name of the variable declaration to build.
    /// \param[in] Init The initalization expression to assign to the variable
    ///  declaration.
    /// \param[in] DirectInit A check for if the initialization expression is a
    /// C style initalization.
    /// \param[in] TSI The type source information of the variable declaration.
    /// \returns The newly built variable declaration.
    clang::VarDecl*
    BuildVarDecl(clang::QualType Type, llvm::StringRef prefix = "_t",
                 clang::Expr* Init = nullptr, bool DirectInit = false,
                 clang::TypeSourceInfo* TSI = nullptr,
                 clang::VarDecl::InitializationStyle IS =
                     clang::VarDecl::InitializationStyle::CInit);
    /// Creates a namespace declaration and enters its context. All subsequent
    /// Stmts are built inside that namespace, until
    /// m_Sema.PopDeclContextIsUsed.
    clang::NamespaceDecl* BuildNamespaceDecl(clang::IdentifierInfo* II,
                                             bool isInline);
    /// Wraps a declaration in DeclStmt.
    /// \n Variable declaration cannot be added to code directly, instead we
    /// have to build a declaration staement.
    /// \param[in] D The declaration to build a declaration statement from.
    /// \returns The declration statement expression corresponding to the input
    /// variable declaration.
    clang::DeclStmt* BuildDeclStmt(clang::Decl* D);
    /// Wraps a set of declarations in a DeclStmt.
    /// \n This function is useful to wrap multiple variable declarations in one
    /// single declaration statement.
    /// \param[in] D The declarations to build a declaration statement from.
    /// \returns The declration statemetn expression corresponding to the input
    /// variable declaration.
    clang::DeclStmt* BuildDeclStmt(llvm::MutableArrayRef<clang::Decl*> DS);

    /// Builds a DeclRefExpr to a given Decl.
    /// \n To emit variables into code, we need to use their corresponding
    /// declaration reference expressions. This function builds a declaration
    /// reference given a declaration.
    /// \param[in] D The declaration to build a DeclRefExpr for.
    /// \returns the DeclRefExpr for the given declaration.
    clang::DeclRefExpr* BuildDeclRef(clang::DeclaratorDecl* D);

    /// Stores the result of an expression in a temporary variable (of the same
    /// type as is the result of the expression) and returns a reference to it.
    /// If force decl creation is true, this will allways create a temporary
    /// variable declaration. Otherwise, temporary variable is created only
    /// if E requires evaluation (e.g. there is no point to store literals or
    /// direct references in intermediate variables)
    clang::Expr* StoreAndRef(clang::Expr* E, Stmts& block,
                             llvm::StringRef prefix = "_t",
                             bool forceDeclCreation = false,
                             clang::VarDecl::InitializationStyle IS =
                                 clang::VarDecl::InitializationStyle::CInit);
    /// A shorthand to store directly to the current block.
    clang::Expr* StoreAndRef(clang::Expr* E, llvm::StringRef prefix = "_t",
                             bool forceDeclCreation = false,
                             clang::VarDecl::InitializationStyle IS =
                                 clang::VarDecl::InitializationStyle::CInit);
    /// An overload allowing to specify the type for the variable.
    clang::Expr* StoreAndRef(clang::Expr* E, clang::QualType Type, Stmts& block,
                             llvm::StringRef prefix = "_t",
                             bool forceDeclCreation = false,
                             clang::VarDecl::InitializationStyle IS =
                                 clang::VarDecl::InitializationStyle::CInit);
    /// A flag for silencing warnings/errors output by diag function.
    bool silenceDiags = false;
    /// Shorthand to issues a warning or error.
    template <std::size_t N>
    void diag(clang::DiagnosticsEngine::Level level, // Warning or Error
              clang::SourceLocation loc, const char (&format)[N],
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

    /// Get an expression used to zero-initialize given type.
    /// Returns 0 for scalar types, otherwise {}.
    clang::Expr* getZeroInit(clang::QualType T);

    /// Split an array subscript expression into a pair of base expr and
    /// a vector of all indices.
    std::pair<const clang::Expr*, llvm::SmallVector<const clang::Expr*, 4>>
    SplitArraySubscript(const clang::Expr* ASE);

    /// Build an array subscript expression with a given base expression and
    /// a sequence of indices.
    clang::Expr*
    BuildArraySubscript(clang::Expr* Base,
                        const llvm::SmallVectorImpl<clang::Expr*>& IS);
    /// Find namespace clad declaration.
    clang::NamespaceDecl* GetCladNamespace();
    /// Find declaration of clad::class templated type
    ///
    /// \param[in] className name of the class to be found
    /// \returns The declaration of the class with the name ClassName
    clang::TemplateDecl*
    LookupTemplateDeclInCladNamespace(llvm::StringRef ClassName);
    /// Instantiate clad::class<TemplateArgs> type
    ///
    /// \param[in] CladClassDecl the decl of the class that is going to be used
    /// in the creation of the type \param[in] TemplateArgs an array of template
    /// arguments \returns The created type clad::class<TemplateArgs>
    clang::QualType
    InstantiateTemplate(clang::TemplateDecl* CladClassDecl,
                        llvm::ArrayRef<clang::QualType> TemplateArgs);
    clang::QualType InstantiateTemplate(clang::TemplateDecl* CladClassDecl,
                                        clang::TemplateArgumentListInfo& TLI);
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

    clang::DeclRefExpr* GetCladTapePushDRE();

    /// Assigns the Init expression to VD after performing the necessary
    /// implicit conversion. This is required as clang doesn't add implicit
    /// conversions while assigning values to variables which are initialized
    /// after it is already declared.
    void PerformImplicitConversionAndAssign(clang::VarDecl* VD,
                                            clang::Expr* Init) {
      // Implicitly convert Init into the type of VD
      clang::ActionResult<clang::Expr*> ICAR = m_Sema
          .PerformImplicitConversion(Init, VD->getType(),
                                     clang::Sema::AA_Casting);
      assert(!ICAR.isInvalid() && "Invalid implicit conversion!");
      // Assign the resulting expression to the variable declaration
      VD->setInit(ICAR.get());
    }

    /// Build a call to member function through Base expr and using the function
    /// name.
    ///
    /// \param[in] Base expr to the object which is used to call the member
    ///  function
    /// \param[in] isArrow if true specifies that the member function is
    /// accessed by an -> otherwise .
    /// \param[in] MemberFunctionName the name of the member function
    /// \param[in] ArgExprs the arguments to be used when calling the member
    ///  function
    /// \returns Built member function call expression
    ///  Base.MemberFunction(ArgExprs) or Base->MemberFunction(ArgExprs)
    clang::Expr*
    BuildCallExprToMemFn(clang::Expr* Base, llvm::StringRef MemberFunctionName,
                         llvm::MutableArrayRef<clang::Expr*> ArgExprs,
                         clang::ValueDecl* memberDecl = nullptr);

    /// Build a call to member function through this pointer.
    ///
    /// \param[in] FD callee member function
    /// \param[in] argExprs function arguments expressions
    /// \param[in] useRefQualifiedThisObj If true, then the `this` object is
    /// perfectly forwarded while calling member functions.
    /// \returns Built member function call expression
    clang::Expr*
    BuildCallExprToMemFn(clang::CXXMethodDecl* FD,
                         llvm::MutableArrayRef<clang::Expr*> argExprs,
                         bool useRefQualifiedThisObj = false);

    /// Build a call to a free function or member function through
    /// this pointer depending on whether the `FD` argument corresponds to a
    /// free function or a member function.
    ///
    /// \param[in] FD callee function
    /// \param[in] argExprs function arguments expressions
    /// \param[in] useRefQualifiedThisObj If true, then the `this` object is
    /// perfectly forwarded while calling member functions.
    /// \returns Built call expression
    clang::Expr*
    BuildCallExprToFunction(clang::FunctionDecl* FD,
                            llvm::MutableArrayRef<clang::Expr*> argExprs,
                            bool useRefQualifiedThisObj = false);
    /// Find declaration of clad::array_ref templated type.
    clang::TemplateDecl* GetCladArrayRefDecl();
    /// Create clad::array_ref<T> type.
    clang::QualType GetCladArrayRefOfType(clang::QualType T);
    /// Checks if the type is of clad::array<T> or clad::array_ref<T> type
    bool isCladArrayType(clang::QualType QT);
    /// Find declaration of clad::array templated type.
    clang::TemplateDecl* GetCladArrayDecl();
    /// Create clad::array<T> type.
    clang::QualType GetCladArrayOfType(clang::QualType T);
    /// Creates the expression Base.size() for the given Base expr. The Base
    /// expr must be of clad::array_ref<T> type
    clang::Expr* BuildArrayRefSizeExpr(clang::Expr* Base);
    /// Checks if the type is of clad::ValueAndPushforward<T,U> type
    bool isCladValueAndPushforwardType(clang::QualType QT);
    /// Creates the expression Base.slice(Args) for the given Base expr and Args
    /// array. The Base expr must be of clad::array_ref<T> type
    clang::Expr*
    BuildArrayRefSliceExpr(clang::Expr* Base,
                           llvm::MutableArrayRef<clang::Expr*> Args);
    clang::ParmVarDecl* CloneParmVarDecl(const clang::ParmVarDecl* PVD,
                                         clang::IdentifierInfo* II,
                                         bool pushOnScopeChains = false,
                                         bool cloneDefaultArg = true);
    /// A function to get the single argument "forward_central_difference"
    /// call expression for the given arguments.
    ///
    /// \param[in] targetFuncCall The function to get the derivative for.
    /// \param[in] targetArg The argument to get the derivative with respect to.
    /// \param[in] targetPos The relative position of 'targetArg'.
    /// \param[in] numArgs The total number of 'args'.
    /// \param[in] args All the arguments to the target function.
    ///
    /// \returns The derivative function call.
    clang::Expr* GetSingleArgCentralDiffCall(
        clang::Expr* targetFuncCall, clang::Expr* targetArg, 
        unsigned targetPos, unsigned numArgs, 
        llvm::SmallVectorImpl<clang::Expr*>& args);
    /// A function to get the multi-argument "central_difference"
    /// call expression for the given arguments.
    ///
    /// \param[in] targetFuncCall The function to get the derivative for.
    /// \param[in] retType The return type of the target call expression.
    /// \param[in] numArgs The total number of 'args'.
    /// \param[in] NumericalDiffMultiArg The built statements to add to block
    /// later.
    /// \param[in] args All the arguments to the target function.
    /// \param[in] outputArgs The output gradient arguments.
    ///
    /// \returns The derivative function call.
    clang::Expr* GetMultiArgCentralDiffCall(
        clang::Expr* targetFuncCall, clang::QualType retType, 
        unsigned numArgs,
        llvm::SmallVectorImpl<clang::Stmt*>& NumericalDiffMultiArg,
        llvm::SmallVectorImpl<clang::Expr*>& args,
        llvm::SmallVectorImpl<clang::Expr*>& outputArgs);
    /// Emits diagnostic messages on differentiation (or lack thereof) for 
    /// call expressions.
    ///
    /// \param[in] \c funcName The name of the underlying function of the 
    /// call expression.
    /// \param[in] \c srcLoc Any associated source location information. 
    /// \param[in] \c isDerived A flag to determine if differentiation of the
    /// call expression was successful.
    void CallExprDiffDiagnostics(llvm::StringRef funcName,
                                 clang::SourceLocation srcLoc,
                                 bool isDerived);

    clang::QualType DetermineCladArrayValueType(clang::QualType T);
  public:
    /// Rebuild a sequence of nested namespaces ending with DC.
    clang::NamespaceDecl* RebuildEnclosingNamespaces(clang::DeclContext* DC);
    /// Clones a statement
    clang::Stmt* Clone(const clang::Stmt* S);
    /// A shorthand to simplify cloning of expressions.
    clang::Expr* Clone(const clang::Expr* E);
  };
} // end namespace clad

#endif // CLAD_VISITOR_BASE_H
