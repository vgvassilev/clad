// This file contains utility functions which do not belong anywhere else

#ifndef CLAD_UTILS_CLADUTILS_H
#define CLAD_UTILS_CLADUTILS_H

#include "DiffMode.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclarationName.h"
#include "clang/AST/Type.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Sema/Ownership.h"
#include "clang/Sema/Sema.h"
#include "llvm/ADT/StringRef.h"

#include <string>

namespace clang {
  class ASTContext;
  class FunctionDecl;
  class StringLiteral;
}

namespace clad {
  namespace utils {
    /// If `FD` is an overloaded operator, returns a name, unique for
    /// each operator, that can be used to create valid C++ identifiers.
    /// Otherwise if `FD` is an ordinary function, returns the name of the
    /// function `FD`.
    std::string ComputeEffectiveFnName(const clang::FunctionDecl* FD);

    // Unwraps S to a single statement if it's a compound statement only
    // containing 1 statement.
    clang::Stmt* unwrapIfSingleStmt(clang::Stmt* S);

    /// Creates and returns a compound statement having statements as follows:
    /// {`S`, all the statement of `initial` in sequence}
    clang::CompoundStmt* PrependAndCreateCompoundStmt(clang::ASTContext& C,
                                                      clang::Stmt* initial,
                                                      clang::Stmt* S);

    /// Creates and returns a compound statement having statements as follows:
    /// {all the statements of `initial` in sequence, `S`}
    clang::CompoundStmt* AppendAndCreateCompoundStmt(clang::ASTContext& C,
                                                     clang::Stmt* initial,
                                                     clang::Stmt* S);

    /// Shorthand to issues a warning or error.
    template <std::size_t N>
    void EmitDiag(clang::Sema& semaRef,
              clang::DiagnosticsEngine::Level level, // Warning or Error
              clang::SourceLocation loc, const char (&format)[N],
              llvm::ArrayRef<llvm::StringRef> args = {}) {
      unsigned diagID = semaRef.Diags.getCustomDiagID(level, format);
      clang::Sema::SemaDiagnosticBuilder stream = semaRef.Diag(loc, diagID);
      for (auto arg : args)
        stream << arg;
    }

    /// Creates nested name specifier associated with declaration context
    /// argument `DC`.
    ///
    /// For example, given a structure defined as,
    /// namespace A {
    /// namespace B {
    ///   struct SomeStruct {};
    /// }
    /// }
    ///
    /// Passing `SomeStruct` as declaration context will create
    /// nested name specifier of the form `::A::B::struct SomeClass::` or
    /// `A::B::struct SomeClass::` depending on if `addGlobalNS` is true or
    /// false in the `CXXScopeSpec` argument `CSS`.
    ///
    /// \note Currently only namespace and class/struct nested name specifiers
    /// are supported.
    ///
    /// \param[in] DC
    /// \param[out] CSS
    /// \param[in] addGlobalNS if true, then the global namespace specifier will
    /// be added as well.
    void BuildNNS(clang::Sema& semaRef, clang::DeclContext* DC,
                  clang::CXXScopeSpec& CSS, bool addGlobalNS = false);

    /// Add the namespace specifier to the type if it is not already an elaborated type.
    /// For example, if the type is `SomeClass` and it is declared in namespace `A::B`,
    /// as:
    /// ```
    /// namespace A {
    ///  namespace B {
    ///    struct SomeClass {};
    ///  }
    /// }
    /// ```
    /// then the function will return `A::B::SomeClass`.
    /// If the type is already an elaborated type, then it is returned as is.
    ///
    /// \param semaRef
    /// \param[in] C
    /// \param[in] QT
    /// \returns  type with namespace specifier added.
    clang::QualType AddNamespaceSpecifier(clang::Sema& semaRef, clang::ASTContext& C, clang::QualType QT);

    /// Finds declaration context associated with the DC1::DC2, but doesn't
    /// replicate the common part of the declaration contexts.
    /// For example, consider DC1 corresponds to the following declaration
    /// context:
    ///
    /// ```
    /// namespace custom_derivatives {}
    /// ```
    ///
    /// and DC2 corresponds to the following declaration context:
    /// ```
    /// namespace custom_derivatives {
    ///   namespace A {
    ///     namespace B {}
    ///   }
    /// }
    /// ```
    /// then the function returns declartion context that correponds to
    /// `custom_derivatives::A::B::`
    ///
    /// \param semaRef
    /// \param[in] DC1
    /// \param[in] DC2
    /// \returns found declaration context corresponding to `DC1::DC2`, if no
    /// such declaration context is found, then returns `nullptr`.
    clang::DeclContext* FindDeclContext(clang::Sema& semaRef,
                                        clang::DeclContext* DC1,
                                        const clang::DeclContext* DC2);

    /// Finds the qualified name `name` in the declaration context `DC`.
    ///
    /// \param[in] name
    /// \param[in] S
    /// \param[in] DC
    /// \returns lookup result.
    clang::LookupResult LookupQualifiedName(llvm::StringRef name,
                                            clang::Sema& S,
                                            clang::DeclContext* DC = nullptr);

    /// Finds namespace 'namespc` under the declaration context `DC` or the
    /// translation unit declaration if `DC` is null.
    ///
    /// \param S
    /// \param namespc
    /// \param shouldExist If true, then asserts that the specified namespace
    /// is found.
    /// \param DC
    clang::NamespaceDecl* LookupNSD(clang::Sema& S, llvm::StringRef namespc,
                                    bool shouldExist,
                                    clang::DeclContext* DC = nullptr);

    /// Creates a `StringLiteral` node to represent string literal
    /// "`str`".
    ///
    ///\param C reference to `ASTContext` object.
    ///\param[in] str string literal to create.
    clang::StringLiteral* CreateStringLiteral(clang::ASTContext& C,
                                              llvm::StringRef str);

    /// Returns true if `QT` is Array or Pointer Type, otherwise returns false.
    bool isArrayOrPointerType(clang::QualType QT);

    /// Returns true if `T` is auto or auto* type, otherwise returns false.
    bool IsAutoOrAutoPtrType(clang::QualType T);

    clang::DeclarationNameInfo BuildDeclarationNameInfo(clang::Sema& S,
                                                        llvm::StringRef name);

    /// Returns true if the function has any reference or pointer parameter;
    /// otherwise returns false.
    bool HasAnyReferenceOrPointerArgument(const clang::FunctionDecl* FD);

    /// Returns true if `arg` is an argument passed by reference or is of
    /// pointer/array type.
    ///
    /// \note Please note that this function returns false for temporary
    /// expressions.
    bool IsReferenceOrPointerArg(const clang::Expr* arg);

    /// Returns true if `T1` and `T2` have same cononical type; otherwise
    /// returns false.
    bool isSameCanonicalType(clang::QualType T1, clang::QualType T2);

    /// Builds `base->member` expression or `base.member` expression depending
    /// on if the `base` is of pointer type or not.
    clang::MemberExpr* BuildMemberExpr(clang::Sema& semaRef, clang::Scope* S,
                                       clang::Expr* base,
                                       llvm::StringRef memberName);

    /// Returns a valid `SourceLocation` to be used in places where clang
    /// requires a valid `SourceLocation`.
    clang::SourceLocation GetValidSLoc(clang::Sema& semaRef);

    /// Given an expression `E`, this function builds and returns the expression
    /// `(E)`.
    clang::ParenExpr* BuildParenExpr(clang::Sema& semaRef, clang::Expr* E);

    /// Returns `IdentifierInfo` that represents the value in the `identifier`
    /// parameter.
    clang::IdentifierInfo* GetIdentifierInfo(clang::Sema& semaRef,
                                             llvm::StringRef identifier);

    /// Builds parameter variable declaration.
    ///
    /// This function is just a convenient routine that internally calls
    /// `clang::ParmVarDecl::Create`.
    ///
    /// \note `TSI` parameter only needs to be provided if the type should be
    /// represented exactly how it was represented in the source code.
    clang::ParmVarDecl*
    BuildParmVarDecl(clang::Sema& semaRef, clang::DeclContext* DC,
                     clang::IdentifierInfo* II, clang::QualType T,
                     clang::StorageClass SC = clang::StorageClass::SC_None,
                     clang::Expr* defArg = nullptr,
                     clang::TypeSourceInfo* TSI = nullptr,
                     clang::SourceLocation Loc = clang::SourceLocation());

    /// If `T` represents an array or a pointer type then returns the
    /// corresponding array element or the pointee type. If `T` is a reference
    /// type then return the corresponding non-reference type. Otherwise, if `T`
    /// is neither an array nor a pointer type, then simply returns `T`.
    clang::QualType GetValueType(clang::QualType T);

    /// Returns the same type as GetValueType but without const qualifier.
    clang::QualType GetNonConstValueType(clang::QualType T);

    clang::QualType getNonConstType(clang::QualType T, clang::Sema& S);

    /// Builds and returns the init expression to initialise `clad::array` and
    /// `clad::array_ref` from a constant array.
    ///
    /// More concretely, it builds the following init list expression:
    /// `{arr, arrSize}`
    clang::Expr* BuildCladArrayInitByConstArray(clang::Sema& semaRef,
                                                clang::Expr* constArrE);

    /// Returns true if `FD` is a class static method; otherwise returns
    /// false.
    bool IsStaticMethod(const clang::FunctionDecl* FD);

    bool IsCladValueAndPushforwardType(clang::QualType T);

    /// Returns a valid `SourceRange` to be used in places where clang
    /// requires a valid `SourceRange`.
    clang::SourceRange GetValidSRange(clang::Sema& semaRef);

    /// Builds and returns `new` expression.
    ///
    /// This function is just a convenient routine that internally calls
    /// `clang::Sema::BuildCXXNew`.
    clang::CXXNewExpr* BuildCXXNewExpr(clang::Sema& semaRef,
                                       clang::QualType qType,
                                       clang::Expr* arraySize,
                                       clang::Expr* initializer,
                                       clang::TypeSourceInfo* TSI = nullptr,
                                       clang::MultiExprArg ArgExprs = {});

    /// Builds a static cast to RValue expression for the expression `E`.
    ///
    /// If type of `E` is `T`. Then this function effectively creates:
    // ```
    // static_cast<T&&>(E)
    // ```
    clang::Expr* BuildStaticCastToRValue(clang::Sema& semaRef, clang::Expr* E);

    /// Returns true if expression `E` is PRValue or XValue.
    bool IsRValue(const clang::Expr* E);

    /// Append statements from `S` to `block`.
    ///
    /// If `S` is a compound statement, then each individual statement is
    /// is appended to `block`.
    /// If `S` is any other statement, then it is appended to `block`.
    /// If `S` is `null`, then nothing happens.
    void AppendIndividualStmts(llvm::SmallVectorImpl<clang::Stmt*>& block,
                               clang::Stmt* S);
    /// Builds a nested member expression that consist of base expression
    /// specified by `base` argument and data members specified in `fields`
    /// argument in the original sequence.
    ///
    /// For example, if `base` represents `b` -- an expression of a record type,
    /// and `fields` is the sequence {'mem1', 'mem2', 'mem3'}, then the function
    /// builds and returns the following expression:
    /// ```
    /// b.mem1.mem2.mem3
    /// ```
    clang::MemberExpr*
    BuildMemberExpr(clang::Sema& semaRef, clang::Scope* S, clang::Expr* base,
                    llvm::ArrayRef<llvm::StringRef> fields);

    /// Returns true if member expression path specified by `fields` is correct;
    /// otherwise returns false.
    ///
    /// For example, if `base` represents `b` -- an expression of a record type,
    /// and `fields` is the sequence {'mem1', 'mem2', 'mem3'}, then the function
    /// returns true if `b.mem1.mem2.mem3` is a valid data member reference
    /// expression, otherwise returns false.
    ///
    /// \note This function returns true if `fields` is an empty sequence.
    bool IsValidMemExprPath(clang::Sema& semaRef, clang::RecordDecl* RD,
                     llvm::ArrayRef<llvm::StringRef> fields);

    /// Perform lookup for data member with name `name`. If lookup finds a
    /// declaration, then return the field declaration; otherwise returns
    /// `nullptr`.
    clang::FieldDecl* LookupDataMember(clang::Sema& semaRef,
                                       clang::RecordDecl* RD,
                                       llvm::StringRef name);

    /// Computes the type of a data member of the record specified by `RD`
    /// and nested fields specified in `fields` argument.
    /// For example, if `RD` represents `std::pair<std::pair<std::complex,
    /// double>, std::pair<double, double>`, and `fields` is the sequence
    /// {'first', 'first'}, then the corresponding data member is
    // of type `std::complex`.
    clang::QualType
    ComputeMemExprPathType(clang::Sema& semaRef, clang::RecordDecl* RD,
                           llvm::ArrayRef<llvm::StringRef> fields);

    /// Instantiate clad::class<TemplateArgs> type
    ///
    /// \param[in] CladClassDecl the decl of the class that is going to be used
    /// in the creation of the type \param[in] TemplateArgs an array of template
    /// arguments \returns The created type clad::class<TemplateArgs>
    clang::QualType
    InstantiateTemplate(clang::Sema& S, clang::TemplateDecl* CladClassDecl,
                        llvm::ArrayRef<clang::QualType> TemplateArgs);
    clang::QualType InstantiateTemplate(clang::Sema& S,
                                        clang::TemplateDecl* CladClassDecl,
                                        clang::TemplateArgumentListInfo& TLI);
    /// Builds the QualType of the derivative to be generated.
    ///
    /// \param[in] moveBaseToParams If true, turns member functions into regular
    /// functions by moving the base to the parameters.
    clang::QualType
    GetDerivativeType(clang::Sema& S, const clang::FunctionDecl* FD,
                      DiffMode mode,
                      llvm::ArrayRef<const clang::ValueDecl*> diffParams,
                      bool moveBaseToParams = false,
                      llvm::ArrayRef<clang::QualType> customParams = {});
    /// Find declaration of clad::class templated type
    ///
    /// \param[in] className name of the class to be found
    /// \returns The declaration of the class with the name ClassName
    clang::TemplateDecl*
    LookupTemplateDeclInCladNamespace(clang::Sema& S,
                                      llvm::StringRef ClassName);

    bool hasNonDifferentiableAttribute(const clang::Decl* D);

    bool hasNonDifferentiableAttribute(const clang::Expr* E);

    /// Collects every DeclRefExpr, MemberExpr, ArraySubscriptExpr in an
    /// assignment operator or a ternary if operator. This is useful to when we
    /// need to decide what needs to be stored on tape in reverse mode.
    void GetInnermostReturnExpr(const clang::Expr* E,
                                llvm::SmallVectorImpl<clang::Expr*>& Exprs);

    void
    getRecordDeclFields(const clang::RecordDecl* RD,
                        llvm::SmallVectorImpl<const clang::FieldDecl*>& fields);

    clang::Expr* getZeroInit(clang::QualType T, clang::Sema& S);

    bool ContainsFunctionCalls(const clang::Stmt* E);

    /// Find namespace clad declaration.
    clang::NamespaceDecl* GetCladNamespace(clang::Sema& S);
    /// Create clad::array<T> type.
    clang::QualType GetCladArrayOfType(clang::Sema& S, clang::QualType T);
    /// Create clad::matrix<T> type.
    clang::QualType GetCladMatrixOfType(clang::Sema& S, clang::QualType T);
    /// Create clad::array_ref<T> type.
    clang::QualType GetCladArrayRefOfType(clang::Sema& S, clang::QualType T);

    clang::QualType GetParameterDerivativeType(clang::Sema& S, DiffMode Mode,
                                               clang::QualType Type);

    void SetSwitchCaseSubStmt(clang::SwitchCase* SC, clang::Stmt* subStmt);

    bool IsLiteral(const clang::Expr* E);
    bool IsZeroOrNullValue(const clang::Expr* E);

    bool IsMemoryFunction(const clang::FunctionDecl* FD);
    bool IsMemoryDeallocationFunction(const clang::FunctionDecl* FD);

    /// Returns true if QT is a non-const reference type.
    bool isNonConstReferenceType(clang::QualType QT);

    bool isCopyable(const clang::CXXRecordDecl* RD);

    bool isLinearConstructor(const clang::CXXConstructorDecl* CD,
                             const clang::ASTContext& C);

    bool IsDifferentiableType(clang::QualType T);

    /// Returns true if FD can be differentiated as a pushforward
    /// And be used in the reverse mode.
    bool canUsePushforwardInRevMode(const clang::FunctionDecl* FD);

    /// We need to replace std::initializer_list with clad::array in the reverse
    /// mode because the former is temporary by design and it's not possible to
    /// create modifiable adjoints.
    clang::QualType replaceStdInitListWithCladArray(clang::Sema& S,
                                                    clang::QualType origTy);
    } // namespace utils
    } // namespace clad

#endif
