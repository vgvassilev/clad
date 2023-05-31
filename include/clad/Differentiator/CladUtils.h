// This file contains utility functions which do not belong anywhere else

#ifndef CLAD_UTILS_CLADUTILS_H
#define CLAD_UTILS_CLADUTILS_H

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclarationName.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Sema/Sema.h"
#include "clang/AST/Type.h"
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

    /// Finds declaration context associated with the DC1::DC2.
    /// For example, consider DC1 corresponds to the following declaration
    /// context:
    ///
    /// ```
    /// namespace custom_derivatives {}
    /// ```
    ///
    /// and DC2 corresponds to the following declaration context:
    /// ```
    /// namespace A {
    ///   namespace B {}
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
                                        clang::DeclContext* DC2);

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

    /// Returns the outermost declaration context, other than the translation
    /// unit declaration, associated with DC. For example, consider a struct `S`
    /// as follows:
    ///
    /// ```
    /// namespace A {
    ///   namespace B {
    //    struct S {};
    ///   }
    /// }
    /// ```
    ///
    /// In this case, outermost declaration context associated with `S` is of
    /// namespace `A`.
    ///
    /// \param semaRef
    /// \param[in] DC
    clang::DeclContext* GetOutermostDC(clang::Sema& semaRef,
                                       clang::DeclContext* DC);

    /// Creates a `StringLiteral` node to represent string literal
    /// "`str`".
    ///
    ///\param C reference to `ASTContext` object.
    ///\param[in] str string literal to create.
    clang::StringLiteral* CreateStringLiteral(clang::ASTContext& C,
                                              llvm::StringRef str);

    /// Returns true if `QT` is Array or Pointer Type, otherwise returns false.
    bool isArrayOrPointerType(const clang::QualType QT);

    clang::DeclarationNameInfo BuildDeclarationNameInfo(clang::Sema& S,
                                                        llvm::StringRef name);

    /// Returns true if the function has any reference or pointer parameter;
    /// otherwise returns false.
    bool HasAnyReferenceOrPointerArgument(const clang::FunctionDecl* FD);

    /// Returns true if `T` is a reference, pointer or array type.
    ///
    /// \note Please note that this function returns true for array types as
    /// well.
    bool IsReferenceOrPointerType(clang::QualType T);

    /// Returns true if `T1` and `T2` have same cononical type; otherwise
    /// returns false.
    bool SameCanonicalType(clang::QualType T1, clang::QualType T2);

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
                     clang::TypeSourceInfo* TSI = nullptr);

    /// If `T` represents an array or a pointer type then returns the
    /// corresponding array element or the pointee type. If `T` is a reference
    /// type then return the corresponding non-reference type. Otherwise, if `T`
    /// is neither an array nor a pointer type, then simply returns `T`.
    clang::QualType GetValueType(clang::QualType T);

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
                                       clang::TypeSourceInfo* TSI = nullptr);

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

    bool hasNonDifferentiableAttribute(const clang::Decl* D);

    bool hasNonDifferentiableAttribute(const clang::Expr* E);
  } // namespace utils
}

#endif
