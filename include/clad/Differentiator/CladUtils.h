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

    bool HasAnyReferenceOrPointerArgument(const clang::FunctionDecl* FD);

    /// Returns true if `T` is a reference, pointer or array type.
    ///
    /// \note Please note that this function returns true for array types as
    /// well.
    bool IsReferenceOrPointerType(clang::QualType T);

    /// Returns true if `T1` and `T2` have same cononical type; otherwise
    /// returns false.
    bool SameCanonicalType(clang::QualType T1, clang::QualType T2);
  } // namespace utils
}

#endif