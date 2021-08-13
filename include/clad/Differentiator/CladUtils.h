// This file contains utility functions which do not belong anywhere else

#ifndef CLAD_UTILS_CLADUTILS_H
#define CLAD_UTILS_CLADUTILS_H

#include <llvm/ADT/StringRef.h>

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

    /// Creates a `StringLiteral` node to represent string literal
    /// "`str`".
    ///
    ///\param C reference to `ASTContext` object.
    ///\param[in] str string literal to create.
    clang::StringLiteral* CreateStringLiteral(clang::ASTContext& C,
                                              llvm::StringRef str);
  }
}

#endif