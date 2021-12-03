// This file contains utility functions which do not belong anywhere else

#ifndef CLAD_UTILS_CLADUTILS_H
#define CLAD_UTILS_CLADUTILS_H

#include "clang/AST/Type.h"
#include "llvm/ADT/StringRef.h"
#include <string>
#include <utility>

namespace clang {
  class FunctionDecl;
}

namespace clad {
  namespace utils {
    /// If `FD` is an overloaded operator, returns a name, unique for
    /// each operator, that can be used to create valid C++ identifiers.
    /// Otherwise if `FD` is an ordinary function, returns the name of the
    /// function `FD`.
    std::string ComputeEffectiveFnName(const clang::FunctionDecl* FD);

    // TODO: Make a separate type to store types of Y and X perhaps?
    std::pair<std::string, std::string>
    ComputeYAndXTypeNames(llvm::StringRef cladDerivedTypeName);

    std::string CreateDerivedTypeName(llvm::StringRef YName,
                                      llvm::StringRef XName);

    std::string GetRecordName(clang::QualType qType);                                      
  } // namespace utils
} // namespace clad

#endif