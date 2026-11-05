//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#ifndef CLAD_GENERATED_SOURCE_FILE_H
#define CLAD_GENERATED_SOURCE_FILE_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace clang {
class CompilerInstance;
class Decl;
class FieldDecl;
class FunctionDecl;
class ParmVarDecl;
class Sema;
class ValueDecl;
} // namespace clang

namespace clad {
/// The file -fgenerate-source-file asks for: every derivative clad made for
/// the translation unit, written once, at its end.
class GeneratedSourceFile {
  struct Entry {
    const clang::Decl* D;
    bool DeclarationOnly;
    /// The object a free member derivative takes, as `self`.
    const clang::ParmVarDecl* Self;
    /// The members it may not access: those it reads through clad::member,
    /// and those it cannot.
    llvm::SmallVector<const clang::FieldDecl*, 2> Opened;
    llvm::SmallVector<const clang::ValueDecl*, 2> Unopened;
  };
  llvm::SmallVector<Entry, 32> m_Entries;

public:
  /// Adds a derivative clad generated, of \p Primal when it has one. A custom
  /// derivative is the user's, already in their source, and must not be
  /// added.
  void add(clang::Sema& S, clang::Decl* D, bool DeclarationOnly,
           const clang::FunctionDecl* Primal);
  /// Writes the file to \p Path, reporting through \p CI's diagnostics when
  /// it cannot, and forgets what was added.
  void write(clang::CompilerInstance& CI, llvm::StringRef Path);
};
} // namespace clad

#endif // CLAD_GENERATED_SOURCE_FILE_H
