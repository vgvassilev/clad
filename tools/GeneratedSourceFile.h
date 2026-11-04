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
} // namespace clang

namespace clad {
/// The file -fgenerate-source-file asks for: every derivative clad made for
/// the translation unit, written once, at its end.
class GeneratedSourceFile {
  struct Entry {
    const clang::Decl* D;
    bool DeclarationOnly;
  };
  llvm::SmallVector<Entry, 32> m_Entries;

public:
  /// Adds a derivative clad generated. A custom derivative is the user's,
  /// already in their source, and must not be added.
  void add(const clang::Decl* D, bool DeclarationOnly) {
    m_Entries.push_back({D, DeclarationOnly});
  }
  /// Writes the file to \p Path, reporting through \p CI's diagnostics when
  /// it cannot, and forgets what was added.
  void write(clang::CompilerInstance& CI, llvm::StringRef Path);
};
} // namespace clad

#endif // CLAD_GENERATED_SOURCE_FILE_H
