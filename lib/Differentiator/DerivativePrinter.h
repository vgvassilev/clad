#ifndef CLAD_DIFFERENTIATOR_DERIVATIVEPRINTER_H
#define CLAD_DIFFERENTIATOR_DERIVATIVEPRINTER_H

#include "GeneratedCode.h"

#include "clang/Basic/SourceLocation.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"

#include <vector>

namespace clang {
class FunctionDecl;
class Sema;
class Stmt;
} // namespace clang

namespace clad {

/// Prints clad's generated code where its nodes already point.
///
/// A generated node has no text of its own, but it does have a slot in one of
/// GeneratedCode's buffers. Printing the derivative into that same buffer
/// gives the node text at the position it already claims: a diagnostic gets a
/// caret and a source line, and a line note can send a debugger to the same
/// bytes.
///
/// A function is printed once, the first time anything is asked about it, and
/// the answer kept. A compilation that asks nothing prints nothing.
class DerivativePrinter {
public:
  DerivativePrinter(clang::Sema& S, GeneratedCode& Locs)
      : m_Sema(S), m_Locs(Locs) {}

  /// Where \p S appears in the printout of \p FD, or an invalid location if
  /// there is no printout or \p S is not in it.
  clang::SourceLocation locationOf(const clang::FunctionDecl* FD,
                                   const clang::Stmt* S);

  /// The span \p S occupies in the printout, so a diagnostic can underline
  /// the whole expression rather than mark its first character.
  clang::SourceRange rangeOf(const clang::FunctionDecl* FD,
                             const clang::Stmt* S);

  /// The line of the chunk \p S was printed on, or zero if it was not. This is
  /// what a slot standing for \p S has to be presented as.
  unsigned lineOf(const clang::FunctionDecl* FD, const clang::Stmt* S);

  /// The line the printout of \p FD begins on, which is its signature.
  unsigned lineOf(const clang::FunctionDecl* FD);

  /// Where the printout of \p FD begins, so a caller can tell which chunk it
  /// went into.
  clang::SourceLocation startOf(const clang::FunctionDecl* FD);

  /// The printed text of \p FD, for a caller that wants to show it rather than
  /// point into it.
  llvm::StringRef textOf(const clang::FunctionDecl* FD);

private:
  struct Printout {
    GeneratedCode::Placement At;
    unsigned Size = 0;
    llvm::DenseMap<const clang::Stmt*, unsigned> Offsets;
    /// Offset of each line's first character, so an offset can be turned into
    /// a line without rescanning the text for every statement.
    std::vector<unsigned> LineStarts;
  };
  const Printout& print(const clang::FunctionDecl* FD);

  clang::Sema& m_Sema;
  GeneratedCode& m_Locs;
  llvm::DenseMap<const clang::FunctionDecl*, Printout> m_Printed;
};

} // namespace clad

#endif // CLAD_DIFFERENTIATOR_DERIVATIVEPRINTER_H
