#ifndef CLAD_DIFFERENTIATOR_DERIVATIVEPRINTER_H
#define CLAD_DIFFERENTIATOR_DERIVATIVEPRINTER_H

#include "clang/Basic/SourceLocation.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"

namespace clang {
class FunctionDecl;
class Sema;
class Stmt;
} // namespace clang

namespace clad {

/// Gives clad-generated code somewhere to be pointed at.
///
/// A generated node has no source location of its own -- clad builds it, so
/// there is no file it came from -- and the placeholder every visitor reaches
/// for is the start of the main file, which puts a caret on the user's first
/// line. That is not a diagnostic anyone can act on.
///
/// Rendering the derivative into a buffer and registering it with the
/// SourceManager gives the node a real location: one inside text that reads
/// like the code clad produced. Diagnostics then render with a caret and
/// context, exactly as they do for code the user wrote. The same mechanism
/// backs interactive input in cling and clang-repl, and macro expansions in
/// Swift.
///
/// The rendering is built on first use and cached per function: a compilation
/// that reports nothing never pays for one.
class DerivativePrinter {
public:
  explicit DerivativePrinter(clang::Sema& S) : m_Sema(S) {}

  /// Where \p S appears within the rendering of \p FD, or an invalid location
  /// if \p FD has no body or the printer never announced \p S.
  clang::SourceLocation locationOf(const clang::FunctionDecl* FD,
                                   const clang::Stmt* S);

  /// The span \p S occupies in the rendering, so a diagnostic can underline
  /// the whole expression rather than mark its first character. A loop's
  /// condition or increment is several nodes wide and reads as one thing.
  clang::SourceRange rangeOf(const clang::FunctionDecl* FD,
                             const clang::Stmt* S);

  /// The rendered text of \p FD, for a caller that wants to show it rather
  /// than point into it.
  llvm::StringRef textOf(const clang::FunctionDecl* FD);

  /// Records that \p FD exists because of the differentiation at \p Loc.
  ///
  /// A rendering has to be entered from somewhere in the translation unit, and
  /// a diagnostic prints that place above itself. The request is the useful
  /// answer -- it is the line a reader would edit -- so a caller that knows it
  /// should say so before the first diagnostic about \p FD. Without this the
  /// rendering is entered from the start of the main file, which is sortable
  /// but tells a reader nothing.
  void noteRequestedAt(const clang::FunctionDecl* FD,
                       clang::SourceLocation Loc);

private:
  struct Rendering {
    clang::FileID File;
    llvm::DenseMap<const clang::Stmt*, unsigned> Offsets;
  };
  const Rendering& render(const clang::FunctionDecl* FD);

  clang::Sema& m_Sema;
  llvm::DenseMap<const clang::FunctionDecl*, Rendering> m_Rendered;
  llvm::DenseMap<const clang::FunctionDecl*, clang::SourceLocation>
      m_RequestedAt;
};

} // namespace clad

#endif // CLAD_DIFFERENTIATOR_DERIVATIVEPRINTER_H
