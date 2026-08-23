#ifndef CLAD_DIFFERENTIATOR_LOOPANALYZER_H
#define CLAD_DIFFERENTIATOR_LOOPANALYZER_H

#include "clang/AST/Stmt.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include "llvm/ADT/SmallVector.h"
#include <cstdint>

namespace clang {
class Expr;
class ForStmt;
class FunctionDecl;
class VarDecl;
} // namespace clang

namespace clad {

/// A `for` loop that steps one integer variable by one, from an initial
/// value, while it compares below a bound.
///
/// Several parts of clad need this same shape for different reasons: what
/// range of a buffer the body writes, how many times a reverse sweep must
/// run, how to invert the loop. Recognising it once keeps those answers
/// from drifting apart.
struct CountedForLoop {
  /// The variable the loop steps. Null when the loop is not of this shape.
  const clang::VarDecl* IndVar = nullptr;
  /// What it starts at, and what it is compared against. Both are the
  /// loop's own expressions, not copies.
  const clang::Expr* Init = nullptr;
  const clang::Expr* Bound = nullptr;
  /// Whether the comparison is `<=` rather than `<`, so the bound is the
  /// last value taken rather than the first not taken.
  bool Inclusive = false;
  /// Whether the variable starts at or above zero. Together with the bound
  /// this is what puts a subscript by the variable inside [0, Bound).
  ///
  /// Only CountedLoopStack fills this in: a start like `i + 1` is
  /// non-negative because `i` is an enclosing loop's index, so answering it
  /// needs the loops around this one, which recognition alone does not see.
  bool InitIsNonNegative = false;

  explicit operator bool() const { return IndVar != nullptr; }
};

/// Recognises the shape above in \p FS, or returns an empty result.
///
/// This is syntax only. Whether a caller may act on it depends on facts
/// this does not look at -- whether the bound still holds the same value
/// when the caller reads it, whether the body leaves the loop early or
/// moves the induction variable itself -- so each caller adds the
/// conditions its own use needs.
CountedForLoop recogniseCountedForLoop(const clang::ForStmt* FS);

/// The counted loops enclosing whatever a visitor is currently looking at.
///
/// A visitor calls enter() on the way into a `for` statement and leave() on
/// the way out. Both the written-extent analysis and reverse mode need to ask
/// the same question of the result -- which enclosing loop, if any, steps a
/// given variable -- and each was keeping its own stack to answer it.
class CountedLoopStack {
  llvm::SmallVector<CountedForLoop, 4> m_Loops;

public:
  /// Recognises \p FS and pushes it. Returns whether it was pushed, which the
  /// caller must hand back to leave() so the two stay paired.
  bool enter(const clang::ForStmt* FS);
  void leave(bool Entered) {
    if (Entered)
      m_Loops.pop_back();
  }

  /// The innermost enclosing loop that steps \p V, or null if none does.
  const CountedForLoop* steppedBy(const clang::VarDecl* V) const {
    for (const CountedForLoop& L : llvm::reverse(m_Loops))
      if (L.IndVar == V)
        return &L;
    return nullptr;
  }

  bool empty() const { return m_Loops.empty(); }

private:
  /// Whether \p E is provably at or above zero, given the loops already on
  /// the stack -- which is what makes `i + 1` non-negative inside a loop over
  /// `i` that starts at zero.
  bool isNonNegative(const clang::Expr* E) const;
};

/// The range of one pointer parameter that a function writes, described in
/// terms of that function's own parameters so a call site can evaluate it by
/// substituting arguments.
///
/// The description over-approximates: it names a range that contains every
/// write, never one that misses any. A caller may therefore record more than
/// the callee strictly touches, but never less.
struct WrittenExtent {
  enum class Kind : std::uint8_t {
    /// The function does not write through this parameter. Nothing to record.
    None,
    /// Exactly one element, at a constant offset.
    Element,
    /// Elements [0, Bound), where Bound is a parameter or a constant.
    Range,
    /// The function writes through this parameter but the range could not be
    /// bounded. Callers must assume nothing and keep the conservative
    /// protocol.
    Unknown
  };

  Kind K = Kind::None;
  /// For Kind::Element: the constant offset written.
  std::uint64_t Offset = 0;
  /// For Kind::Range: whether Bound names a parameter or is a constant.
  bool BoundIsParam = false;
  /// For Kind::Range with BoundIsParam: index of the bounding parameter.
  unsigned BoundParamIdx = 0;
  /// For Kind::Range without BoundIsParam: the constant bound.
  std::uint64_t BoundConst = 0;

  [[nodiscard]] bool isProven() const { return K != Kind::Unknown; }
};

/// Computes, for each parameter of `FD`, the extent of that parameter the
/// function writes. The result is indexed by parameter position.
///
/// This is deliberately a small whitelist of shapes that can be proven by
/// inspection -- a constant subscript, a dereference, and a subscript by the
/// induction variable of an enclosing counted loop. Everything else, including
/// every write clad cannot attribute to a parameter, yields Kind::Unknown, so
/// a caller that gates on isProven() stays conservative as the whitelist
/// grows.
llvm::SmallVector<WrittenExtent, 8>
computeWrittenExtents(const clang::FunctionDecl* FD);

} // namespace clad

#endif // CLAD_DIFFERENTIATOR_LOOPANALYZER_H
