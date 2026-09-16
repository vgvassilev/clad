//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang Automatic Differentiator
//----------------------------------------------------------------------------//

#ifndef CLAD_DIFFERENTIATOR_LOOPANALYZER_H
#define CLAD_DIFFERENTIATOR_LOOPANALYZER_H

#include "clang/Basic/SourceLocation.h"

#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <unordered_map>

namespace clang {
class Expr;
class ForStmt;
class FunctionDecl;
class VarDecl;
} // namespace clang

namespace clad {
struct DiffRequest;

/// What the loop analysis proved about one `for` loop, in the loop's own
/// expressions rather than anything built from them. Empty when the loop is
/// not counted: an integer stepped by one from a start while it compares
/// below a bound, with no early exit and nothing else moving the index.
///
/// Several parts of clad need this same shape for different reasons: how many
/// times a reverse sweep must run, what range of a buffer the body writes.
/// Deciding it once keeps those answers from drifting apart.
struct LoopFacts {
  /// The variable the loop steps, or null when the loop is not counted.
  const clang::VarDecl* IndVar = nullptr;
  /// What it starts at, and what it is compared against. Both are the loop's
  /// own expressions, not copies.
  const clang::Expr* Init = nullptr;
  const clang::Expr* Bound = nullptr;
  /// Whether the comparison is `<=` rather than `<`, so the bound is the last
  /// value taken rather than the first not taken.
  bool Inclusive = false;
  /// Whether the variable starts at or above zero. Together with the bound
  /// this is what puts a subscript by the variable inside [0, Bound). Needs
  /// the loops around this one -- `i + 1` is non-negative because `i` is an
  /// enclosing index -- so only the written-extent walk fills it in.
  bool InitIsNonNegative = false;
  /// Whether the loop declares its index, so no statement after the loop can
  /// read what it left there.
  bool OwnsIndVar = false;
  /// Whether Init and Bound read in the reverse sweep as they did in the
  /// forward one, which is what makes a trip count worth building from them.
  bool BoundsAreStable = false;
  /// An adjoint this loop sums rather than stores: `Base[Index]` is read on
  /// every iteration at an index the loop never moves, so its adjoint is a
  /// sum over the loop. Accumulated in place, `_d_Base[Index] +=` is a store
  /// to an address that never changes, and that is what keeps the loop from
  /// vectorising.
  struct AdjointReduction {
    const clang::VarDecl* Base = nullptr;
    const clang::Expr* Index = nullptr;
  };
  llvm::SmallVector<AdjointReduction, 2> Reductions;

  /// The reduction whose array \p Base names, or null when this loop sums no
  /// such array. Which expressions name the same array is the analysis's
  /// rule, so it is asked here rather than repeated by every reader.
  [[nodiscard]] const AdjointReduction*
  reductionFor(const clang::Expr* Base) const;

  explicit operator bool() const { return IndVar != nullptr; }
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

  /// Why a write could not be bounded, recorded where the analysis gives up.
  /// A caller that only knows an extent is Unknown can say that it declined;
  /// one that knows why can say what to change. Re-deriving the reason at the
  /// reporting site instead is how the two drift apart.
  enum class Refusal : std::uint8_t {
    /// Not refused -- the extent is proven.
    None,
    /// The subscript is a variable no counted loop steps, so nothing bounds
    /// it: the loop is not counted, counts down, or steps by more than one.
    IndexNotCounted,
    /// The subscript is neither a constant nor a variable.
    IndexNotUnderstood,
    /// Two writes that do not describe one range.
    WritesDisagree,
    /// A counted loop does step the subscript, but a call site cannot use its
    /// bound: it is neither a by-value parameter nor a usable constant.
    BoundNotUsable,
    /// A write through a pointer that could not be attributed to a parameter,
    /// so it may have been through any of them.
    OpaqueWrite,
    /// The function has no body here, so nothing about it could be proven.
    NoDefinition
  };

  Kind K = Kind::None;
  Refusal Why = Refusal::None;
  /// The write, or the part of it, a refusal is about -- for a diagnostic to
  /// point at. Set on every write, since a disagreement between two of them
  /// is only discovered at the second.
  clang::SourceLocation RefusedAt;
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

/// Everything the loop analysis proved about one function, filled in one go
/// and kept on the DiffRequest for every reader.
struct FunctionLoopFacts {
  /// The function these facts were proven about, filled in by the request
  /// that caches them. A request that is copied and re-pointed at another
  /// function must not read the old ones.
  const clang::FunctionDecl* Fn = nullptr;
  /// Each `for` in the body, counted or not.
  std::unordered_map<const clang::ForStmt*, LoopFacts> Loops;
  /// The extent each pointer parameter is written over, in parameter order.
  llvm::SmallVector<WrittenExtent, 8> Extents;
};

/// Runs the loop analysis over \p R's function and fills \p Out: which of its
/// `for` loops are counted, with the facts LoopFacts lists, and the extent each
/// pointer parameter is written over.
///
/// Every check in here is a whitelist of shapes that can be proven by
/// inspection; anything outside it yields an empty LoopFacts or an Unknown
/// extent, so a caller stays conservative as the whitelist grows. Called
/// through DiffRequest::getLoopFacts, which runs it once per function.
void analyzeLoops(const DiffRequest& R, FunctionLoopFacts& Out);

} // namespace clad

#endif // CLAD_DIFFERENTIATOR_LOOPANALYZER_H
