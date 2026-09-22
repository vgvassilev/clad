//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang Automatic Differentiator
//----------------------------------------------------------------------------//

#ifndef CLAD_DIFFERENTIATOR_LOOPSCOPE_H
#define CLAD_DIFFERENTIATOR_LOOPSCOPE_H

#include "LoopAnalyzer.h"

#include "clad/Differentiator/ReverseModeVisitor.h"

#include "llvm/ADT/SmallVector.h"

namespace clad {

/// The loop whose statements are being differentiated.
///
/// Loops nest, so a scope starts out with the state of the one around it and
/// changes only what its own loop decides. A scope is open for as long as the
/// object lives.
struct ReverseModeVisitor::LoopScope {
  ReverseModeVisitor& m_V;
  /// The scope this one was opened inside, restored when it closes.
  LoopScope* m_Enclosing;

  /// One adjoint this loop sums, and what it is summed in.
  struct Accumulator {
    const LoopFacts::AdjointReduction* Fact;
    clang::VarDecl* Acc;
    clang::Expr* Target; // the `_d_Base[Index]` the sum is added to
  };

  explicit LoopScope(ReverseModeVisitor& V);
  ~LoopScope();
  LoopScope(const LoopScope&) = delete;
  LoopScope& operator=(const LoopScope&) = delete;
  LoopScope(LoopScope&&) = delete;
  LoopScope& operator=(LoopScope&&) = delete;

  /// Whether a store here happens once per iteration, and so goes on a tape
  /// rather than into a single variable. False where the loop recomputes
  /// instead, which is what a checkpointed loop does.
  bool Tapes;
  /// Whether this loop or one around it is checkpointed.
  bool Checkpointed;
  /// The loop whose accumulators an adjoint read here is summed in: this one,
  /// one around it, or none. A loop that sums nothing hides the accumulators
  /// of the loops around it, since a sum kept there would not be its own.
  LoopScope* Sums;
  /// What the analysis proved about this loop, read where Sums names it.
  const LoopFacts* Facts = nullptr;
  llvm::SmallVector<Accumulator, 2> Accumulators;

  /// The accumulator that stands in for \p Target, an adjoint subscript of
  /// \p Base, or null when no loop here sums it.
  clang::Expr* accumulatorFor(const clang::Expr* Base, clang::Expr* Target);
};

} // namespace clad

#endif // CLAD_DIFFERENTIATOR_LOOPSCOPE_H
