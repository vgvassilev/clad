//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang Automatic Differentiator
//----------------------------------------------------------------------------//

#ifndef CLAD_DIFFERENTIATOR_REDUCTIONSCOPE_H
#define CLAD_DIFFERENTIATOR_REDUCTIONSCOPE_H

#include "LoopAnalyzer.h"

#include "clad/Differentiator/ReverseModeVisitor.h"

#include "llvm/ADT/SmallVector.h"

namespace clad {

/// The accumulators of one loop being differentiated.
///
/// The loop analysis says which adjoints are sums over the loop. This holds the
/// variable each sum is kept in while the body is visited, and the adjoint it
/// is added to once the loop is done.
struct ReverseModeVisitor::ReductionScope {
  struct Accumulator {
    const LoopFacts::AdjointReduction* Fact;
    clang::VarDecl* Acc;
    clang::Expr* Target; // the `_d_Base[Index]` the sum is added to
  };
  const LoopFacts& Facts;
  llvm::SmallVector<Accumulator, 2> Accumulators;

  /// The accumulator that stands in for \p Target, an adjoint subscript of
  /// \p Base, or null when this loop does not sum it.
  clang::Expr* accumulatorFor(ReverseModeVisitor& V, const clang::Expr* Base,
                              clang::Expr* Target);
};

} // namespace clad

#endif // CLAD_DIFFERENTIATOR_REDUCTIONSCOPE_H
