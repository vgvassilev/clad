//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
//
// Where a hand-written derivative would do much less work than the one clad
// generated, and why. A reader of finished code: it runs once every derivative
// of the translation unit exists, looks at them and at the functions they
// came from, and changes nothing.
//------------------------------------------------------------------------------

#ifndef CLAD_DIFFERENTIATOR_PERFADVISOR_H
#define CLAD_DIFFERENTIATOR_PERFADVISOR_H

#include "clad/Differentiator/DiffMode.h"

#include "clang/Basic/SourceLocation.h"

#include "llvm/ADT/ArrayRef.h"

namespace clang {
class FunctionDecl;
class Sema;
} // namespace clang

namespace clad {

/// A derivative clad generated, as the advisor reads it.
struct AdvisedDerivative {
  const clang::FunctionDecl* Derivative = nullptr;
  /// The function differentiated; null for one clad asked for itself.
  const clang::FunctionDecl* Original = nullptr;
  DiffMode Mode = DiffMode::unknown;
};

/// Reports, as remarks, the functions whose generated derivative spends a
/// large share of a gradient's work on what a hand-written derivative would
/// not do: running the forward sweep again in a pullback, or treating an
/// object that is constant except for a few entries as wholly varying.
///
/// The share is estimated over the whole gradient: each derivative is weighted
/// by how often the gradient runs it, from the loops its calls sit in.
void adviseOnPerformance(clang::Sema& S,
                         llvm::ArrayRef<AdvisedDerivative> Derivatives);
} // namespace clad

#endif // CLAD_DIFFERENTIATOR_PERFADVISOR_H
