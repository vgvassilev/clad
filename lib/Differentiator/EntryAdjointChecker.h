#ifndef CLAD_DIFFERENTIATOR_ENTRYADJOINTCHECKER_H
#define CLAD_DIFFERENTIATOR_ENTRYADJOINTCHECKER_H

#include "llvm/ADT/STLExtras.h" // IWYU pragma: keep -- function_ref on LLVM<14

namespace clang {
class FunctionDecl;
class Sema;
} // namespace clang

namespace clad {

/// How the reverse sweep of a gradient differentiates a call: not at all,
/// through a pullback whose body clad derives, or through a derivative it
/// cannot see into (a custom one, or a pushforward).
enum class CallDerivative { None, Pullback, Opaque };

/// Reports where the gradient of \p FD would read the adjoint of a global
/// pointer or reference before giving it a value.
///
/// Reverse mode gives a global pointer `gp` a global adjoint `_d_gp`, and
/// points it at the adjoint of an object only where the differentiated code
/// assigns `gp`. What `gp` points to when the gradient starts has no adjoint
/// clad could point `_d_gp` at, so `_d_gp` is null there, and the derivative
/// of an access through `gp` dereferences it. A global reference cannot be
/// assigned at all and gets no adjoint. Either way the derivative cannot be
/// built, and this says so at the access instead of emitting a gradient that
/// crashes or silently loses the access' derivative.
///
/// The check follows the value through local copies, member and element
/// paths, and into the pullbacks of the callees \p HowDifferentiated reports.
/// It is conservative: an assignment counts only where it is certain to have
/// run, and a call that may reassign a global pointer leaves its adjoint
/// unknown, as the call's derivative does not update it.
void diagnoseUnknownEntryAdjoints(
    clang::Sema& S, const clang::FunctionDecl* FD,
    llvm::function_ref<CallDerivative(const clang::FunctionDecl*)>
        HowDifferentiated);

} // namespace clad

#endif // CLAD_DIFFERENTIATOR_ENTRYADJOINTCHECKER_H
