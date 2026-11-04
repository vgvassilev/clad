//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang Automatic Differentiator
//----------------------------------------------------------------------------//

#ifndef CLAD_DIFFERENTIATOR_LIVENESSANALYZER_H
#define CLAD_DIFFERENTIATOR_LIVENESSANALYZER_H

#include <set>

namespace clang {
class Stmt;
} // namespace clang

namespace clad {
struct DiffRequest;

/// Adds to \p Unread each statement of the body of \p R's function that is a
/// call whose results nothing reads: it changes only variables of the
/// function passed to it by reference, and no later statement reads them.
/// The reverse sweep puts back what such a call changed before its pullback
/// runs, so the forward sweep need not run it. Records on \p R why a call that
/// nothing reads still has to run.
void findUnreadCalls(const DiffRequest& R,
                     std::set<const clang::Stmt*>& Unread);

} // namespace clad

#endif // CLAD_DIFFERENTIATOR_LIVENESSANALYZER_H
