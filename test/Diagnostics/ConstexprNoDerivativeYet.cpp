// RUN: not %cladclang %s -I%S/../../include -std=c++20 -fsyntax-only 2>&1 \
// RUN:     | FileCheck %s
// UNSUPPORTED: clang-14, clang-15, clang-16

// A derivative asked for during compilation that does not arrive, and what
// the compiler says about it. It used to say nothing: execute handed back a
// default-constructed value, which cannot be told from a derivative that
// really is zero.
//
// A namespace-scope initialiser is worked out before clad has put the
// derivative into the call, so there is nothing to call. See #2188.

#include "clad/Differentiator/Differentiator.h"

constexpr double f(double a, double b) { return a * b; }

constexpr double AtNamespaceScope =
    clad::differentiate<clad::immediate_mode>(f, "a").execute(3., 5.);

//CHECK: error: constexpr variable 'AtNamespaceScope' must be initialized by a constant expression
//CHECK: note: non-constexpr function 'NoDerivativeYet' cannot be used in a constant expression

int main() {}
