// RUN: not %cladclang %s -I%S/../../include -std=c++20 -fsyntax-only 2>&1 \
// RUN:     | FileCheck %s
// UNSUPPORTED: clang-14, clang-15, clang-16

// Two ways a derivative asked for during compilation does not arrive, and
// what the compiler says about each. Neither used to say anything: execute
// handed back a default-constructed value, which cannot be told from a
// derivative that really is zero.
//
// The first is where the call is. clad builds a derivative early only for a
// call inside a constexpr function, so a namespace-scope initialiser is
// worked out before the derivative exists. See #2188.
//
// The second is typing, and the standard decides it. The gradient is built in
// time, but clad calls it through a pointer whose adjoint parameters are
// void* -- see GradientDerivedFnTraits in FunctionTraits.h -- while the
// generated function takes double*. A cast from void* only became a constant
// expression in C++26, so the same program compiled with -std=c++2c works and
// gives the right gradient. Before that it cannot, however early the
// derivative arrives. See #2190.

#include "clad/Differentiator/Differentiator.h"

constexpr double f(double a, double b) { return a * b; }

constexpr double AtNamespaceScope = clad::differentiate(f, "a").execute(3., 5.);

//CHECK: error: constexpr variable 'AtNamespaceScope' must be initialized by a constant expression
//CHECK: note: non-constexpr function 'NoDerivativeYet' cannot be used in a constant expression

constexpr double reverse_mode() {
  auto g = clad::gradient(f);
  double da = 0, db = 0;
  g.execute(3., 5., &da, &db);
  return db;
}

constexpr double InReverseMode = reverse_mode();

//CHECK: error: constexpr variable 'InReverseMode' must be initialized by a constant expression
//CHECK: note: cast from 'void *' is not allowed in a constant expression

int main() {}
