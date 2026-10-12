// RUN: not %cladclang %s -I%S/../../include -std=c++20 -fsyntax-only 2>&1 \
// RUN:     | FileCheck %s
// UNSUPPORTED: clang-14, clang-15, clang-16

// clad puts a derivative into a call by rewriting that call, and it is handed
// a declaration only once the compiler has finished with it. A variable the
// language requires to be initialised by a constant expression is therefore
// out of reach: its initialiser is worked out, and diagnosed, before clad
// sees anything. An ordinary variable is fine, because its value is worked
// out again after the rewrite.
//
// clad cannot fix these, so it says what to write instead. See #2188.

#include "clad/Differentiator/Differentiator.h"

constexpr double f(double a, double b) { return a * b; }

// The forms that work. Nothing is said about either.
//CHECK-NOT: warning: clad cannot
constexpr double good() {
  auto d = clad::differentiate(f, "a");
  return d.execute(3., 5.);
}
constexpr double kGood = good();
static_assert(kGood == 5., "d(a*b)/da at (3,5) is b");

auto kOrdinary = clad::differentiate(f, "a");

// And the forms that cannot work.
constexpr double kNamespace = clad::differentiate(f, "a").execute(3., 5.);
//CHECK: warning: clad cannot put the derivative into this call: the compiler works out the initialiser of 'kNamespace' before clad is handed the declaration
//CHECK: note: call clad from a constexpr function, keep the result in an ordinary variable there, and evaluate that function here

constinit double kConstinit = clad::differentiate(f, "a").execute(3., 5.);
//CHECK: warning: clad cannot put the derivative into this call: the compiler works out the initialiser of 'kConstinit' before clad is handed the declaration

void plain() {
  constexpr double kLocal = clad::differentiate(f, "a").execute(3., 5.);
  (void)kLocal;
}
//CHECK: warning: clad cannot put the derivative into this call: the compiler works out the initialiser of 'kLocal' before clad is handed the declaration

// A constexpr local is no better inside a constexpr function: the compiler
// works it out while that function is still being parsed.
constexpr double inConstexpr() {
  constexpr double kL = clad::differentiate(f, "a").execute(3., 5.);
  return kL;
}
//CHECK: warning: clad cannot put the derivative into this call: the compiler works out the initialiser of 'kL' before clad is handed the declaration

// Reported once, on the pattern, whether or not it is ever instantiated.
template <int N>
constexpr double kTmpl = clad::differentiate(f, "a").execute(3., 5.);
//CHECK: warning: clad cannot put the derivative into this call: the compiler works out the initialiser of 'kTmpl' before clad is handed the declaration

int main() {}
