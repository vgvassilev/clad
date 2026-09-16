// RUN: %cladclang -std=c++20 %s -I%S/../../include -fsyntax-only -Xclang -verify

#include "clad/Differentiator/Differentiator.h"

// Clad diagnoses a statement kind it cannot differentiate and then clones it
// into the derivative unchanged, which is what lets the compile finish. The
// cloning is what used to crash: StmtClone had no case for these kinds, its
// fallback returned null once NDEBUG removed the assert, and ReferencesUpdater
// walked the null. Reaching the warning below at all is the regression test.

// Diagnosed once per derivation: the forward mode below and the reverse one.
// expected-warning@+1 2 {{statement kind 'BinaryConditionalOperator' is not supported}}
double elvis(double x) { double y = x ?: 1.0; return y; }

double likely_attr(double x) {
  double r = x;
  // expected-warning@+1 {{statement kind 'AttributedStmt' is not supported}}
  if (r > 0) [[likely]] { r *= 2; }
  return r;
}

struct Base {
  double v;
  Base(double a) : v(a) {}
};
struct Derived : Base {
  // Only the reverse mode reaches the inherited constructor, and it derives
  // the constructor for both sweeps, so the diagnostic lands here twice.
  // expected-warning@+1 2 {{statement kind 'CXXInheritedCtorInitExpr' is not supported}}
  using Base::Base;
};

double inherited(double x) { Derived d(x); return d.v; }

int main() {
  clad::differentiate(elvis, "x");
  // Reverse mode clones the body twice, once per sweep, which is what the
  // opaque-value sharing in the BinaryConditionalOperator clone is for: two
  // derivations of one `?:` must not end up sharing a node. DerivativeBuilder
  // checks for that even when the derivation only emitted a diagnostic.
  clad::gradient(elvis, "x");
  clad::differentiate(likely_attr, "x");
  clad::gradient(inherited, "x");
}
