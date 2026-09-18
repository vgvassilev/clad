// RUN: %cladclang -fsyntax-only -Xclang -verify -std=c++17 -I%S/../../include %s

// Regression test for https://github.com/vgvassilev/clad/issues/2088.
//
// The GNU binary conditional operator `x ?: y` has no rule in clad, so
// visiting it reports the statement as unsupported. The VisitStmt fallback
// used to clone the node anyway; StmtClone has no case for it, so the clone
// came out malformed and ReferencesUpdater crashed walking it. Diagnosed and
// dropped, the compile finishes instead.

#include "clad/Differentiator/Differentiator.h"

// Forward mode: the unsupported initializer is dropped and the rest of the
// function still differentiates.
double fwd(double x) {
  double y = x ?: 1.0; // expected-warning {{statement kind 'BinaryConditionalOperator' is not supported}} // expected-warning {{statement kind 'BinaryConditionalOperator' is not supported}}
  return y;
}

// Reverse mode with the unsupported node as the returned value, the form from
// the issue.
double rev(double x) {
  return x ?: 1.0; // expected-warning {{statement kind 'BinaryConditionalOperator' is not supported}}
}

int main() {
  clad::gradient(fwd, "x");
  clad::differentiate(fwd, "x");
  clad::gradient(rev, "x");
}
