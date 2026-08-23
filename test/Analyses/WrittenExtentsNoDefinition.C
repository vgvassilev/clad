// A function declared but not defined here writes nothing this analysis can
// see -- which is not the same as writing nothing. Reporting `none` would be a
// proof drawn from never having looked, and an extern is free to fill the
// whole buffer it is handed. That clad separately declines to differentiate
// such a call, and warns, says nothing about what the call does to memory.
//
// -verify consumes those warnings, so this checks only the report. Syntax-only
// because there is deliberately no definition to link against.
//
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -fdump-analysis=loop \
// RUN:   -Xclang -verify -fsyntax-only %s -I%S/../../include 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK %s

#include "clad/Differentiator/Differentiator.h"

void declaredOnly(int n, double* out);

// Only parameters it could write through are affected: an int by value cannot
// carry a write back to the caller, so it stays `none`.
// CHECK: written-extent: declaredOnly: n = none
// CHECK-NEXT: written-extent: declaredOnly: out = unknown (the function has no definition here at line [[@LINE-5]])

double f(double a) {
  double o[4] = {0, 0, 0, 0};
  // Differentiated once for the pullback and once for the forward pass, so
  // each diagnostic is seen twice.
  declaredOnly(4, o); // expected-warning 2{{attempted differentiation of function 'declaredOnly' without definition and no suitable overload was found in namespace 'custom_derivatives'}} expected-note 2{{numerical differentiation is not viable for 'declaredOnly'; considering 'declaredOnly' as 0}}
  return o[0] * a;
}

int main() {
  auto g = clad::gradient(f);
  return 0;
}
