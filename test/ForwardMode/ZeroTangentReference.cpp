// RUN: %cladclang %s -I%S/../../include -oZeroTangentReference.out 2>&1 \
// RUN:   | %filecheck %s
// RUN: ./ZeroTangentReference.out | %filecheck_exec %s

// The useful analysis leaves a variable no output depends on without a
// tangent, and a call that takes it is handed a zero one. A tangent taken by
// non-const reference needs a variable to bind to: the zero one gets its own.

#include "clad/Differentiator/Differentiator.h"

#include <cstdio>

struct Element {
  explicit Element(double k) : k(k) {}
  double k;
};

void step(const Element& e, double& c) { c = c * e.k; }

double f(double x) {
  Element e(2.0);
  double d = x;
  step(e, d);
  return x;
}

// CHECK: double f_darg0(double x) {
// CHECK: double _t0 = 0.;
// CHECK-NEXT: step_pushforward(e, d, clad::zero_like(e), _t0);

int main() {
  auto df = clad::differentiate<clad::opts::enable_ua>(f, "x");
  printf("%.2f\n", df.execute(3)); // CHECK-EXEC: 1.00
}
