// RUN: %cladclang %s -I%S/../../include -oUsefulConditional.out 2>&1 \
// RUN:   | %filecheck %s
// RUN: ./UsefulConditional.out | %filecheck_exec %s

// The useful analysis leaves a variable no output depends on without a
// tangent. A conditional whose branch reads one takes zero for that branch,
// and one whose branches both do has no tangent.

#include "clad/Differentiator/Differentiator.h"

#include <cstdio>
#include <limits>

double f(double a, double b) {
  double const nan = std::numeric_limits<double>::quiet_NaN();
  double const one = 1;
  double t = (b > 0.0) ? one : nan;
  double u = (b > 0.0) ? a / b : nan;
  double v = (b > 0.0) ? nan : a / b;
  return a * b;
}

// CHECK: double f_darg0(double a, double b) {
// CHECK: double t = _t0 ? one : nan0;
// CHECK-NEXT: bool _t1 = (b > 0.);
// CHECK-NEXT: double u = _t1 ? a / b : nan0;
// CHECK-NEXT: bool _t2 = (b > 0.);
// CHECK-NEXT: double v = _t2 ? nan0 : a / b;
// CHECK-NEXT: return _d_a * b + a * _d_b;

int main() {
  auto d = clad::differentiate<clad::opts::enable_ua>(f, "a");
  printf("%.2f\n", d.execute(3, 2)); // CHECK-EXEC: 2.00
}
