// RUN: %cladclang %s -I%S/../../include -fsyntax-only 2>&1 | FileCheck %s
// XFAIL:*

//CHECK-NOT: {{.*error|warning|note:.*}}
#include "clad/Differentiator/Differentiator.h"

// f(x) = | +x*x, x >= 0
//        | -x*x, x < 0
//
// f'(x)= | 2*x, x >= 0
//        | -2*x, x < 0

double f(double x) {
  double result = 0.;
  if (x < 0)
    result = -x*x;
  else
    result = x*x;
  return result;
}

// CHECK: double f_darg0(double x) {
// CHECK-NEXT: double result = 0.;
// CHECK-NEXT: if (x < 0)
// CHECK-NEXT:   result = (-1. * x + -x * 1.);
// CHECK-NEXT: else
// CHECK-NEXT:   result = (1. * x + x * 1.);
// CHECK-NEXT: return result; // Now returns 0.
// CHECK-NEXT: }

int main () {
  clad::differentiate(f, 0);
  return 0;
}
