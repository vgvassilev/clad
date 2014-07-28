// RUN: %cladclang %s -I%S/../../include -fsyntax-only 2>&1 | FileCheck %s
// XFAIL:*
#include "clad/Differentiator/Differentiator.h"

// f(x) = | +x*x, x >= 0
//        | -x*x, x < 0
//
// f'(x)= | 2*x, x >= 0
//        | -2*x, x < 0

int f(int x) {
  int result = 0;
  if (x < 0)
    result = -x*x;
  else
    result = x*x;
  return result;
}
// CHECK: int f_dx(int x) {
// CHECK-NEXT: int result = 0;
// CHECK-NEXT: if (x < 0)
// CHECK-NEXT:   result = -2 * x;
// CHECK-NEXT: else
// CHECK-NEXT:   result = 2 * x;
// CHECK-NEXT: return result;
// CHECK-NEXT: }

int main () {
  clad::differentiate(f, 1);
  return 0;
}
