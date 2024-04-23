// RUN: %cladclang %s -I%S/../../include -oNonContinuous.out 2>&1 | FileCheck %s
// RUN: ./NonContinuous.out | FileCheck -check-prefix=CHECK-EXEC %s

//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

// f(x) = | +x*x, x >= 0
//        | -x*x, x < 0
//
// f'(x)= | 2*x, x >= 0
//        | -2*x, x < 0
float f(float x) {
  if (x < 0)
    return -x*x;
  return x*x;
}
// CHECK: float f_darg0(float x) {
// CHECK-NEXT: float _d_x = 1;
// CHECK-NEXT: if (x < 0)
// CHECK-NEXT:   return -_d_x * x + -x * _d_x;
// CHECK-NEXT: return _d_x * x + x * _d_x;
// CHECK-NEXT: }

// Semantically equivallent to f(x), but implemented differently.
float f1(float x) {
  if (x < 0)
    return -x*x;
  else
    return x*x;
}
// CHECK: float f1_darg0(float x) {
// CHECK-NEXT:  float _d_x = 1;
// CHECK-NEXT:  if (x < 0)
// CHECK-NEXT:    return -_d_x * x + -x * _d_x;
// CHECK-NEXT:  else
// CHECK-NEXT:    return _d_x * x + x * _d_x;
// CHECK-NEXT: }

// g(y) = | 1, y >= 0
//        | 2, y < 0
//
// g'(y)= 0

float g(double y) {
  if (y)
    return 1;
  else
    return 2;
}
// CHECK: float g_darg0(double y) {
// CHECK-NEXT: double _d_y = 1;
// CHECK-NEXT: if (y)
// CHECK-NEXT:   return 0;
// CHECK-NEXT: else
// CHECK-NEXT:   return 0;
// CHECK-NEXT: }
float f_darg0(float x);
float f1_darg0(float x);
float g_darg0(double y);

int main () {
  clad::differentiate(f, 0); // expected-no-diagnostics
  printf("Result is = %.2f\n", f_darg0(1)); // CHECK-EXEC: Result is = 2

  clad::differentiate(f1, 0);
  printf("Result is = %.2f\n", f1_darg0(1)); // CHECK-EXEC: Result is = 2

  clad::differentiate(g, 0);
  printf("Result is = %.2f\n", g_darg0(1)); // CHECK-EXEC: Result is = 0

  return 0;
}
