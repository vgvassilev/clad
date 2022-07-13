// RUN: %cladclang %s -I%S/../../include -oNonContinuous.out 
// RUN: ./NonContinuous.out | FileCheck -check-prefix=CHECK-EXEC %s

//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

// f(x) = | +x*x, x >= 0
//        | -x*x, x < 0
//
// f'(x)= | 2*x, x >= 0
//        | -2*x, x < 0
int f(int x) {
  if (x < 0)
    return -x*x;
  return x*x;
}
// CHECK: int f_darg0(int x) {
// CHECK-NEXT: int _d_x = 1;
// CHECK-NEXT: if (x < 0)
// CHECK-NEXT:   return -_d_x * x + -x * _d_x;
// CHECK-NEXT: return _d_x * x + x * _d_x;
// CHECK-NEXT: }

// Semantically equivallent to f(x), but implemented differently.
int f1(int x) {
  if (x < 0)
    return -x*x;
  else
    return x*x;
}
// CHECK: int f1_darg0(int x) {
// CHECK-NEXT:  int _d_x = 1;
// CHECK-NEXT:  if (x < 0)
// CHECK-NEXT:    return -_d_x * x + -x * _d_x;
// CHECK-NEXT:  else
// CHECK-NEXT:    return _d_x * x + x * _d_x;
// CHECK-NEXT: }

// g(y) = | 1, y >= 0
//        | 2, y < 0
//
// g'(y)= 0

int g(long y) {
  if (y)
    return 1;
  else
    return 2;
}
// CHECK: int g_darg0(long y) {
// CHECK-NEXT: long _d_y = 1;
// CHECK-NEXT: if (y)
// CHECK-NEXT:   return 0;
// CHECK-NEXT: else
// CHECK-NEXT:   return 0;
// CHECK-NEXT: }
int f_darg0(int x);
int f1_darg0(int x);
int g_darg0(long y);

int main () {
  int x = 4;
  clad::differentiate(f, 0); // expected-no-diagnostics
  printf("Result is = %d\n", f_darg0(1)); // CHECK-EXEC: Result is = 2

  clad::differentiate(f1, 0);
  printf("Result is = %d\n", f1_darg0(1)); // CHECK-EXEC: Result is = 2

  clad::differentiate(g, 0);
  printf("Result is = %d\n", g_darg0(1)); // CHECK-EXEC: Result is = 0

  return 0;
}
