// RUN: %cladclang %s -I%S/../../include -oNonContinuous.out 2>&1 | FileCheck %s
// RUN: ./NonContinuous.out | FileCheck -check-prefix=CHECK-EXEC %s

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
// CHECK: int f_derived_x(int x) {
// CHECK-NEXT: if (x < 0)
// CHECK-NEXT:   return (-1 * x + -x * 1);
// CHECK-NEXT: return (1 * x + x * 1);
// CHECK-NEXT: }

// Semantically equivallent to f(x), but implemented differently.
int f1(int x) {
  if (x < 0)
    return -x*x;
  else
    return x*x;
}
// CHECK: int f1_derived_x(int x) {
// CHECK-NEXT: if (x < 0)
// CHECK-NEXT:   return (-1 * x + -x * 1);
// CHECK-NEXT: else
// CHECK-NEXT:   return (1 * x + x * 1);
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
// CHECK: int g_derived_y(long y) {
// CHECK-NEXT: if (y)
// CHECK-NEXT:   return 0;
// CHECK-NEXT: else
// CHECK-NEXT:   return 0;
// CHECK-NEXT: }

int f_derived_x(int x);
int f1_derived_x(int x);
int g_derived_y(long y);

int main () {
  int x = 4;
  diff(f, 1); // expected-no-diagnostics
  printf("Result is = %d\n", f_derived_x(1)); // CHECK-EXEC: Result is = 2

  diff(f1, 1);
  printf("Result is = %d\n", f1_derived_x(1)); // CHECK-EXEC: Result is = 2

  diff(g, 1);
  printf("Result is = %d\n", g_derived_y(1)); // CHECK-EXEC: Result is = 0

  return 0;
}
