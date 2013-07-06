// RUN: %autodiff %s -I%S/../../include -fsyntax-only 2>&1 | FileCheck %s
// XFAIL:*
#include "autodiff/Differentiator/Differentiator.h"

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
// CHECK: int f_derived(int x) {
// CHECK-NEXT: if (x < 0)
// CHECK-NEXT:   return -2 * x;
// CHECK-NEXT: return 2 * x;
// CHECK-NEXT: }

// Semantically equivallent to f(x), but implemented differently.
int f1(int x) {
  if (x < 0)
    return -x*x;
  else
    return x*x;
}
// CHECK: int f1_derived(int x) {
// CHECK-NEXT: if (x < 0)
// CHECK-NEXT:   return -2 * x;
// CHECK-NEXT: else
// CHECK-NEXT:   return 2 * x;
// CHECK-NEXT: }

int f2(int x) {
  int result = 0;
  if (x < 0)
    result = -x*x;
  else
    result = x*x;
  return result;
}
// CHECK: int f2_derived(int x) {
// CHECK-NEXT: int result = 0;
// CHECK-NEXT: if (x < 0)
// CHECK-NEXT:   result = -2 * x;
// CHECK-NEXT: else
// CHECK-NEXT:   result = 2 * x;
// CHECK-NEXT: return result;
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
// CHECK: int g_derived(long x) {
// CHECK-NEXT: if (y)
// CHECK-NEXT:   return 0;
// CHECK-NEXT: else
// CHECK-NEXT:   return 0;
// CHECK-NEXT: }

// Or even better:
// CHECK: int g_derived(long x) {
// CHECK-NEXT: return 0;
// CHECK-NEXT: }

int main () {
  int x = 4;
  diff(f, x);
  diff(f1, x);
  diff(f2, x);
  diff(g, x);
  return 0;
}
