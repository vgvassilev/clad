// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -fdump-written-extents %s \
// RUN:   -I%S/../../include -oWrittenExtentsOpaque.out 2>&1 | %filecheck %s
// RUN: ./WrittenExtentsOpaque.out | %filecheck_exec %s

// A function that writes a parameter only through a call it makes must not be
// reported as leaving that parameter untouched: a caller gating on the extent
// would then record nothing and the gradient would be silently wrong. The
// analysis has to distinguish "provably not written" from "no write seen".
//
// The distinction is which storage the argument designates. Handing a callee a
// pointer into this function's own locals says nothing about its parameters;
// handing it a parameter does.

#include "clad/Differentiator/Differentiator.h"

#include <cstdio>

void inner(int n, double* out) {
  for (int i = 0; i < n; i++)
    out[i] = out[i] * 2;
}
// CHECK: written-extent: inner: n = none
// CHECK-NEXT: written-extent: inner: out = [0, n)

// `out` is written only inside inner, so its extent is not visible here.
void viaCall(int n, const double* x, double* out) {
  for (int i = 0; i < n; i++)
    out[i] = x[i];
  inner(n, out);
}
// CHECK: written-extent: viaCall: n = none
// CHECK-NEXT: written-extent: viaCall: x = none
// CHECK-NEXT: written-extent: viaCall: out = unknown

// The same call shape, but the buffer handed over belongs to this function, so
// nothing the callee does to it can reach `out`. `out` is nonetheless reported
// unknown: designatesLocallyOwnedStorage does not see through a local array
// decaying to a pointer, only through accessors and subscripts. Conservative
// in the safe direction -- a caller keeps the tracker where it need not have.
// Widening that helper would turn this line into `[0, n)`.
void viaLocal(int n, const double* x, double* out) {
  double scratch[8];
  for (int i = 0; i < n; i++)
    scratch[i] = x[i];
  inner(n, scratch);
  for (int i = 0; i < n; i++)
    out[i] = scratch[i];
}
// CHECK: written-extent: viaLocal: n = none
// CHECK-NEXT: written-extent: viaLocal: x = none
// CHECK-NEXT: written-extent: viaLocal: out = unknown

double f(const double* x) {
  double a[3] = {0, 0, 0};
  double b[3] = {0, 0, 0};
  viaCall(3, x, a);
  viaLocal(3, x, b);
  return a[0] * a[0] + a[1] * a[1] + a[2] * a[2] + b[0] + b[1] + b[2];
}

int main() {
  auto g = clad::gradient(f, "x");
  double x[3] = {1, 2, 3};
  double dx[3] = {0, 0, 0};
  g.execute(x, dx);
  // a_i = 2 x_i so d(a_i^2)/dx_i = 8 x_i; b_i = 2 x_i contributes 2.
  printf("%.2f %.2f %.2f\n", dx[0], dx[1], dx[2]);
  // CHECK-EXEC: 10.00 18.00 26.00
  return 0;
}
