// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -fdump-analysis=loop %s \
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
// CHECK-NEXT: written-extent: viaCall: out = unknown (a write could not be attributed to a parameter at line [[@LINE-4]])

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
// CHECK-NEXT: written-extent: viaLocal: out = unknown (a write could not be attributed to a parameter at line [[@LINE-7]])

// A member call keeps its object out of the argument list, so the write below
// is invisible to a walk over the arguments alone.
struct Box {
  double v[3];
  void twice() {
    for (int i = 0; i < 3; i++)
      v[i] = v[i] * 2;
  }
};
void viaMemberCall(Box* b) { b->twice(); }
// CHECK: written-extent: viaMemberCall: b = unknown (a write could not be attributed to a parameter at line [[@LINE-1]])

// A non-const reference is as good as a pointer for writing through, and the
// callee's body is no more visible here.
void bump(double& r) { r = r + 1; }
void viaReference(double* out) { bump(out[0]); }
// CHECK: written-extent: viaReference: out = unknown (a write could not be attributed to a parameter at line [[@LINE-1]])

double f(const double* x) {
  double a[3] = {0, 0, 0};
  double b[3] = {0, 0, 0};
  viaCall(3, x, a);
  viaLocal(3, x, b);
  Box box;
  for (int i = 0; i < 3; i++)
    box.v[i] = x[i];
  viaMemberCall(&box);
  viaReference(a);
  return a[0] * a[0] + a[1] * a[1] + a[2] * a[2] + b[0] + b[1] + b[2] +
         box.v[0];
}

int main() {
  auto g = clad::gradient(f, "x");
  double x[3] = {1, 2, 3};
  double dx[3] = {0, 0, 0};
  g.execute(x, dx);
  // a_i = 2 x_i, and viaReference adds one to a_0, so d(a_0^2)/dx_0 is
  // 8 x_0 + 4 and the rest 8 x_i; b_i = 2 x_i contributes 2, and
  // box.v[0] = 2 x_0 contributes 2 more to the first.
  printf("%.2f %.2f %.2f\n", dx[0], dx[1], dx[2]);
  // CHECK-EXEC: 16.00 18.00 26.00
  return 0;
}
