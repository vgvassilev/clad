// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -fdump-written-extents %s \
// RUN:   -I%S/../../include -oWrittenExtents.out 2>&1 | %filecheck %s
// RUN: ./WrittenExtents.out

// What a callee writes through each pointer parameter, proven from its body
// and expressed in its own parameters so a call site can evaluate it. The
// whitelist is deliberately narrow; the useful half of the report is what
// comes back `unknown`, because that is what a caller must stay conservative
// about. Widening the whitelist may turn an `unknown` below into a range, but
// must never turn a range into a different range.

#include "clad/Differentiator/Differentiator.h"

// The canonical shape: a counted loop over a parameter bound.
void subtract(int d, const double* x, const double* y, double* out) {
  for (int i = 0; i < d; i++)
    out[i] = x[i] - y[i];
}
// CHECK: written-extent: subtract: d = none
// CHECK-NEXT: written-extent: subtract: x = none
// CHECK-NEXT: written-extent: subtract: y = none
// CHECK-NEXT: written-extent: subtract: out = [0, d)

// Two loops write `out`, the second from `i + 1` rather than zero. Both stay
// inside [0, d), so the two records agree and the parameter is still proven.
void qtimesx(int d, const double* Qd, const double* x, double* out) {
  for (int i = 0; i < d; i++)
    out[i] = Qd[i] * x[i];
  for (int i = 0; i < d; i++)
    for (int j = i + 1; j < d; j++)
      out[j] = out[j] + x[i];
}
// CHECK: written-extent: qtimesx: d = none
// CHECK-NEXT: written-extent: qtimesx: Qd = none
// CHECK-NEXT: written-extent: qtimesx: x = none
// CHECK-NEXT: written-extent: qtimesx: out = [0, d)

// Two different constant offsets do not describe one range, so they widen to
// unknown rather than to a guessed hull.
void constIdx(double* v) {
  v[0] = 1.0;
  v[2] = 3.0;
}
// CHECK: written-extent: constIdx: v = unknown

// A constant bound is as usable as a parameter one.
void fixedLoop(double* v) {
  for (int i = 0; i < 5; ++i)
    v[i] = v[i] * 2;
}
// CHECK: written-extent: fixedLoop: v = [0, 5)

// The index is not an induction variable, so nothing bounds the write.
void dataDependent(int n, const double* x, double* out) {
  for (int i = 0; i < n; i++)
    out[(int)x[i]] = x[i];
}
// CHECK: written-extent: dataDependent: n = none
// CHECK-NEXT: written-extent: dataDependent: x = none
// CHECK-NEXT: written-extent: dataDependent: out = unknown

// Only `for` is recognised; an equivalent while loop is not.
void whileLoop(int n, double* out) {
  int i = 0;
  while (i < n) {
    out[i] = out[i] * 2;
    i++;
  }
}
// CHECK: written-extent: whileLoop: n = none
// CHECK-NEXT: written-extent: whileLoop: out = unknown

// A scalar written through a dereference is a single element.
void scalarOut(double a, double* err) { *err = a * a; }
// CHECK: written-extent: scalarOut: a = none
// CHECK-NEXT: written-extent: scalarOut: err = [0, 1)

double f(double a) {
  double x[4] = {a, a, a, a};
  double y[4] = {a, a, a, a};
  double o[4] = {0, 0, 0, 0};
  double o2[4] = {0, 0, 0, 0};
  subtract(4, x, y, o);
  qtimesx(4, x, y, o2);
  constIdx(o);
  fixedLoop(o);
  dataDependent(4, x, o);
  whileLoop(4, o);
  double e = 0;
  scalarOut(a, &e);
  return o[0] + o2[0] + e;
}

int main() {
  auto g = clad::gradient(f);
  double da = 0;
  g.execute(1.0, &da);
  return 0;
}
