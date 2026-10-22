// RUN: %cladclang %s -I%S/../../include -oEarlyReturnNested.out 2>&1 | %filecheck %s
// RUN: ./EarlyReturnNested.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr %s -I%S/../../include -oEarlyReturnNested.out
// RUN: ./EarlyReturnNested.out | %filecheck_exec %s

// When the original function returns early, the derivative must not undo
// the statements after that return. Here the returns are nested inside an
// `if`, in an `else if`, in a loop, or a `return;` of a void function.

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

// The adjoint of the last write restored p[0] from a record never taken.
double sq(double* p, double c) {
  if (c > 0) {
    p[0] = p[0] * p[0];
    if (c > 1)
      return 0;
  }
  p[0] = p[0] * 2;
  return 0;
}

// CHECK: void sq_pullback(double *p, double c, double _d_y, double *_d_p, double *_d_c) {
// CHECK:                 _returnedAt0 = 1;
// CHECK:     if (1 < _returnedAt0) {

// x^3 for c > 1, 2 x^3 for 0 < c <= 1.
double f(double x, double c) {
  double p[1] = {x};
  sq(p, c);
  return p[0] * x;
}

// The adjoint of the write ran when the `else if` returned.
double chain(double* p, double c) {
  if (c > 1)
    return 0;
  else if (c < -1)
    return 0;
  p[0] = p[0] * p[0];
  return 0;
}

// x^2, or x^3 for -1 <= c <= 1.
double h(double x, double c) {
  double p[1] = {x};
  chain(p, c);
  return p[0] * x;
}

// The loop popped the restore tape of the multiply on the iteration that
// returned, which never pushed it.
double w(double x, double t) {
  double y = x;
  int i = 0;
  while (i < 4) {
    {
      if (y > t)
        return y * y;
    }
    y = y * x;
    i++;
  }
  return y;
}

// A `return;` of a void function leaves the forward sweep as well.
void acc(double* p, double t) {
  for (int i = 0; i < 3; i++) {
    p[0] = p[0] * p[1];
    if (p[0] > t) {
      if (i > 0)
        return;
    }
    p[1] = p[1] + p[0];
  }
}

// x^2 (x + x^2) + x + x^2 for x^2 <= t < x^2 (x + x^2).
double g(double x, double t) {
  double p[2] = {x, x};
  acc(p, t);
  return p[0] + p[1];
}

// A void function that ends in `return;` has no early return.
void inc(double* p) {
  p[0] = 2 * p[0];
  return;
}

double fi(double x) {
  double o[1] = {x};
  inc(o);
  return o[0] * x;
}

int main() {
  double dx = 0;
  auto df = clad::gradient(f, "x");
  df.execute(1.3, 2.0, &dx);
  printf("%.2f\n", dx); // CHECK-EXEC: 5.07
  dx = 0;
  df.execute(1.3, 0.5, &dx);
  printf("%.2f\n", dx); // CHECK-EXEC: 10.14

  dx = 0;
  auto dh = clad::gradient(h, "x");
  dh.execute(1.3, -2.0, &dx);
  printf("%.2f\n", dx); // CHECK-EXEC: 2.60

  dx = 0;
  auto dw = clad::gradient(w, "x");
  dw.execute(1.3, 2.0, &dx);
  printf("%.2f\n", dx); // CHECK-EXEC: 22.28

  dx = 0;
  auto dg = clad::gradient(g, "x");
  dg.execute(1.3, 2.0, &dx);
  printf("%.2f\n", dx); // CHECK-EXEC: 17.46

  dx = 0;
  clad::gradient(fi).execute(1.3, &dx);
  printf("%.2f\n", dx); // CHECK-EXEC: 5.20
}
