// RUN: %cladclang %s -I%S/../../include -oNonActiveCallBase.out 2>&1 | %filecheck %s
// RUN: ./NonActiveCallBase.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr %s -I%S/../../include -oNonActiveCallBase.out
// RUN: ./NonActiveCallBase.out | %filecheck_exec %s

// Regression test for issue #1960: a Hessian routed through the pushforward of
// a member call whose base object has no tangent. The forward pass must call
// the pushforward with a zero placeholder for `_d_this` instead of yielding a
// zero derivative, and the reverse pass over that pushforward must keep the
// adjoint-of-`this` slot of the pullback filled even though the base has no
// adjoint of its own.

#include "clad/Differentiator/Differentiator.h"

#include <cstdio>

struct Functor {
  double operator()(double x) const { return x * x; }
};

// gFunc is only read, never differentiated with respect to.
Functor gFunc;

double helper(double x) { return gFunc(x); }
double func(double *p) { return helper(p[0]) * p[1]; }

// CHECK: void helper_pushforward_pullback(double x, double _d_x, clad::ValueAndPushforward<double, double> _d_y, double *_d_x0, double *_d_d_x) {
// CHECK: Functor _r0 = {};
// CHECK: gFunc.operator_call_pushforward_pullback(x, &_d_base, _d_x, _d_t0, &_r0, &_r1, &_d_d_base, &_r2);

// The same shape reached through a cast address, as JIT-generated wrappers
// produce when referring to an object living in the host process.
double helperCast(double x) {
  return reinterpret_cast<Functor const *>(&gFunc)->operator()(x);
}
double funcCast(double *p) { return helperCast(p[0]) * p[1]; }

int main() {
  double p[2] = {3., 5.};

  double h[4] = {0., 0., 0., 0.};
  clad::hessian(func, "p[0:1]").execute(p, h);
  printf("%.2f %.2f %.2f %.2f\n", h[0], h[1], h[2], h[3]);
  // CHECK-EXEC: 10.00 6.00 6.00 0.00

  double hCast[4] = {0., 0., 0., 0.};
  clad::hessian(funcCast, "p[0:1]").execute(p, hCast);
  printf("%.2f %.2f %.2f %.2f\n", hCast[0], hCast[1], hCast[2], hCast[3]);
  // CHECK-EXEC: 10.00 6.00 6.00 0.00
}
