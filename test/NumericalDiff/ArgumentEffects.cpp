// An argument is evaluated as often as the primal evaluates it, however many
// times numerical differentiation calls the function.
// RUN: %cladnumdiffclang %s -I%S/../../include -oArgumentEffects.out \
// RUN:   -Xclang -verify 2>&1 | %filecheck %s
// RUN: ./ArgumentEffects.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"

#include <cmath>
#include <cstdio>

double sideEffect(double x) {
  double y = x;
  // expected-warning@+2 {{attempted differentiation of function 'tgamma'}}
  // expected-note@+1 {{falling back to numerical differentiation for 'tgamma'}}
  double t = std::tgamma(y++);
  return t * y;
}

// CHECK: double sideEffect_darg0(double x) {
// CHECK: double _t0 = _d_y;
// CHECK-NEXT: double _t1 = y++;
// CHECK: forward_central_difference_tangent<double (*)(double){{.*}}>(std::tgamma, _t0, _t1, 0, 0, _t1)

// The call runs only when x > 10. Evaluated beforehand, y++ would run on
// every path, and in place, as often as the derivative calls tgamma: the call
// keeps its argument, evaluated once, and has no numerical derivative.
double inBranch(double x) {
  double y = x;
  // expected-warning@+1 {{numerical differentiation of 'tgamma' is not supported here: its arguments have side effects, and the call is evaluated only on some paths, or once per iteration; considering its derivative as 0}}
  double t = (x > 10) ? std::tgamma(y++) : 1.0;
  return t * y * y;
}

// CHECK: double inBranch_darg0(double x) {
// CHECK-NOT: = y++;
// CHECK-NOT: forward_central_difference
// CHECK: std::tgamma(y++)

// A call with no visible side effect is not one: the nested call to tgamma is
// differentiated numerically in the branch as on any other path.
double pureInBranch(double x) {
  return (x > 0) ? std::tgamma(std::tgamma(x)) : 0.0;
}

// CHECK: double pureInBranch_darg0(double x) {
// CHECK: forward_central_difference

// A loop condition runs once per iteration, so its side effects stay in it.
double inCondition(double x) {
  double y = x, s = 0;
  int i = 0;
  // expected-warning@+1 {{numerical differentiation of 'tgamma' is not supported here: its arguments have side effects, and the call is evaluated only on some paths, or once per iteration; considering its derivative as 0}}
  while ((s = std::tgamma(y++)) > 0 && i < 3)
    ++i;
  return s * x;
}

int main() {
  // d/dx tgamma(x) * (x + 1) = tgamma'(x) * (x + 1) + tgamma(x); at x = 3
  // that is 1.85 * 4 + 2.
  auto d = clad::differentiate(sideEffect, "x");
  printf("%.2f\n", d.execute(3)); // CHECK-EXEC: 9.38

  // y * y at x = 3, where the branch is not taken.
  auto d2 = clad::differentiate(inBranch, "x");
  printf("%.2f\n", d2.execute(3)); // CHECK-EXEC: 6.00

  // tgamma(tgamma(x))' = tgamma'(tgamma(x)) * tgamma'(x); at x = 3,
  // tgamma(3) = 2 and tgamma'(2) = 0.42, tgamma'(3) = 1.85.
  auto d3 = clad::differentiate(pureInBranch, "x");
  printf("%.2f\n", d3.execute(3)); // CHECK-EXEC: 0.78

  // The loop stops when i reaches 3, with s = tgamma(x + 3) as the primal
  // computes it; s's derivative is taken as 0, leaving s = tgamma(4) at x = 1.
  auto d4 = clad::differentiate(inCondition, "x");
  printf("%.2f\n", d4.execute(1)); // CHECK-EXEC: 6.00
}
