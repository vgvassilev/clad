// RUN: %cladclang %s -I%S/../../include -oConditionalOperatorCalls.out 2>&1 \
// RUN:   | %filecheck %s
// RUN: ./ConditionalOperatorCalls.out | %filecheck_exec %s

// The pushforward of a call in an arm of `?:` runs only when its arm is
// taken: run either way, the recursion that `?:` stops never stops.

#include "clad/Differentiator/Differentiator.h"

#include <cstdio>

double power(double a, int n) { return n ? power(a * a, n - 1) : a; }
double fourth(double x) { return power(x, 2); }

// CHECK: clad::ValueAndPushforward<double, double> power_pushforward(double a, int n, double _d_a, int _d_n) {
// CHECK: if (n) {
// CHECK-NEXT: clad::ValueAndPushforward<double, double> _t0 = power_pushforward(a * a, n - 1, _d_a * a + a * _d_a, _d_n - 0);

// An arm that is a reference is not lowered: both pushforwards still run.
double& at(double* p, int i) { return p[i]; }
double pick(double x) {
  double a[2] = {x, 2 * x};
  return (x > 0 ? at(a, 0) : at(a, 1)) * x;
}

// CHECK: double pick_darg0(double x) {
// CHECK: clad::ValueAndPushforward<double &, double &> _t0 = at_pushforward(a, 0, _d_a, 0);
// CHECK-NEXT: clad::ValueAndPushforward<double &, double &> _t1 = at_pushforward(a, 1, _d_a, 0);
// CHECK-NEXT: bool _t2 = x > 0;
// CHECK-NEXT: double &_t3 = (_t2 ? _t0.value : _t1.value);
// CHECK-NEXT: return (_t2 ? _t0.pushforward : _t1.pushforward) * x + _t3 * _d_x;

int main() {
  printf("%.2f\n", clad::differentiate(fourth, "x").execute(1.5)); // CHECK-EXEC: 13.50
  auto d = clad::differentiate(pick, "x");
  printf("%.2f %.2f\n", d.execute(1.5), d.execute(-1.5)); // CHECK-EXEC: 3.00 -6.00
}
