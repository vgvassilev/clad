// RUN: %cladclang %s -I%S/../../include -oEarlyReturnRefPair.out 2>&1 \
// RUN:   | %filecheck %s
// RUN: ./EarlyReturnRefPair.out | %filecheck_exec %s

// With an early return the forward sweep runs in a closure. A reference
// returned with its adjoint stays in the closure; the reverse sweep reaches it
// through pointers declared outside, not through a tape.

#include "clad/Differentiator/Differentiator.h"

#include <cstdio>

struct M {
  double m[2];
  double& at(int i) { return m[i]; }
};

double f(double x, double c) {
  if (c == 0)
    return 0;
  M a{{x, 0}};
  a.at(1) = a.at(0) * x;
  return a.at(1);
}

// CHECK: void f_grad(double x, double c, double *_d_x, double *_d_c) {
// CHECK-NOT: clad::tape
// CHECK: double *_t1 = &_dummy0;
// CHECK: clad::forward_sweep([&] {
// CHECK: clad::ValueAndAdjoint<double &, double &> _t0 = a.at_reverse_forw(1, &_d_a, 0);
// CHECK-NEXT: _t1 = &_t0.value;

int main() {
  double dx = 0, dc = 0;
  clad::gradient(f).execute(3, 1, &dx, &dc);
  printf("%.2f\n", dx); // CHECK-EXEC: 6.00
  dx = 0;
  clad::gradient(f).execute(3, 0, &dx, &dc);
  printf("%.2f\n", dx); // CHECK-EXEC: 0.00
}
