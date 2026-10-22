// RUN: %cladclang %s -I%S/../../include -oMemberEarlyReturn.out 2>&1 | %filecheck %s
// RUN: ./MemberEarlyReturn.out | %filecheck_exec %s

// The forward sweep of a function with an early return runs in a closure.
// In a member function it reads `this`, which the closure did not capture:
// generating code for it crashed clang.

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

struct S {
  double r;
  double f(double x) const {
    if (r == 0)
      return x;
    return x * x * r;
  }
};

// CHECK: void f_grad(double x, S *_d_this, double *_d_x) const {
// CHECK:     clad::forward_sweep([&] {
// CHECK:             _cond0 = this->r == 0;

int main() {
  S s{2}, ds{0};
  double dx = 0;
  auto g = clad::gradient(&S::f);
  g.execute(s, 1.5, &ds, &dx);
  printf("%.2f %.2f\n", dx, ds.r); // CHECK-EXEC: 6.00 2.25
}
