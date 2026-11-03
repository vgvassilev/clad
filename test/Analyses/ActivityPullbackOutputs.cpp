// RUN: %cladclang %s -I%S/../../include -o %t 2>&1 | %filecheck %s
// RUN: %t | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"

extern "C" int printf(const char*, ...);

double multiply(const double& x, const double& y) { return x * y; }

double left(double x) {
  double fixed = 3;
  return multiply(x, fixed);
}

double right(double x) {
  double fixed = 5;
  return multiply(fixed, x);
}

// Both requests used to emit the same name and type, with one double* output.
// The result then depended on which body was emitted first.
// CHECK-LABEL: void multiply_pullback(const double &x, const double &y, double _d_y0, double *_d_x, clad::no_adjoint)
// CHECK-LABEL: void left_grad(double x, double *_d_x)
// CHECK: multiply_pullback(x, fixed, 1, _d_x, clad::no_adjoint());
// CHECK-LABEL: void multiply_pullback(const double &x, const double &y, double _d_y0, clad::no_adjoint, double *_d_y)
// CHECK-LABEL: void right_grad(double x, double *_d_x)
// CHECK: multiply_pullback(fixed, x, 1, clad::no_adjoint(), _d_x);

int main() {
  auto dl = clad::gradient<clad::opts::enable_va>(left);
  auto dr = clad::gradient<clad::opts::enable_va>(right);
  double dx_left = 0, dx_right = 0;
  dl.execute(2, &dx_left);
  dr.execute(2, &dx_right);
  printf("left: %.0f, right: %.0f\n", dx_left, dx_right);
  // CHECK-EXEC: left: 3, right: 5
}
