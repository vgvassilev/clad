// RUN: %cladclang -std=c++17 -I%S/../../include %s -o %t 2>&1 | %filecheck %s
// RUN: %t | %filecheck_exec %s

// Issue #2534: std::fdim drops partial derivative w.r.t second argument in
// forward mode and reverse mode.

#include "clad/Differentiator/Differentiator.h"
#include <cmath>
#include <cstdio>

double f_dim(double x, double y) {
  return std::fdim(x, y);
}

// CHECK: double f_dim_darg0(double x, double y) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     double _d_y = 0;
// CHECK-NEXT:     {{.*}}ValueAndPushforward<double, double> _t0 = {{.*}}fdim_pushforward(x, y, _d_x, _d_y);
// CHECK-NEXT:     return _t0.pushforward;
// CHECK-NEXT: }

// CHECK: double f_dim_darg1(double x, double y) {
// CHECK-NEXT:     double _d_x = 0;
// CHECK-NEXT:     double _d_y = 1;
// CHECK-NEXT:     {{.*}}ValueAndPushforward<double, double> _t0 = {{.*}}fdim_pushforward(x, y, _d_x, _d_y);
// CHECK-NEXT:     return _t0.pushforward;
// CHECK-NEXT: }

// CHECK: void f_dim_grad(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r_d0 = 1;
// CHECK-NEXT:         double _r0 = 0.;
// CHECK-NEXT:         double _r1 = 0.;
// CHECK-NEXT:         clad::custom_derivatives::std::fdim_pullback(x, y, _r_d0, &_r0, &_r1);
// CHECK-NEXT:         *_d_x += _r0;
// CHECK-NEXT:         *_d_y += _r1;
// CHECK-NEXT:     }
// CHECK-NEXT: }

float f_dimf(float x, float y) {
  return std::fdim(x, y);
}

int main() {
  // Test x > y branch
  auto df_dx = clad::differentiate(f_dim, 0);
  auto df_dy = clad::differentiate(f_dim, 1);
  printf("Forward df/dx (3.0, 1.0) = %.6f\n", df_dx.execute(3.0, 1.0));
  // CHECK-EXEC: Forward df/dx (3.0, 1.0) = 1.000000

  printf("Forward df/dy (3.0, 1.0) = %.6f\n", df_dy.execute(3.0, 1.0));
  // CHECK-EXEC: Forward df/dy (3.0, 1.0) = -1.000000

  double gx = 0.0, gy = 0.0;
  auto grad_dim = clad::gradient(f_dim);
  grad_dim.execute(3.0, 1.0, &gx, &gy);
  printf("Reverse grad (3.0, 1.0) = {%.6f, %.6f}\n", gx, gy);
  // CHECK-EXEC: Reverse grad (3.0, 1.0) = {1.000000, -1.000000}

  // Test x <= y branch (subgradient is 0)
  printf("Forward df/dx (1.0, 3.0) = %.6f\n", df_dx.execute(1.0, 3.0));
  // CHECK-EXEC: Forward df/dx (1.0, 3.0) = 0.000000

  printf("Forward df/dy (1.0, 3.0) = %.6f\n", df_dy.execute(1.0, 3.0));
  // CHECK-EXEC: Forward df/dy (1.0, 3.0) = 0.000000

  gx = gy = 0.0;
  grad_dim.execute(1.0, 3.0, &gx, &gy);
  printf("Reverse grad (1.0, 3.0) = {%.6f, %.6f}\n", gx, gy);
  // CHECK-EXEC: Reverse grad (1.0, 3.0) = {0.000000, 0.000000}

  // Float variant
  float gfx = 0.0f, gfy = 0.0f;
  auto grad_dimf = clad::gradient(f_dimf);
  grad_dimf.execute(5.0f, 2.0f, &gfx, &gfy);
  printf("Reverse gradf (5.0, 2.0) = {%.6f, %.6f}\n", gfx, gfy);
  // CHECK-EXEC: Reverse gradf (5.0, 2.0) = {1.000000, -1.000000}

  return 0;
}
