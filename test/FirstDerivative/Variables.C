// RUN: %cladclang %s -lm -I%S/../../include -oVariables.out 2>&1 | FileCheck %s
// RUN: ./Variables.out
//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"
#include <cmath>

double f_x(double x) {
  double t0 = x;
  double t1 = t0;
  return t1;
}

// CHECK: double f_x_darg0(double x) {
// CHECK-NEXT:    double _d_x = 1;
// CHECK-NEXT:    double _d_t0 = _d_x;
// CHECK-NEXT:    double t0 = x;
// CHECK-NEXT:    double _d_t1 = _d_t0;
// CHECK-NEXT:    double t1 = t0;
// CHECK-NEXT:    return _d_t1;
// CHECK-NEXT: }

double f_ops1(double x) {
  double t0 = x;
  double t1 = 2 * x;
  double t2 = 0;
  double t3 = t1 * 2 + t2;
  return t3;
}

// CHECK: double f_ops1_darg0(double x) {
// CHECK-NEXT:    double _d_x = 1;
// CHECK-NEXT:    double _d_t0 = _d_x;
// CHECK-NEXT:    double t0 = x;
// CHECK-NEXT:    double _d_t1 = 0 * x + 2 * _d_x;
// CHECK-NEXT:    double t1 = 2 * x;
// CHECK-NEXT:    double _d_t2 = 0;
// CHECK-NEXT:    double t2 = 0;
// CHECK-NEXT:    double _d_t3 = _d_t1 * 2 + t1 * 0 + _d_t2;
// CHECK-NEXT:    double t3 = t1 * 2 + t2;
// CHECK-NEXT:    return _d_t3;
// CHECK-NEXT: }

double f_ops2(double x) {
  double t0 = x;
  double t1 = 2 * x;
  double t2 = 5;
  double t3 = t1 * x + t2;
  return t3;
}

// CHECK: double f_ops2_darg0(double x) {
// CHECK-NEXT:    double _d_x = 1;
// CHECK-NEXT:    double _d_t0 = _d_x;
// CHECK-NEXT:    double t0 = x;
// CHECK-NEXT:    double _d_t1 = 0 * x + 2 * _d_x;
// CHECK-NEXT:    double t1 = 2 * x;
// CHECK-NEXT:    double _d_t2 = 0;
// CHECK-NEXT:    double t2 = 5;
// CHECK-NEXT:    double _d_t3 = _d_t1 * x + t1 * _d_x + _d_t2;
// CHECK-NEXT:    double t3 = t1 * x + t2;
// CHECK-NEXT:    return _d_t3;
// CHECK-NEXT: }

double f_sin(double x, double y) {
  double xsin = std::sin(x);
  double ysin = std::sin(y);
  auto xt = xsin * xsin;
  auto yt = ysin * ysin;
  return xt + yt;
}

// CHECK: double f_sin_darg0(double x, double y) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     double _d_y = 0;
// CHECK-NEXT:     double _d_xsin = clad::custom_derivatives{{(::std)?}}::sin_pushforward(x, _d_x);
// CHECK-NEXT:     double xsin = std::sin(x);
// CHECK-NEXT:     double _d_ysin = clad::custom_derivatives{{(::std)?}}::sin_pushforward(y, _d_y);
// CHECK-NEXT:     double ysin = std::sin(y);
// CHECK-NEXT:     double _d_xt = _d_xsin * xsin + xsin * _d_xsin;
// CHECK-NEXT:     double xt = xsin * xsin;
// CHECK-NEXT:     double _d_yt = _d_ysin * ysin + ysin * _d_ysin;
// CHECK-NEXT:     double yt = ysin * ysin;
// CHECK-NEXT:     return _d_xt + _d_yt;
// CHECK-NEXT: }


int main() {
  clad::differentiate(f_x, 0);
  clad::differentiate(f_ops1, 0);
  clad::differentiate(f_ops2, 0);
  clad::differentiate(f_sin, 0);
}


