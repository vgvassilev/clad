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
// CHECK-NEXT:    double t0 = x;
// CHECK-NEXT:    double t1 = t0;
// CHECK-NEXT:    return 1.;
// CHECK-NEXT: }

double f_ops1(double x) {
  double t0 = x;
  double t1 = 2 * x;
  double t2 = 0;
  double t3 = t1 * 2 + t2;
  return t3;
}

// CHECK: double f_ops1_darg0(double x) {
// CHECK-NEXT:    double t0 = x;
// CHECK-NEXT:    double t1 = 2 * x;
// CHECK-NEXT:    double t2 = 0;
// CHECK-NEXT:    double t3 = t1 * 2 + t2;
// CHECK-NEXT:    return ((0 * x + 2 * 1.) * 2 + t1 * 0) + (0);
// CHECK-NEXT: }

double f_ops2(double x) {
  double t0 = x;
  double t1 = 2 * x;
  double t2 = 5;
  double t3 = t1 * x + t2;
  return t3;
}

// CHECK: double f_ops2_darg0(double x) {
// CHECK-NEXT:    double t0 = x;
// CHECK-NEXT:    double t1 = 2 * x;
// CHECK-NEXT:    double t2 = 5;
// CHECK-NEXT:    double t3 = t1 * x + t2;
// CHECK-NEXT:    return ((0 * x + 2 * 1.) * x + t1 * 1.) + (0);
// CHECK-NEXT: }

double f_sin(double x, double y) {
  double xsin = std::sin(x);
  double ysin = std::sin(y);
  auto xt = xsin * xsin;
  auto yt = ysin * ysin;
  return xt + yt;
}

// CHECK: double f_sin_darg0(double x, double y) {
// CHECK-NEXT:     double xsin = sin(x);
// CHECK-NEXT:     double ysin = sin(y);
// CHECK-NEXT:     auto xt = xsin * xsin;
// CHECK-NEXT:     auto yt = ysin * ysin;
// CHECK-NEXT:     return (sin_darg0(x) * (1.) * xsin + xsin * sin_darg0(x) * (1.)) + ((sin_darg0(y) * (0.) * ysin + ysin * sin_darg0(y) * (0.)));
// CHECK-NEXT: }


int main() {
  clad::differentiate(f_x, 0);
  clad::differentiate(f_ops1, 0);
  clad::differentiate(f_ops2, 0);
  clad::differentiate(f_sin, 0);
}


