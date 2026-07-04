// RUN: %cladclang -std=c++17 -I%S/../../include %s -o %t 2>&1 | %filecheck %s
// RUN: %t | %filecheck_exec %s
// XFAIL: valgrind

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

double inline_divisor_2(double* p) {
  double a = p[0], s = 0.0;
  for (int i = 0; i < 2; ++i) {
    double x = i + 1.0;
    s += 1.0 / (x + a * a);
  }
  return s;
}

// CHECK: void inline_divisor_2_grad(double *p, double *_d_p) {
// CHECK: double _r{{[0-9]+}} = clad::pop(_t{{[0-9]+}});
// CHECK-NEXT: double _r{{[0-9]+}} = _d_s * -(1. / (_r{{[0-9]+}} * _r{{[0-9]+}}));

// Correct at one iteration before #1865, wrong for two or more.
double inline_divisor_3(double* p) {
  double a = p[0], s = 0.0;
  for (int i = 0; i < 3; ++i) {
    double x = i + 1.0;
    s += 1.0 / (x + a * a);
  }
  return s;
}

// Naming the quotient did not avoid #1865.
double named_quotient_4(double* p) {
  double a = p[0], s = 0.0;
  for (int i = 0; i < 4; ++i) {
    double x = i + 1.0;
    double r = 1.0 / (x + a * a);
    s += r;
  }
  return s;
}

#define CHECK_GRAD(F)                                                          \
  do {                                                                         \
    auto grad = clad::gradient(F, "p");                                        \
    double p[1] = {1.3};                                                       \
    double d[1] = {0.0};                                                       \
    grad.execute(p, d);                                                        \
    printf(#F " %.6f\n", d[0]);                                               \
  } while (0)

int main() {
  CHECK_GRAD(inline_divisor_2); // CHECK-EXEC: inline_divisor_2 -0.550260
  CHECK_GRAD(inline_divisor_3); // CHECK-EXEC: inline_divisor_3 -0.668463
  CHECK_GRAD(named_quotient_4); // CHECK-EXEC: named_quotient_4 -0.748769
}