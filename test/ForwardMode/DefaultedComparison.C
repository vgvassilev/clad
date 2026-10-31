// RUN: %cladclang -std=c++20 %s -I%S/../../include -o%t -Xclang -verify 2>&1 | %filecheck %s
// RUN: %t | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"
#include <compare>
#include <cstdio>

// expected-no-diagnostics

struct S {
  double v;
  auto operator<=>(const S&) const = default;
};

double f(double x) {
  S a{x}, b{1.0};
  return (a < b) ? x : x * 2;
}

// CHECK-LABEL: double f_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     S _d_a{_d_x}, _d_b{0.};
// CHECK-NEXT:     S a{x}, b{1.};
// CHECK-NEXT:     bool _t0 = (a < b);
// CHECK-NEXT:     return _t0 ? _d_x : _d_x * 2;
// CHECK-NEXT: }

double g(double x) {
  S a{x}, b{1.0};
  bool c = a < b;
  return c ? x : x * 2;
}

// CHECK-LABEL: double g_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     S _d_a{_d_x}, _d_b{0.};
// CHECK-NEXT:     S a{x}, b{1.};
// CHECK-NEXT:     bool _d_c = false;
// CHECK-NEXT:     bool c = a < b;
// CHECK-NEXT:     return c ? _d_x : _d_x * 2;
// CHECK-NEXT: }

// CHECK-LABEL: void f_grad(double x, double *_d_x) {
// CHECK-NEXT:     S _d_a = {0.}, _d_b = {0.};
// CHECK-NEXT:     S a{x}, b{1.};
// CHECK-NEXT:     bool _cond0 = a < b;
// CHECK-NEXT:     if (_cond0)
// CHECK-NEXT:         *_d_x += 1;
// CHECK-NEXT:     else
// CHECK-NEXT:         *_d_x += 1 * 2;
// CHECK-NEXT:     *_d_x += _d_a.v;
// CHECK-NEXT: }

int main() {
  auto df = clad::differentiate(f, "x");
  printf("%.1f %.1f\n", df.execute(0.5), df.execute(2)); // CHECK-EXEC: 1.0 2.0
  auto dg = clad::differentiate(g, "x");
  printf("%.1f %.1f\n", dg.execute(0.5), dg.execute(2)); // CHECK-EXEC: 1.0 2.0
  auto gradf = clad::gradient(f);
  double dx = 0, dx2 = 0;
  gradf.execute(0.5, &dx);
  gradf.execute(2, &dx2);
  printf("%.1f %.1f\n", dx, dx2); // CHECK-EXEC: 1.0 2.0
}
