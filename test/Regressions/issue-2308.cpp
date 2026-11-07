// RUN: %cladclang -std=c++17 -I%S/../../include %s -o %t 2>&1 | %filecheck %s
// RUN: %t | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

struct T {
  double x;
};

// Differentiating with respect to another parameter: the const T parameter's
// derivative must be declared non-const (T _d_t, not const T _d_t).
double g(const T t, double y) {
  return t.x * y;
}

// CHECK: double g_darg1(const T t, double y) {
// CHECK-NOT: const T _d_t
// CHECK: T _d_t
// CHECK: double _d_y = 1;
// CHECK: return

// Differentiating with respect to another parameter with a const reference
// parameter: the derivative must be declared non-const (T _d_t).
double g_ref(const T& t, double y) {
  return t.x * y;
}

// CHECK: double g_ref_darg1(const T &t, double y) {
// CHECK-NOT: const T _d_t
// CHECK: T _d_t
// CHECK: double _d_y = 1;
// CHECK: return

// Differentiating with respect to a field of the const parameter: seeding
// _d_t.x = 1 must compile and succeed.
double f(const T t) {
  return t.x * t.x;
}

// CHECK: double f_darg0_x(const T t) {
// CHECK-NOT: const T _d_t
// CHECK: T _d_t
// CHECK: _d_t.x = 1;
// CHECK: return

struct Point {
  double x;
  double y;
};

double h(const Point p) {
  return p.x * p.x + p.y * p.y;
}

// CHECK: double h_darg0_x(const Point p) {
// CHECK-NOT: const Point _d_p
// CHECK: Point _d_p
// CHECK: _d_p.x = 1;
// CHECK: return

// CHECK: double h_darg0_y(const Point p) {
// CHECK-NOT: const Point _d_p
// CHECK: Point _d_p
// CHECK: _d_p.y = 1;
// CHECK: return

int main() {
  auto d_g = clad::differentiate(g, "y");
  printf("Result g: %g\n", d_g.execute(T{3}, 2)); // CHECK-EXEC: Result g: 3

  T t3{3};
  auto d_g_ref = clad::differentiate(g_ref, "y");
  printf("Result g_ref: %g\n",
         d_g_ref.execute(t3, 2)); // CHECK-EXEC: Result g_ref: 3

  auto d_f = clad::differentiate(f, "t.x");
  printf("Result f: %g\n", d_f.execute(T{4})); // CHECK-EXEC: Result f: 8

  auto d_hx = clad::differentiate(h, "p.x");
  printf("Result hx: %g\n",
         d_hx.execute(Point{3, 4})); // CHECK-EXEC: Result hx: 6

  auto d_hy = clad::differentiate(h, "p.y");
  printf("Result hy: %g\n",
         d_hy.execute(Point{3, 4})); // CHECK-EXEC: Result hy: 8
}
