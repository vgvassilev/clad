// RUN: %cladclang -std=c++20 -ftrivial-auto-var-init=pattern -I%S/../../include %s -o %t 2>&1 | %filecheck %s
// RUN: %t | %filecheck_exec %s

// The adjoint of `V x;` copied the primal's default-initialization and read
// indeterminate members. The pattern fill makes that visible. Reduced from
// ImpactX's SmallMatrix.

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

struct V {
  double m[2];
  constexpr V() = default;
};

V sw(V const& a) {
  V r;
  r.m[0] = a.m[1];
  r.m[1] = a.m[0];
  return r;
}

double f(double k) {
  V a{};
  a.m[0] = k;
  V b = sw(a);
  return b.m[1] * a.m[0];
}

// CHECK: void sw_pullback(const V &a, V _d_y, V *_d_a) {
// CHECK-DAG:     V r;
// CHECK-DAG:     V _d_r{};
// CHECK:         r.m[0] = a.m[1];

// An array keeps `T x[N];`, which compiles with an explicit constructor.
struct E {
  explicit E() = default;
  double v;
};
double g(double x) {
  E a[2];
  a[1].v = x;
  return a[1].v * x;
}
// CHECK: void g_grad(double x, double *_d_x) {
// CHECK-NEXT:     E _d_a[2];

int main() {
  double dk = 0;
  auto df = clad::gradient(f);
  df.execute(0.7, &dk);
  printf("%.2f\n", dk); // CHECK-EXEC: 1.40
  auto dg = clad::gradient(g);
  (void)dg;
}
