// RUN: %cladclang -std=c++17 -I%S/../../include %s -o %t 2>&1 | %filecheck %s
// RUN: %t | %filecheck_exec %s

// A pullback inherited its primal's [[nodiscard]], and clad dropped the call
// to it as an unused result. Reduced from ImpactX's SmallMatrix.

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

struct V { double m[2]; };

[[nodiscard]] V sw(V const& a) {
  V r;
  r.m[0] = a.m[1];
  r.m[1] = a.m[0];
  return r;
}

double nodiscard_fn(double k) {
  V a{};
  a.m[0] = k;
  V b = sw(a);
  return b.m[1] * a.m[0];
}

// CHECK-NOT: nodiscard
// CHECK: void sw_pullback(const V &a, V _d_y, V *_d_a) {
// CHECK: void nodiscard_fn_grad(double k, double *_d_k) {
// CHECK: sw_pullback(a, _d_b, &_d_a);

int main() {
  double dk = 0;
  clad::gradient(nodiscard_fn).execute(0.7, &dk);
  printf("%.2f\n", dk); // CHECK-EXEC: 1.40
}
