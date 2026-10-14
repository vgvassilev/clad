// RUN: %cladclang %s -I%S/../../include -std=c++20 \
// RUN:     -oConstexprReverseMode.out | %filecheck %s
// RUN: ./ConstexprReverseMode.out | %filecheck_exec %s
// UNSUPPORTED: clang-14, clang-15, clang-16

// A reverse-mode gradient worked out while the program compiles.
//
// Full gradients have exact types from FullGradientDerivedFnTraits matching
// the generated gradient, so they evaluate during compilation under C++20.

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

constexpr double prod(double a, double b, double c) { return a * b * c; }

//CHECK: constexpr void prod_grad(double a, double b, double c, double *_d_a, double *_d_b, double *_d_c) {

constexpr double d_wrt_b() {
  auto g = clad::gradient(prod);
  double da = 0, db = 0, dc = 0;
  g.execute(2., 3., 5., &da, &db, &dc);
  return db;
}

// A primal that returns before its tail. Its forward sweep runs in a closure
// that each early return leaves, and a return from a closure is a constant
// expression where a goto is not ([expr.const]).
constexpr double larger(double a, double b) {
  if (a > b)
    return a * a;
  return a * b;
}

//CHECK: constexpr void larger_grad(double a, double b, double *_d_a, double *_d_b) {
//CHECK-NEXT:     bool _cond0 = false;
//CHECK-NEXT:     clad::forward_sweep([&] {

constexpr double d_larger_wrt_a(double a, double b) {
  auto g = clad::gradient(larger);
  double da = 0, db = 0;
  g.execute(a, b, &da, &db);
  return da;
}

int main() {
  // Worked out during compilation: d(a*b*c)/db at (2,3,5) is a*c.
  constexpr double compiled = d_wrt_b();
  static_assert(compiled == 10., "the gradient is wrong at compile time");
  printf("%.0f\n", compiled);
  //CHECK-EXEC: 10

  // Both paths of the early return, worked out during compilation: 2a on
  // the early path, b on the fall-through.
  static_assert(d_larger_wrt_a(5., 3.) == 10., "the early path is wrong");
  static_assert(d_larger_wrt_a(3., 5.) == 5., "the fall-through is wrong");
  printf("%.0f %.0f\n", d_larger_wrt_a(5., 3.), d_larger_wrt_a(3., 5.));
  //CHECK-EXEC: 10 5
}
