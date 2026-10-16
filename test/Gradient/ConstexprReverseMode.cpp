// RUN: %cladclang %s -I%S/../../include -std=c++2c \
// RUN:     -oConstexprReverseMode.out | %filecheck %s
// RUN: ./ConstexprReverseMode.out | %filecheck_exec %s
// RUN: %cladclang %s -I%S/../../include -std=c++20 -fsyntax-only
// UNSUPPORTED: clang-14, clang-15, clang-16

// A reverse-mode gradient worked out while the program compiles.
//
// When every parameter of the function has the same floating-point type, the
// adjoints clad generates all have one pointer type whatever the argument
// spec selects, so the gradient is stored and called with that type and no
// cast is involved. Such a gradient is a constant expression in C++20; the
// static_asserts on f below check that, also under -std=c++20. See #2190.
//
// The guards in main keep the test honest on a compiler that takes -std=c++2c
// without having all the C++26 constexpr relaxations yet: the answer is
// checked either way, only the compile-time assertion is skipped. See #2188.

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

constexpr double f(double x, double y) {
  return x * y;
}

constexpr double test_full() {
  auto g = clad::gradient(f);

  double dx = 0;
  double dy = 0;

  g.execute(3., 5., &dx, &dy);

  return dx + dy;
}

constexpr double test_partial() {
  auto g = clad::gradient(f, "x");

  double dx = 0;

  g.execute(3., 5., &dx);

  return dx;
}

static_assert(test_full() == 8.);
static_assert(test_partial() == 5.);

int main() {
#if __cpp_constexpr >= 202406L
  // Worked out during compilation: d(a*b*c)/db at (2,3,5) is a*c.
  constexpr double compiled = d_wrt_b();
  static_assert(compiled == 10., "the gradient is wrong at compile time");
  printf("%.0f\n", compiled);
#else
  printf("%.0f\n", d_wrt_b());
#endif
  //CHECK-EXEC: 10

#if __cpp_constexpr >= 202406L
  // Both paths of the early return, worked out during compilation: 2a on
  // the early path, b on the fall-through.
  static_assert(d_larger_wrt_a(5., 3.) == 10., "the early path is wrong");
  static_assert(d_larger_wrt_a(3., 5.) == 5., "the fall-through is wrong");
#endif
  printf("%.0f %.0f\n", d_larger_wrt_a(5., 3.), d_larger_wrt_a(3., 5.));
  //CHECK-EXEC: 10 5
}
