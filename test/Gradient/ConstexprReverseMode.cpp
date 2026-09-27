// RUN: %cladclang %s -I%S/../../include -std=c++2c \
// RUN:     -oConstexprReverseMode.out | %filecheck %s
// RUN: ./ConstexprReverseMode.out | %filecheck_exec %s
// UNSUPPORTED: clang-14, clang-15, clang-16

// A reverse-mode gradient worked out while the program compiles.
//
// This needs C++26. Clad calls the generated gradient through a pointer whose
// adjoint parameters are void*, from GradientDerivedFnTraits, while the
// function itself takes double*, and a cast from void* only became a constant
// expression in C++26. The guard below keeps the test honest on a compiler
// that takes -std=c++2c without having that relaxation yet: the answer is
// checked either way, only the compile-time assertion is skipped. See #2188
// for the typing this is working around.

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
}
