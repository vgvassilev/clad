// RUN: %cladclang -std=c++20 -I%S/../../include %s -o %t 2>&1
// RUN: %t | %filecheck_exec %s
// UNSUPPORTED: clang-14, clang-15, clang-16

// Issue #2190: clad::gradient calls the generated function through a mismatched pointer type.
// For full gradients without an argument specifier, the pointer type should be exact
// (e.g. double* instead of void*) so that the call is well-defined and can evaluate
// in a C++20 constant expression.

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>
#include <type_traits>

constexpr double f(double x, double y, double z) {
  return x * y * z;
}

constexpr double eval_full_gradient() {
  auto all = clad::gradient(f);
  // Verify that the derived function pointer type has exact double* adjoint parameters
  static_assert(std::is_same<decltype(all.getFunctionPtr()),
                             void (*)(double, double, double, double*, double*, double*)>::value);
  double dx = 0, dy = 0, dz = 0;
  all.execute(2.0, 3.0, 4.0, &dx, &dy, &dz);
  return dx + dy + dz;
}

struct Multiplier {
  double x, y;
  double compute(double a, double b) const { return (x + y) * a + a * b; }
};

int main() {
  // Test full gradient evaluated as a constant expression under C++20
  constexpr double full_res = eval_full_gradient();
  static_assert(full_res == (3.0 * 4.0) + (2.0 * 4.0) + (2.0 * 3.0)); // 12 + 8 + 6 = 26
  printf("full: %.2f\n", full_res);
  // CHECK-EXEC: full: 26.00

  // Test partial gradient with string spec at runtime
  auto one = clad::gradient(f, "y");
  // Partial gradient uses void* in traits to accommodate any subset of arguments
  static_assert(std::is_same<decltype(one.getFunctionPtr()),
                             void (*)(double, double, double, void*, void*, void*)>::value);
  double dy = 0;
  one.execute(2.0, 3.0, 4.0, &dy);
  printf("partial: %.2f\n", dy);
  // CHECK-EXEC: partial: 8.00

  // Test member function full gradient type
  auto gm = clad::gradient(&Multiplier::compute);
  static_assert(std::is_same<decltype(gm.getFunctionPtr()),
                             void (Multiplier::*)(double, double, Multiplier*, double*, double*) const>::value);
  Multiplier m{2.0, 3.0}, dm{};
  double da = 0, db = 0;
  gm.execute(m, 4.0, 5.0, &dm, &da, &db);
  printf("member: {%.2f, %.2f, %.2f, %.2f}\n", da, db, dm.x, dm.y);
  // CHECK-EXEC: member: {10.00, 4.00, 4.00, 4.00}
}
