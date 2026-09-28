// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -fdump-derived-fn %s -I%S/../../include -oConstantFolding.out 2>&1 | %filecheck %s
// RUN: ./ConstantFolding.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

// This tests the `ConstantFolder` component in Clad.
// The constant folder is responsible for simplifying AST expressions
// generated during the derivative pass (e.g. removing `* 1` or `* 0`, 
// or folding `0 + 0`).

double test_fold(double x, double y) {
  // y is treated as a constant if we differentiate w.r.t. x.
  // The derivative w.r.t x of (x * 1.0 + y * 0.0) should be folded heavily.
  return (x * 1.0) + (y * 0.0);
}

// CHECK: double test_fold_darg0(double x, double y) {
// CHECK-NOT: * 0
// CHECK-NOT: * 1
// CHECK: }

double test_pow(double x) {
  // Derivative of x * x * x involves a lot of additions and multiplications.
  // Constant folder should simplify the intermediate zeros from constant derivatives.
  return x * x * x;
}

// CHECK: double test_pow_darg0(double x) {
// CHECK-NOT: * 0
// CHECK: }

int main() {
  auto df_fold = clad::differentiate(test_fold, "x");
  double dx1 = df_fold.execute(2.0, 3.0);
  printf("dx1 = %.2f\n", dx1); // CHECK-EXEC: dx1 = 1.00

  auto df_pow = clad::differentiate(test_pow, "x");
  double dx2 = df_pow.execute(3.0);
  // df/dx = 3 * x^2 = 3 * 9 = 27
  printf("dx2 = %.2f\n", dx2); // CHECK-EXEC: dx2 = 27.00

  return 0;
}
