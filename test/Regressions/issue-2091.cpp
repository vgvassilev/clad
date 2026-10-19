// RUN: %cladclang -std=c++17 -I%S/../../include -Xclang -verify %s -o %t
// RUN: %t | %filecheck_exec %s

// expected-no-diagnostics

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

// Issue #2091: Support differentiating a function that throws.
// A throw is a control-flow edge that leaves the function:
// nothing after the throw executes on the original path, and the reverse
// sweep must not replay the part of the primal that never ran.

double f(double x) {
  if (x < 0)
    throw 1;
  return x * x;
}

double cond(double x) {
  return x > 0 ? x * x : throw 1;
}

double cond_throw_true(double x) {
  return x < 0 ? throw 1 : x * x;
}

int main() {
  auto df = clad::differentiate(f, "x");
  auto f_grad = clad::gradient(f, "x");

  // 1. Forward mode normal execution
  printf("df(3.0) = %.2f\n", df.execute(3.0));
  // CHECK-EXEC: df(3.0) = 6.00

  // 2. Forward mode throw execution
  try {
    df.execute(-3.0);
  } catch (int e) {
    printf("Caught throw in forward mode: %d\n", e);
  }
  // CHECK-EXEC: Caught throw in forward mode: 1

  // 3. Reverse mode normal execution
  double dx = 0;
  f_grad.execute(3.0, &dx);
  printf("f_grad(3.0) = %.2f\n", dx);
  // CHECK-EXEC: f_grad(3.0) = 6.00

  // 4. Reverse mode throw execution
  dx = 0;
  try {
    f_grad.execute(-3.0, &dx);
  } catch (int e) {
    printf("Caught throw in reverse mode: %d\n", e);
  }
  // CHECK-EXEC: Caught throw in reverse mode: 1

  auto dcond = clad::differentiate(cond, "x");
  auto cond_grad = clad::gradient(cond, "x");

  // 5. Forward mode conditional normal execution
  printf("dcond(3.0) = %.2f\n", dcond.execute(3.0));
  // CHECK-EXEC: dcond(3.0) = 6.00

  // 6. Forward mode conditional throw execution
  try {
    dcond.execute(-3.0);
  } catch (int e) {
    printf("Caught conditional throw in forward mode: %d\n", e);
  }
  // CHECK-EXEC: Caught conditional throw in forward mode: 1

  // 7. Reverse mode conditional normal execution
  dx = 0;
  cond_grad.execute(3.0, &dx);
  printf("cond_grad(3.0) = %.2f\n", dx);
  // CHECK-EXEC: cond_grad(3.0) = 6.00

  // 8. Reverse mode conditional throw execution
  dx = 0;
  try {
    cond_grad.execute(-3.0, &dx);
  } catch (int e) {
    printf("Caught conditional throw in reverse mode: %d\n", e);
  }
  // CHECK-EXEC: Caught conditional throw in reverse mode: 1

  auto dcond2 = clad::differentiate(cond_throw_true, "x");
  auto cond2_grad = clad::gradient(cond_throw_true, "x");

  // 9. Forward mode conditional (true arm throws) normal execution
  printf("dcond2(3.0) = %.2f\n", dcond2.execute(3.0));
  // CHECK-EXEC: dcond2(3.0) = 6.00

  // 10. Forward mode conditional (true arm throws) throw execution
  try {
    dcond2.execute(-3.0);
  } catch (int e) {
    printf("Caught true-arm throw in forward mode: %d\n", e);
  }
  // CHECK-EXEC: Caught true-arm throw in forward mode: 1

  // 11. Reverse mode conditional (true arm throws) normal execution
  dx = 0;
  cond2_grad.execute(3.0, &dx);
  printf("cond2_grad(3.0) = %.2f\n", dx);
  // CHECK-EXEC: cond2_grad(3.0) = 6.00

  // 12. Reverse mode conditional (true arm throws) throw execution
  dx = 0;
  try {
    cond2_grad.execute(-3.0, &dx);
  } catch (int e) {
    printf("Caught true-arm throw in reverse mode: %d\n", e);
  }
  // CHECK-EXEC: Caught true-arm throw in reverse mode: 1

  return 0;
}
