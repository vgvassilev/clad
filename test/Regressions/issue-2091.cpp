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

  return 0;
}
