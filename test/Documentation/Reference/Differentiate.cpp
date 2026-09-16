// RUN: %cladclang %s -I%S/../../../include -o%t 2>&1 | %filecheck_nodiag %s
// RUN: %t | %filecheck_prints %s

// docs-begin-differentiate
#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

double func(double x, double y) { return x * x * y + y * y; }

int main() {
  // fn_dx is a CladFunction, a tiny wrapper over the derived function pointer.
  // It differentiates 'func' with respect to 'x'.
  auto fn_dx = clad::differentiate(func, "x");

  // Call it at (x, y) = (5, 3).
  double func1stOrderDerivative = fn_dx.execute(5, 3);
  printf("Result is %g\n", func1stOrderDerivative); // prints: Result is 30
}
// docs-end-differentiate
