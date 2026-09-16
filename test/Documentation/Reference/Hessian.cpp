// RUN: %cladclang %s -I%S/../../../include -o%t 2>&1 | %filecheck_nodiag %s
// RUN: %t | %filecheck_prints %s

// docs-begin-hessian
#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

double func(double i, double j) {
  double a = i * j;
  double b = 4 * a;
  return b * i;
}

int main() {
  auto fn_hesn = clad::hessian(func);

  // Two independent variables, so the hessian needs 2 * 2 elements.
  double matrix[4] = {0};
  fn_hesn.execute(8, 2, matrix);

  printf("Result is %g, %g, %g, %g\n", matrix[0], matrix[1], matrix[2],
         matrix[3]); // prints: Result is 16, 64, 64, 0
}
// docs-end-hessian
