// RUN: %cladclang %s -I%S/../../../include -o%t 2>&1 | %filecheck_nodiag %s
// RUN: %t | %filecheck_prints %s

// docs-begin-estimate-error
#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

double func(double x, double y) {
  double z = x * y;
  return z + x;
}

int main() {
  auto fn_err = clad::estimate_error(func);

  double d_x = 0, d_y = 0, error = 0;
  fn_err.execute(3, 5, &d_x, &d_y, error);

  printf("Result is %g, %g with an error of %.2e\n", d_x, d_y, error);
  // prints: Result is 6, 3 with an error of 7.87e-06
}
// docs-end-estimate-error
