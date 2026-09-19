// RUN: %cladclang %s -I%S/../../../include -o%t 2>&1 | %filecheck_nodiag %s
// RUN: %t | %filecheck_prints %s

// docs-begin-gradient
#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

double func(double i, double j) { return 5 * i * i + 2 * j; }

int main() {
  auto fn_grad = clad::gradient(func);
  double d_i = 0, d_j = 0;
  fn_grad.execute(3, 5, &d_i, &d_j);
  printf("Result is %g, %g\n", d_i, d_j); // prints: Result is 30, 2
}
// docs-end-gradient
