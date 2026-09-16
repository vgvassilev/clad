// RUN: %cladclang %s -I%S/../../../include -o%t 2>&1 | %filecheck_nodiag %s
// RUN: %t | %filecheck_prints %s
// REQUIRES: Enzyme

// docs-begin-enzyme
#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

double array_product(double* arr) { return arr[0] * arr[1]; }

int main() {
  auto grad = clad::gradient<clad::opts::use_enzyme>(array_product);
  double v[2] = {3, 4};
  double g[2] = {0};
  grad.execute(v, g);
  printf("d_x = %.2f, d_y = %.2f\n", g[0], g[1]); // prints: d_x = 4.00, d_y = 3.00
}
// docs-end-enzyme
