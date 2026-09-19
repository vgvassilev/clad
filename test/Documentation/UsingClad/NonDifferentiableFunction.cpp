// RUN: %cladclang %s -I%S/../../../include -o%t 2>&1 | %filecheck_nodiag %s
// RUN: %t | %filecheck_prints %s

// docs-begin-non-differentiable-function
#include "clad/Differentiator/Differentiator.h"
#include <iostream>

CLAD_NONDIFFERENTIABLE double get_scaling_factor(double i, double j) {
  return i * j;
}

double compute(double i, double j) {
  // get_scaling_factor will skip differentiation completely.
  return get_scaling_factor(i, j) + i * j;
}

int main() {
  auto d_compute = clad::gradient(compute);

  double d_i = 0, d_j = 0;
  d_compute.execute(3, 5, &d_i, &d_j);

  // Only the second term contributes: the call behaves as a constant.
  std::cout << d_i << " " << d_j << "\n";
  // prints: 5 3
}
// docs-end-non-differentiable-function
