// RUN: %cladclang %s -I%S/../../../include -o%t 2>&1 | %filecheck_nodiag %s
// RUN: %t | %filecheck_prints %s

// docs-begin-jacobian
#include "clad/Differentiator/Differentiator.h"
#include <iostream>

void f(double x, double y, double z, double* output) {
  output[0] = x * y;
  output[1] = y * y * x;
  output[2] = 6 * x * y * z;
}

int main() {
  auto f_jac = clad::jacobian(f);

  // One row per output element; one column per independent scalar, counting
  // the three elements output itself contributes.
  clad::matrix<double> d_output(3, 6);
  double output[3] = {0};
  f_jac.execute(3, 4, 5, output, &d_output);
  for (int row = 0; row < 3; ++row)
    std::cout << d_output[row][0] << " " << d_output[row][1] << " "
              << d_output[row][2] << "\n";
  // prints: 4 3 0
  // prints: 16 24 0
  // prints: 120 90 72
}
// docs-end-jacobian
