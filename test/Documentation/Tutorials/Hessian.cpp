// RUN: %cladclang %s -I%S/../../../include -o%t 2>&1 | %filecheck_nodiag %s
// RUN: %t | %filecheck_prints %s

// docs-begin-hessian
#include "clad/Differentiator/Differentiator.h"
#include <iostream>

double f(double x, double y, double z) { return x * y * z; }

int main() {
  // Two independent variables, so the hessian is 2 x 2 and needs 4 elements.
  auto f_hess = clad::hessian(f, "x, y");
  double matrix_f[4] = {0};
  f_hess.execute(3, 4, 5, matrix_f);
  std::cout << "[" << matrix_f[0] << ", " << matrix_f[1] << "\n"
            << " " << matrix_f[2] << ", " << matrix_f[3] << "]\n";
  // prints: [0, 5
  // prints:  5, 0]
}
// docs-end-hessian
