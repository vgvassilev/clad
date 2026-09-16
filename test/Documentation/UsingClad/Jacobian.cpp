// RUN: %cladclang %s -I%S/../../../include -o%t 2>&1 | %filecheck_nodiag %s
// RUN: %t | %filecheck_prints %s

// docs-begin-jacobian
#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

void fn_jacobian(double i, double j, double* res) {
  res[0] = i * i;
  res[1] = j * j;
  res[2] = i * j;
}

int main() {
  // Generates all first-order partial derivatives columns of a jacobian matrix
  // and stores CallExprs to them inside a single function.
  auto jacobian = clad::jacobian(fn_jacobian);

  // An empty matrix to store the jacobian in. It must have enough space:
  // 5 columns (the sum of the independent variable sizes) and 3 rows (the
  // size of res).
  clad::matrix<double> d_res(3, 5);

  // Substitutes these values into the jacobian function and pipes the result
  // into the d_res variable.
  double res[3] = {0, 0, 0};
  jacobian.execute(3, 5, res, &d_res);

  // Now the derivatives are available as d_res[i][j].
  printf("%g %g\n%g %g\n%g %g\n", d_res[0][0], d_res[0][1], d_res[1][0],
         d_res[1][1], d_res[2][0], d_res[2][1]);
  // prints: 6 0
  // prints: 0 10
  // prints: 5 3
}
// docs-end-jacobian
