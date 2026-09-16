// RUN: %cladclang %s -I%S/../../../include -o%t 2>&1 | %filecheck_nodiag %s
// RUN: %t | %filecheck_prints %s

// docs-begin-jacobian
#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

void func(double i, double j, double result[]) {
  result[0] = i * i * j;
  result[1] = j * j * i;
  result[2] = j * i;
}

int main() {
  auto fn_jcbn = clad::jacobian(func);

  // One row per element of result, one column per independent scalar: i, j and
  // the three elements of result itself.
  clad::matrix<double> d_res(3, 5);
  double res[3] = {0};

  fn_jcbn.execute(8, 2, res, &d_res);

  printf("Result is\n %g %g\n %g %g\n %g %g\n", d_res[0][0], d_res[0][1],
         d_res[1][0], d_res[1][1], d_res[2][0], d_res[2][1]);
  // prints: Result is
  // prints:  32 64
  // prints:  4 32
  // prints:  2 8
}
// docs-end-jacobian
