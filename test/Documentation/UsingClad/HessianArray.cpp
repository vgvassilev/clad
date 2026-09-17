// RUN: %cladclang %s -I%S/../../../include -o%t 2>&1 | %filecheck_nodiag %s
// RUN: %t | %filecheck_prints %s

// docs-begin-hessian-array
#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

double fn(double x, double arr[2]) { return x * arr[0] * arr[1]; }

int main() {
  auto fn_hessian = clad::hessian(fn, "x, arr[0:1]");

  // We have 3 independent variables, thus we require space for 9 elements.
  double mat_fn[9] = {0};
  double num[2] = {1, 2};
  fn_hessian.execute(3, num, mat_fn);

  printf("%g %g %g\n%g %g %g\n%g %g %g\n", mat_fn[0], mat_fn[1], mat_fn[2],
         mat_fn[3], mat_fn[4], mat_fn[5], mat_fn[6], mat_fn[7], mat_fn[8]);
  // prints: 0 2 1
  // prints: 2 0 3
  // prints: 1 3 0
}
// docs-end-hessian-array
