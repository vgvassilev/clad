// RUN: %cladclang %s -I%S/../../../include -o%t 2>&1 | %filecheck_nodiag %s
// RUN: %t | %filecheck_prints %s

// docs-begin-array-forward
#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

double f(double arr[4]) { return arr[0] * arr[1] * arr[2] * arr[3]; }

int main() {
  // Differentiating the function f w.r.t arr[1]:
  auto f_diff = clad::differentiate(f, "arr[1]");

  double arr[4] = {1, 2, 3, 4};
  // Pass the input to f to the execute function.
  // The output is stored in a variable with the same type as the return type
  // of f.
  double f_dx = f_diff.execute(arr);

  printf("df/darr[1] = %g\n", f_dx); // prints: df/darr[1] = 12
}
// docs-end-array-forward
