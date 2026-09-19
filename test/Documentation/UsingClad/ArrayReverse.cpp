// RUN: %cladclang %s -I%S/../../../include -o%t 2>&1 | %filecheck_nodiag %s
// RUN: %t | %filecheck_prints %s

// docs-begin-array-reverse
#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

double g(double x, double arr[2]) { return x * arr[0] + x * arr[1]; }

int main() {
  // Differentiating g w.r.t all the input variables (x, arr).
  auto g_grad = clad::gradient(g);

  double x = 2, arr[2] = {1, 2};
  // Create memory for the output of differentiation. clad adds into it, so
  // it starts at zero.
  double dx = 0, darr[2] = {0};

  // The inputs to the original function g (i.e x and arr) are passed
  // followed by the variables to store the output (i.e dx and darr).
  g_grad.execute(x, arr, &dx, darr);

  printf("dg/dx = %g \ndg/darr = { %g, %g } \n", dx, darr[0], darr[1]);
  // prints: dg/dx = 3
  // prints: dg/darr = { 2, 2 }
}
// docs-end-array-reverse
