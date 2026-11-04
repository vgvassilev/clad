// RUN: %cladclang %s -I%S/../../include -oReferenceToTemporary.out
// RUN: ./ReferenceToTemporary.out | %filecheck_exec %s

// A const reference bound to a temporary keeps the temporary alive, so it
// holds a value of its own and needs an adjoint of its own.

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

// f = 2 x^2, f' = 4 x.
double f(double x) {
  const double& r = x * 2;
  return r * x;
}

// In a loop the reference is moved out of the loop body.
// g = sum_{i<3} i x^2 = 3 x^2, g' = 6 x.
double g(double x) {
  double s = 0;
  for (int i = 0; i < 3; ++i) {
    const double& r = x * i;
    s += r * x;
  }
  return s;
}

int main() {
  double dx = 0;
  clad::gradient(f).execute(3, &dx);
  printf("%.2f\n", dx); // CHECK-EXEC: 12.00
  dx = 0;
  clad::gradient(g).execute(3, &dx);
  printf("%.2f\n", dx); // CHECK-EXEC: 18.00
}
