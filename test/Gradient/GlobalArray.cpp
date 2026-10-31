// RUN: %cladclang %s -I%S/../../include -oGlobalArray.out 2>&1 | %filecheck %s
// RUN: ./GlobalArray.out | %filecheck_exec %s

// A gradient resets the adjoint of a global it uses. An array cannot be
// assigned, so it is reset element by element.

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

// CHECK: warning: gradient uses a global variable 'g'; rerunning the gradient requires 'g' to be reset
double g[2];

// f = x^2, f' = 2 x.
double f(double x) {
  g[1] = x;
  return g[1] * g[1];
}

// CHECK: void f_grad(double x, double *_d_x) {
// CHECK-NEXT:     clad::zero_init(_d_g);

int main() {
  auto df = clad::gradient(f);
  double dx = 0;
  df.execute(1.5, &dx);
  printf("%.2f\n", dx); // CHECK-EXEC: 3.00
  dx = 0;
  df.execute(1.5, &dx);
  printf("%.2f\n", dx); // CHECK-EXEC: 3.00
}
