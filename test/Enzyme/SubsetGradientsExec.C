// RUN: %cladclang %s -I%S/../../include -oSubsetGradientsExec.out 2>&1 \
// RUN:   | %filecheck %s
// RUN: ./SubsetGradientsExec.out | %filecheck_exec %s
// REQUIRES: Enzyme

// The activity-annotated call a subset request emits has to mean what the
// positional one means. Checking the emitted code cannot show that, so pin
// the numbers: a wrong marker shifts arguments and the gradient moves.

#include "clad/Differentiator/Differentiator.h"

#include <cstdio>

double byValue(double x, double y, double z) { return x * y * z; }

double byPointer(const double* a, const double* b, int n) {
  double s = 0;
  for (int i = 0; i < n; i++)
    s += a[i] * a[i] * b[i];
  return s;
}

// CHECK: void byValue_grad_0(
// CHECK: void byPointer_grad_0(

int main() {
  auto g1 = clad::gradient<clad::opts::use_enzyme>(byValue, "x");
  double dx = 0;
  g1.execute(2, 3, 5, &dx);
  printf("dx=%.2f\n", dx);
  // df/dx = y*z = 15
  // CHECK-EXEC: dx=15.00

  auto g2 = clad::gradient<clad::opts::use_enzyme>(byPointer, "a");
  double a[3] = {1, 2, 3};
  double b[3] = {4, 5, 6};
  double da[3] = {};
  g2.execute(a, b, 3, da);
  printf("da={%.2f, %.2f, %.2f}\n", da[0], da[1], da[2]);
  // df/da_i = 2 a_i b_i => {8, 20, 36}
  // CHECK-EXEC: da={8.00, 20.00, 36.00}
}
