// RUN: %cladclang %s -I%S/../../include -oCompoundAssignValue.out 2>&1 \
// RUN:   | %filecheck %s
// RUN: ./CompoundAssignValue.out | %filecheck_exec %s

// The reverse sweep reads the value of `j -= e` as j - e and that of
// `j /= e` as j / e, with j as it was before the assignment.

#include "clad/Differentiator/Differentiator.h"

#include <cstdio>

double difference(double x) {
  double c[4] = {x, 2 * x, 3 * x, 4 * x};
  int j = 4, k = 2;
  return c[j -= k - 1] * x;
}

// CHECK: void difference_grad(double x, double *_d_x) {
// CHECK: _d_c[j - (k - 1)] += 1 * x;

double quotient(double x) {
  double c[4] = {x, 2 * x, 3 * x, 4 * x};
  int j = 5, k = 1;
  return c[j /= k + 1] * x;
}

// CHECK: void quotient_grad(double x, double *_d_x) {
// CHECK: _d_c[j / (k + 1)] += 1 * x;

int main() {
  double dx = 0;
  clad::gradient(difference).execute(3, &dx);
  printf("%.2f\n", dx); // CHECK-EXEC: 24.00
  dx = 0;
  clad::gradient(quotient).execute(3, &dx);
  printf("%.2f\n", dx); // CHECK-EXEC: 18.00
}
