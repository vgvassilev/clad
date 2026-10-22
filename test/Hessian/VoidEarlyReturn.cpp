// RUN: %cladclang %s -I%S/../../include -oVoidEarlyReturn.out 2>&1 | %filecheck %s
// RUN: ./VoidEarlyReturn.out | %filecheck_exec %s

// The pushforward of a void function dropped its early `return;`, and so
// ran what the primal skips; a hessian, the pullback of that pushforward,
// went through the rest as well.

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

void push(double* a, double I, double k) {
  if (I == 0)
    return;
  a[0] = a[0] * k + I * a[1];
  a[1] = a[1] * k;
}

// CHECK: void push_pushforward(double *a, double I, double k, double *_d_a, double _d_I, double _d_k) {
// CHECK-NEXT:     if (I == 0)
// CHECK-NEXT:         return;

// k1 k2^2
double off(double k1, double k2) {
  double a[2] = {k2, k1};
  push(a, 0.0, k1);
  a[0] = a[0] * a[1] * k2;
  return a[0];
}

int main() {
  auto d = clad::differentiate(off, "k1");
  printf("%.2f\n", d.execute(1.3, 0.8)); // CHECK-EXEC: 0.64

  double H[4] = {0};
  auto h = clad::hessian(off, "k1, k2");
  h.execute(1.3, 0.8, H);
  printf("%.2f %.2f %.2f %.2f\n", H[0], H[1], H[2], H[3]);
  // CHECK-EXEC: 0.00 1.60 1.60 2.60
}
