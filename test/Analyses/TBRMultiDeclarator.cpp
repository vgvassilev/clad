// RUN: %cladclang %s -I%S/../../include -oTBRMultiDeclarator.out
// RUN: ./TBRMultiDeclarator.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr %s \
// RUN:   -I%S/../../include -oTBRMultiDeclarator.out
// RUN: ./TBRMultiDeclarator.out | %filecheck_exec %s

// Each iteration's t and u have to be recorded, or the reverse sweep reads the
// last iteration's values in every iteration.

#include "clad/Differentiator/Differentiator.h"

#include <cstdio>

// f = sum_{n<3} x_n^3, x_{n+1} = x_n/2, so f' = 3x^2 (1 + 1/8 + 1/64).
double two_declarators(double x) {
  double s = 0;
  for (int n = 0; n < 3; n++) {
    double t = x * x, u = x;
    s += t * u;
    x = 0.5 * x;
  }
  return s;
}

int main() {
  double x = 1.3, d = 0;
  printf("want %.9f\n", 3 * x * x * (1 + 1. / 8 + 1. / 64));
  // CHECK-EXEC: want 5.782968750
  clad::gradient(two_declarators).execute(x, &d);
  printf("two_declarators %.9f\n", d);
  // CHECK-EXEC: two_declarators 5.782968750
}
