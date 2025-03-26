// RUN: %cladclang %s -fsyntax-only -Xclang -verify 2>&1 | FileCheck %s

#include "clad/Differentiator/Differentiator.h"

double test_fn(double x) {
  return x * x;
}

void unused_pullback(double x, double dy, double* dx);

void test_grad() {
  double grad = 0;
  auto d_test = clad::gradient(test_fn);
  d_test.execute(1.0, &grad);
}

int main() {
  test_grad();
  return 0;
}

// CHECK-NOT: undefined reference to 'unused_pullback'