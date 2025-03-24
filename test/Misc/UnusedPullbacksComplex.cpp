// RUN: %cladclang %s -o %t
// RUN: %t | FileCheck %s

#include <iostream>
#include "clad/Differentiator/Differentiator.h"

double fn2(double x) {
  return x * x;
}

void fn2_pullback(double x, double dy, double* dx); // Forward declaration

double fn1(double x, bool condition) {
  if (condition) {
    return fn2(x);
  } else {
    return x;
  }
}

int main() {
  double grad = 0;
  auto d_test = clad::gradient(fn1);
  d_test.execute(2.0, true, &grad);
  std::cout << grad << std::endl; // CHECK: 4
  return 0;
}

// The pullback for fn2 is only needed when condition is true.
// Clad might not generate it if it only sees calls with condition = false.
// The empty definition ensures that the program links correctly even if the
// pullback is not explicitly used in all execution paths.
