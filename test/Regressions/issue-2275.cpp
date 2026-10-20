// RUN: %cladclang %s -I%S/../../include -Xclang -verify -o %t
// RUN: %t | %filecheck %s --check-prefix=CHECK-EXEC
// UNSUPPORTED: clang-11, clang-12, clang-13, clang-14, clang-15, clang-16
// expected-no-diagnostics

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

double copyCapture(double x) {
  auto g = [x](double t) { return x * t; };
  return g(x);
}

double referenceCapture(double x) {
  auto g = [&x](double t) { return x * t; };
  return g(x);
}

double conditionCapture(double x) {
  if (double c = x * x) {
    auto g = [c](double t) { return c * t; };
    return g(x);
  }
  return 0;
}

int main() {
  auto dCopy = clad::differentiate(copyCapture, "x");
  auto dReference = clad::differentiate(referenceCapture, "x");
  auto dCondition = clad::differentiate(conditionCapture, "x");
  printf("copy %.2f\n", dCopy.execute(1.3));
  // CHECK-EXEC: copy 2.60
  printf("reference %.2f\n", dReference.execute(1.3));
  // CHECK-EXEC-NEXT: reference 2.60
  printf("condition %.2f %.2f\n", dCondition.execute(1.3),
         dCondition.execute(0));
  // CHECK-EXEC-NEXT: condition 5.07 0.00
}
