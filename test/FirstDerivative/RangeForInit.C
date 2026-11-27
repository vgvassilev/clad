// RUN: %cladclang -std=c++20 %s -I%S/../../include -oRangeForInit.out
// RUN: ./RangeForInit.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"

#include <cstdio>

double rangeForInit(double x) {
  double result = 0;
  for (double values[] = {x * x, x * x, x * x}; double value : values) {
    result += value;
  }
  return result;
}

int main() {
  auto d = clad::differentiate(rangeForInit, "x");
  std::printf("%.2f\n", d.execute(2));
  // CHECK-EXEC: 12.00

  auto grad = clad::gradient(rangeForInit);
  double dx = 0;
  grad.execute(2, &dx);
  std::printf("%.2f\n", dx);
  // CHECK-EXEC: 12.00
}
