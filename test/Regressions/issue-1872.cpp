// RUN: %cladclang -D_USE_MATH_DEFINES -std=c++17 -I%S/../../include %s -o %t 2>&1
// RUN: %t | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"
#include <algorithm>
#include <cstdio>

double use_max(double x) { return std::max(0.5, x * x); }
double use_min(double x) { return std::min(0.5, x * x); }
double use_max_cmp(double x) {
  return std::max(0.5, x * x, std::less<double>{});
}
double use_min_cmp(double x) {
  return std::min(0.5, x * x, std::less<double>{});
}

int main() {
  double result = 0;
  auto h_max = clad::hessian(use_max, "x");
  h_max.execute(1.0, &result);
  printf("%.2f\n", result); // CHECK-EXEC: 2.00

  result = 0;
  auto h_min = clad::hessian(use_min, "x");
  h_min.execute(0.25, &result);
  printf("%.2f\n", result); // CHECK-EXEC: 2.00

  result = 0;
  auto h_max_cmp = clad::hessian(use_max_cmp, "x");
  h_max_cmp.execute(1.0, &result);
  printf("%.2f\n", result); // CHECK-EXEC: 2.00

  result = 0;
  auto h_min_cmp = clad::hessian(use_min_cmp, "x");
  h_min_cmp.execute(0.25, &result);
  printf("%.2f\n", result); // CHECK-EXEC: 2.00
}
