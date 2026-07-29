// RUN: %cladclang %s -I%S/../../include -o %t 2>&1 | %filecheck %s
// RUN: %t | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"
#include "clad/Differentiator/STLBuiltins.h"

#include <cstdio>
#include <vector>

double replace_and_square(std::vector<double> values, double x) {
  values[0] = x;
  return values[0] * values[0];
}

// CHECK: void replace_and_square_grad_1(std::vector<double> values, double x, double *_d_x) {
// CHECK-NEXT:     std::vector<double> _d_values(values);
// CHECK-NEXT:     clad::zero_init(_d_values);

int main() {
  std::vector<double> values{10.0};
  double d_x = 0.0;
  auto gradient = clad::gradient(replace_and_square, "x");
  gradient.execute(values, 3.0, &d_x);
  std::printf("%.1f\n", d_x);
}

// CHECK-EXEC: 6.0
