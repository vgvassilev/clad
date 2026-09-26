// RUN: %cladclang %s -I%S/../../include -o%t 2>&1
// RUN: %t | %filecheck_prints %s

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>
#include <cmath>

double test_math(double x, double y) {
  return std::sin(x) * std::cos(y) + std::exp(x * y);
}

int main() {
  auto df_math = clad::differentiate<clad::opts::vector_mode>(test_math, "x,y");
  
  double x = 0.5, y = 1.2;
  double dx = 0, dy = 0;
  df_math.execute(x, y, &dx, &dy);
  
  // expected dx = cos(x)*cos(y) + y * exp(x*y)
  // expected dy = -sin(x)*sin(y) + x * exp(x*y)
  
  double exp_dx = std::cos(x) * std::cos(y) + y * std::exp(x * y);
  double exp_dy = -std::sin(x) * std::sin(y) + x * std::exp(x * y);
  
  // Since we don't have exact precision in printf check, we can just print the diff
  // which should be very close to 0.
  printf("diff_dx=%.4f, diff_dy=%.4f\n", dx - exp_dx, dy - exp_dy);
  // prints: diff_dx=0.0000, diff_dy=0.0000

  return 0;
}
