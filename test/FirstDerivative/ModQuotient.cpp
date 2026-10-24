// RUN: %cladclang %s -I%S/../../include -oModQuotient.out
// RUN: ./ModQuotient.out | %filecheck_exec %s

// d/dy of fmod(x, y) and remainder(x, y) is -n, with n the quotient each
// function uses: x / y truncated for fmod, rounded to nearest for remainder.

#include "clad/Differentiator/Differentiator.h"
#include <cmath>
#include <cstdio>

double fm(double x, double y) { return std::fmod(x, y); }
double rm(double x, double y) { return std::remainder(x, y); }

int main() {
  // fmod(-5.5, 2): n = -2, not floor(-2.75) = -3.
  printf("%.2f\n", clad::differentiate(fm, 1).execute(-5.5, 2)); // CHECK-EXEC: 2.00
  // remainder(5.5, 2): n = 3, not floor(2.75) = 2.
  printf("%.2f\n", clad::differentiate(rm, 1).execute(5.5, 2)); // CHECK-EXEC: -3.00
  // remainder(5, 2): x / y = 2.5 rounds to the even 2.
  printf("%.2f\n", clad::differentiate(rm, 1).execute(5, 2)); // CHECK-EXEC: -2.00

  double dx = 0, dy = 0;
  clad::gradient(fm).execute(-5.5, 2, &dx, &dy);
  printf("%.2f %.2f\n", dx, dy); // CHECK-EXEC: 1.00 2.00
  dx = dy = 0;
  clad::gradient(rm).execute(5.5, 2, &dx, &dy);
  printf("%.2f %.2f\n", dx, dy); // CHECK-EXEC: 1.00 -3.00
}
