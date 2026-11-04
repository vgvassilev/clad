// RUN: %cladclang %s -I%S/../../include -oTBRStoredOperand.out
// RUN: ./TBRStoredOperand.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr %s \
// RUN:   -I%S/../../include -oTBRStoredOperand.out
// RUN: ./TBRStoredOperand.out | %filecheck_exec %s

// The reverse sweep differentiates through a stored operand, and the
// operand's non-linear subexpressions read values that are overwritten later,
// so they have to be recorded.

#include "clad/Differentiator/Differentiator.h"

#include <cmath>
#include <cstdio>

// f = 1/(x * x^2) + 3 = x^-3 + 3, f' = -3 x^-4.
double div_straight(double x) {
  double l = x * x;
  double s = 1. / (x * l);
  l = 3.;
  return s + l;
}

// The overwrite is the next iteration (Carlson's R_D has this shape).
// f = sum_{n<3} 1/(x_n * x_n^2), x_{n+1} = x_n / 2, so f' = -3 x^-4 * 73.
double div_loop(double x) {
  double s = 0;
  for (int n = 0; n < 3; n++) {
    double l = x * x;
    s += 1. / (x * l);
    x = 0.5 * x;
  }
  return s;
}

// A factor holding a call is stored. f = 2x (x^3 + sin x) + 3,
// f' = 8x^3 + 2 sin x + 2x cos x.
double mul_stored_left(double x) {
  double l = x * x;
  double s = (x * l + std::sin(x)) * 2.0 * x;
  l = 3.;
  return s + l;
}

// The same with the stored factor on the right. f' = 4x^3 + sin x + x cos x.
double mul_stored_right(double x) {
  double l = x * x;
  double s = x * (x * l + std::sin(x));
  l = 3.;
  return s + l;
}

// A numerator holding a call is stored. f = x^2 + sin(x) / x + 3,
// f' = 2x + cos(x) / x - sin(x) / x^2.
double div_stored_numerator(double x) {
  double l = x * x;
  double s = (x * l + std::sin(x)) / x;
  l = 3.;
  return s + l;
}

#define TEST(F, want)                                                          \
  {                                                                            \
    double d = 0;                                                              \
    clad::gradient(F).execute(x, &d);                                          \
    printf("%s %.9f %.9f\n", #F, d, want);                                     \
  }

int main() {
  double x = 1.3;
  TEST(div_straight, -3. / std::pow(x, 4));
  // CHECK-EXEC: div_straight -1.050383390 -1.050383390
  TEST(div_loop, -3. / std::pow(x, 4) * 73.);
  // CHECK-EXEC: div_loop -76.677987465 -76.677987465
  TEST(mul_stored_left,
       8 * x * x * x + 2 * std::sin(x) + 2 * x * std::cos(x));
  // CHECK-EXEC: mul_stored_left 20.198613325 20.198613325
  TEST(mul_stored_right, 4 * x * x * x + std::sin(x) + x * std::cos(x));
  // CHECK-EXEC: mul_stored_right 10.099306663 10.099306663
  TEST(div_stored_numerator,
       2 * x + std::cos(x) / x - std::sin(x) / (x * x));
  // CHECK-EXEC: div_stored_numerator 2.235615557 2.235615557
}
