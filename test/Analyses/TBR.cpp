// RUN: %cladclang -mllvm -debug-only=clad-tbr %s -I%S/../../include -oReverseLoops.out 2>&1 | %filecheck %s
// REQUIRES: asserts
//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

double f1(double x) {
  double t = 1;
  for (int i = 0; i < 3; i++)
    t *= x;
  return t;
} // == x^3

#define TEST(F, x) { \
  result[0] = 0; \
  auto F##grad = clad::gradient<clad::opts::enable_tbr>(F);\
  F##grad.execute(x, result);\
  printf("{%.2f}\n", result[0]); \
}

int main() {
  double result[3] = {};
  TEST(f1, 3); // CHECK-EXEC: {27.00}

}
