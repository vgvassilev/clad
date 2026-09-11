// RUN: %cladclang %s -I%S/../../include -Xclang -verify -oLibCDerivatives.out 2>&1 | %filecheck %s
// RUN: ./LibCDerivatives.out | %filecheck_exec %s

#include "clad/Differentiator/LibCDerivatives.h"
#include "clad/Differentiator/Differentiator.h"
#include "../TestUtils.h"
#include <cmath>

extern "C" int printf(const char* fmt, ...);

double test_sin(double x) {
  return std::sin(x);
}

double test_exp(double x) {
  return std::exp(x);
}

int main() {
  auto df_sin = clad::differentiate(test_sin);
  auto df_exp = clad::differentiate(test_exp);

  double res_sin = df_sin.execute(0.0);
  double res_exp = df_exp.execute(1.0);

  INIT_TEST;
  TEST(res_sin == 1.0);
  TEST(res_exp == std::exp(1.0));
  END_TEST;
  return 0;
}
