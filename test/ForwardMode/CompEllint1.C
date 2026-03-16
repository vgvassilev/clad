// RUN: %cladclang %s -I%S/../../include -oCompEllint1.out | %filecheck %s
// RUN: ./CompEllint1.out | %filecheck_exec %s
// XFAIL: valgrind

#include "clad/Differentiator/Differentiator.h"
#include <cmath>
#include <cstdio>

// CHECK-EXEC: PASS

#if defined(__cpp_lib_math_special_functions)

// CHECK-NOT: warning: falling back to numerical differentiation

double f(double k) { return std::comp_ellint_1(k); }

int main() {
  auto d_f = clad::differentiate(f, 0);
  double k = 0.5;
  double K = std::comp_ellint_1(k);
  double E = std::comp_ellint_2(k);
  double k2 = k * k;
  double expected = (E - (1 - k2) * K) / (k * (1 - k2));
  double result = d_f.execute(k);

  if (std::abs(result - expected) < 1e-10)
    printf("PASS\n");
  else
    printf("FAIL: expected %.10f, got %.10f\n", expected, result);

  return 0;
}

#else

int main() {
  printf("PASS\n");
  return 0;
}

#endif
