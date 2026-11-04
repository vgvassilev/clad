// RUN: %cladclang -std=c++17 %s -I%S/../../include -oCylBesselDerivatives.out
// RUN: ./CylBesselDerivatives.out | %filecheck_exec %s
// CHECK-EXEC: Bessel derivative regression: {{PASS|SKIP}}

#include "clad/Differentiator/Differentiator.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>

#ifdef __cpp_lib_math_special_functions
double bessel_j(double nu, double x) { return std::cyl_bessel_j(nu, x); }
double bessel_i(double nu, double x) { return std::cyl_bessel_i(nu, x); }
double bessel_k(double nu, double x) { return std::cyl_bessel_k(nu, x); }

double bessel_j_pair(double x, double y) {
  return 2.0 * std::cyl_bessel_j(0.0, x) - 3.0 * std::cyl_bessel_j(0.5, y);
}
double bessel_i_pair(double x, double y) {
  return 2.0 * std::cyl_bessel_i(0.0, x) - 3.0 * std::cyl_bessel_i(0.5, y);
}
double bessel_k_pair(double x, double y) {
  return 2.0 * std::cyl_bessel_k(0.0, x) - 3.0 * std::cyl_bessel_k(0.5, y);
}

void check_close(double actual, double expected) {
  if (!std::isfinite(actual) || !std::isfinite(expected) ||
      std::abs(actual - expected) > 1e-7 * (1.0 + std::abs(expected))) {
    std::fprintf(stderr, "Bessel mismatch: actual=%.17g expected=%.17g\n",
                 actual, expected);
    std::abort();
  }
}

void check_positive_infinity(double actual) {
  if (!std::isinf(actual) || actual < 0.0)
    std::abort();
}

double central_difference(double (*f)(double, double), double nu, double x) {
  const double h = 1e-5;
  return (f(nu, x + h) - f(nu, x - h)) / (2.0 * h);
}
#endif

int main() {
#ifdef __cpp_lib_math_special_functions
#define CHECK_FAMILY(F)                                                        \
  {                                                                            \
    auto forward = clad::differentiate(bessel_##F, "x");                         \
    auto reverse = clad::gradient(bessel_##F, "x");                              \
    for (double nu : {0.0, 0.5, 1.0, 2.5}) {                                    \
      for (double x : {0.5, 1.0, 1.7, 4.0}) {                                   \
        const double expected = central_difference(bessel_##F, nu, x);           \
        check_close(forward.execute(nu, x), expected);                           \
        double dx = 0.0;                                                        \
        reverse.execute(nu, x, &dx);                                             \
        check_close(dx, expected);                                               \
      }                                                                         \
    }                                                                           \
    auto vector = clad::differentiate<clad::opts::vector_mode>(bessel_##F##_pair); \
    auto pair_reverse = clad::gradient(bessel_##F##_pair);                        \
    double vx = 0.0, vy = 0.0, rx = 0.0, ry = 0.0;                                \
    vector.execute(1.7, 1.3, &vx, &vy);                                           \
    pair_reverse.execute(1.7, 1.3, &rx, &ry);                                     \
    const double ex = 2.0 * central_difference(bessel_##F, 0.0, 1.7);             \
    const double ey = -3.0 * central_difference(bessel_##F, 0.5, 1.3);             \
    check_close(vx, ex);                                                         \
    check_close(vy, ey);                                                         \
    check_close(rx, ex);                                                         \
    check_close(ry, ey);                                                         \
  }
  CHECK_FAMILY(j)
  CHECK_FAMILY(i)
  CHECK_FAMILY(k)
#undef CHECK_FAMILY

#define CHECK_ORIGIN(F)                                                         \
  {                                                                            \
    auto forward = clad::differentiate(bessel_##F, "x");                         \
    auto reverse = clad::gradient(bessel_##F, "x");                              \
    for (double nu : {0.0, 1.0, 1.5, 2.5}) {                                    \
      const double expected = nu == 1.0 ? 0.5 : 0.0;                             \
      check_close(forward.execute(nu, 0.0), expected);                            \
      double dx = 0.0;                                                          \
      reverse.execute(nu, 0.0, &dx);                                             \
      check_close(dx, expected);                                                 \
    }                                                                           \
    check_positive_infinity(forward.execute(0.5, 0.0));                           \
    double dx = 0.0;                                                            \
    reverse.execute(0.5, 0.0, &dx);                                               \
    check_positive_infinity(dx);                                                 \
  }
  CHECK_ORIGIN(j)
  CHECK_ORIGIN(i)
#undef CHECK_ORIGIN
  auto second_j = clad::differentiate<2>(bessel_j, "x");
  auto second_i = clad::differentiate<2>(bessel_i, "x");
  check_close(second_j.execute(0.0, 0.0), -0.5);
  check_close(second_i.execute(0.0, 0.0), 0.5);
  check_close(second_j.execute(1.0, 0.0), 0.0);
  check_close(second_i.execute(1.0, 0.0), 0.0);
  check_close(second_j.execute(2.0, 0.0), 0.25);
  check_close(second_i.execute(2.0, 0.0), 0.25);
  std::puts("Bessel derivative regression: PASS");
#else
  std::puts("Bessel derivative regression: SKIP (special functions unavailable)");
#endif
}
