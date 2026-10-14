// Pins the Kokkos math pushforwards (cos/sin/sqrt, KokkosBuiltins.h) against a
// finite-difference tangent, forward and reverse.

#include "TestUtils.h"
#include "clad/Differentiator/Differentiator.h"
#include "clad/Differentiator/KokkosBuiltins.h"
#include <Kokkos_Core.hpp>
#include "gtest/gtest.h"

// Exercises all three functions in one expression; x*x + 1 keeps sqrt's
// argument positive and smooth for either sign of x.
double math_fn(double x, double y) {
  return Kokkos::sqrt(x * x + 1.0) * Kokkos::cos(y) + Kokkos::sin(x * y);
}

TEST(KokkosMath, Forward) {
  const double eps = 1e-6, tol = 1e-5;
  auto fn_x = clad::differentiate(math_fn, "x");
  auto fn_y = clad::differentiate(math_fn, "y");
  for (double x : {1.3, -1.3}) {
    std::function<double(double)> gx = [x](double t) {
      return math_fn(t, 0.7);
    };
    std::function<double(double)> gy = [x](double t) { return math_fn(x, t); };
    EXPECT_NEAR(fn_x.execute(x, 0.7), finite_difference_tangent(gx, x, eps),
                tol);
    EXPECT_NEAR(fn_y.execute(x, 0.7), finite_difference_tangent(gy, 0.7, eps),
                tol);
  }
}

TEST(KokkosMath, Reverse) {
  const double eps = 1e-6, tol = 1e-5;
  auto fn_grad = clad::gradient(math_fn);
  for (double x : {1.3, -1.3}) {
    double dx = 0, dy = 0;
    fn_grad.execute(x, 0.7, &dx, &dy);
    std::function<double(double)> gx = [x](double t) {
      return math_fn(t, 0.7);
    };
    std::function<double(double)> gy = [x](double t) { return math_fn(x, t); };
    EXPECT_NEAR(dx, finite_difference_tangent(gx, x, eps), tol);
    EXPECT_NEAR(dy, finite_difference_tangent(gy, 0.7, eps), tol);
  }
}
