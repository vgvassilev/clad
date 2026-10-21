// Forward mode over a function constructing several Kokkos::Views. Kokkos >= 5
// uses a View constructor the fixed-arity constructor_pushforward did not
// match, so the tangent was built with extent 0 and a second View then
// corrupted the host allocator. See KokkosBuiltins.h. Covers rank-1 and rank-2
// runtime Views.

#include "TestUtils.h"
#include "clad/Differentiator/Differentiator.h"
#include "clad/Differentiator/KokkosBuiltins.h"
#include <Kokkos_Core.hpp>
#include "gtest/gtest.h"

#include <functional>

// Two rank-1 Views. f = a(0) * b(1), a(0) = x, b(1) = y. df/dx = y.
double two_views_rank1(double x, double y) {
  Kokkos::View<double*, Kokkos::HostSpace> a("a", 4), b("b", 4);
  a(0) = x;
  b(1) = y;
  return a(0) * b(1);
}

// Two rank-2 runtime Views -- exercises the rank-2 constructor_pushforward.
double two_views_rank2(double x, double y) {
  Kokkos::View<double**, Kokkos::HostSpace> m("m", 3, 4), n("n", 2, 3);
  m(1, 2) = x;
  n(0, 1) = y;
  return m(1, 2) * n(0, 1);
}

// The ViewAccess::f shape: a compound assignment across two Views (rank-1
// runtime; the second static extent does not reach the constructor).
double view_access_shape(double x, double y) {
  const int N1 = 4;
  Kokkos::View<double* [4], Kokkos::LayoutLeft, Kokkos::HostSpace> a("a", N1);
  Kokkos::View<double* [4], Kokkos::LayoutLeft, Kokkos::HostSpace> b("b", N1);
  a(0, 0) = x;
  b(1, 1) = y;
  b(1, 1) += a(0, 0) * b(1, 1);
  return a(0, 0) * a(0, 0) * b(1, 1) + b(1, 1);
}

// clad::differentiate needs the function at the call site, so the checks are
// spelled out rather than routed through a helper taking a function pointer.
#define EXPECT_TANGENT_MATCHES_FD(f)                                           \
  do {                                                                         \
    const double eps = 1e-6, tol = 1e-5;                                       \
    auto d = clad::differentiate(f, "x");                                      \
    std::function<double(double)> fx = [](double x) { return f(x, 4.0); };     \
    EXPECT_NEAR(d.execute(3.0, 4.0), finite_difference_tangent(fx, 3.0, eps),  \
                tol);                                                          \
  } while (0)

TEST(KokkosViewCtorForward, MultipleViews) {
  EXPECT_TANGENT_MATCHES_FD(two_views_rank1);
  EXPECT_TANGENT_MATCHES_FD(two_views_rank2);
  EXPECT_TANGENT_MATCHES_FD(view_access_shape);
}
