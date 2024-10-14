// Tests originally created by Kim Liegeois

#include "TestUtils.h"
#include <Kokkos_Core.hpp>
#include "clad/Differentiator/Differentiator.h"
#include "clad/Differentiator/KokkosBuiltins.h"
#include "gtest/gtest.h"

double f(double x, double y) {

  const int N1 = 4;

  Kokkos::View<double* [4], Kokkos::LayoutLeft, Kokkos::HostSpace> a("a", N1);
  Kokkos::View<double* [4], Kokkos::LayoutLeft, Kokkos::HostSpace> b("b", N1);

  a(0, 0) = x;
  b(1, 1) = y;

  b(1, 1) += a(0, 0) * b(1, 1);

  return a(0, 0) * a(0, 0) * b(1, 1) + b(1, 1);
}

double f_2(double x, double y) {

  const int N1 = 4;

  Kokkos::View<double* [4], Kokkos::LayoutLeft, Kokkos::HostSpace> a("a", N1);
  Kokkos::View<double* [4], Kokkos::LayoutLeft, Kokkos::HostSpace> b("b", N1);

  Kokkos::deep_copy(a, 3 * x + y);
  b(0, 0) = x;
  Kokkos::deep_copy(b, a);

  b(0, 0) += a(0, 0) * b(0, 0);

  return a(0, 0);
}

double f_3(double x, double y) {

  const int N1 = 4;

  Kokkos::View<double* [4], Kokkos::LayoutLeft, Kokkos::HostSpace> a("a", N1);
  Kokkos::View<double* [4], Kokkos::LayoutLeft, Kokkos::HostSpace> b("b", N1);

  Kokkos::deep_copy(a, 3 * x + y);
  b(0, 0) = y;
  Kokkos::deep_copy(b, a);

  b(0, 0) += a(0, 0) * b(0, 0);

  return a(0, 0) + b(0, 0);
}

TEST(ViewAccess, Test1) {
  EXPECT_NEAR(f(0, 1), 1, 1e-8);
  EXPECT_NEAR(f(0, 2), 2, 1e-8);
}

TEST(ViewAccess, Test2) {

  double tolerance = 1e-8;
  double epsilon = 1e-6;

  auto f_x = clad::differentiate(f, "x");

  std::function<double(double)> f_tmp = [](double x) { return f(x, 4.); };
  double dx_f_FD = finite_difference_tangent(f_tmp, 3., epsilon);
  EXPECT_NEAR(f_x.execute(3, 4), dx_f_FD, tolerance * dx_f_FD);

  auto f_2_x = clad::differentiate(f_2, "x");

  std::function<double(double)> f_2_tmp = [](double x) { return f_2(x, 4.); };
  double dx_f_2_FD = finite_difference_tangent(f_2_tmp, 3., epsilon);
  EXPECT_NEAR(f_2_x.execute(3, 4), dx_f_2_FD, tolerance * dx_f_2_FD);

  auto f_3_y = clad::differentiate(f_3, "y");

  std::function<double(double)> f_3_tmp = [](double y) { return f_3(3., y); };
  double dy_f_3_FD = finite_difference_tangent(f_3_tmp, 4., epsilon);
  EXPECT_NEAR(f_3_y.execute(3, 4), dy_f_3_FD, tolerance * dy_f_3_FD);

  auto f_grad_exe = clad::gradient(f);
  double dx = 0, dy = 0;
  f_grad_exe.execute(3., 4., &dx, &dy);
  EXPECT_NEAR(f_x.execute(3, 4), dx, tolerance * dx);

  double dx_2 = 0, dy_2 = 0;
  auto f_2_grad_exe = clad::gradient(f_2);
  f_2_grad_exe.execute(3., 4., &dx_2, &dy_2);
  EXPECT_NEAR(f_2_x.execute(3, 4), dx_2, tolerance * dx_2);

  double dx_3 = 0, dy_3 = 0;
  auto f_3_grad_exe = clad::gradient(f_3);
  f_3_grad_exe.execute(3., 4., &dx_3, &dy_3);
  EXPECT_NEAR(f_3_y.execute(3, 4), dy_3, tolerance * dy_3);
}