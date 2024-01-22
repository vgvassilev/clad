#include <Kokkos_Core.hpp>

#include "gtest/gtest.h"

#include "clad/Differentiator/Differentiator.h"

#include "parallel_sum.hpp"

template <typename T>
T finiteDifferenceTangent(std::function<T(T)> func, const T& x, const T& epsilon) {
  return (func(x+epsilon)-func(x-epsilon)) / (2 * epsilon);
}

double f(double x, double y) {

  const int N1 = 4;

  Kokkos::View<double *[4], Kokkos::LayoutLeft, Kokkos::HostSpace> a("a", N1);
  Kokkos::View<double *[4], Kokkos::LayoutLeft, Kokkos::HostSpace> b("b", N1);

  a(0,0) = x;
  b(0,0) = y;

  b(0,0) += a(0,0) * b(0,0);

  return a(0,0) * a(0,0) * b(0,0) + b(0,0);
}

double f_2(double x, double y) {

  const int N1 = 4;

  Kokkos::View<double *[4], Kokkos::LayoutLeft, Kokkos::HostSpace> a("a", N1);
  Kokkos::View<double *[4], Kokkos::LayoutLeft, Kokkos::HostSpace> b("b", N1);

  Kokkos::deep_copy(a, 3*x+y);
  b(0,0) = x;
  Kokkos::deep_copy(b, a);

  b(0,0) += a(0,0) * b(0,0);

  return a(0,0);
}

TEST(view_access, test_1) {
  EXPECT_NEAR(f(0,1), 1, 1e-8);
  EXPECT_NEAR(f(0,2), 2, 1e-8);
}

TEST(view_access, test_2) {

  double tolerance = 1e-8;
  double epsilon = 1e-6;

  auto f_x = clad::differentiate(f, "x");

  std::function<double(double)> f_tmp = [](double x){ return f(x,4.); };
  double dx_f_FD = finiteDifferenceTangent(f_tmp, 3., epsilon);

  EXPECT_NEAR(f_x.execute(3, 4),dx_f_FD,tolerance*dx_f_FD);

  auto f_2_x = clad::differentiate(f_2, "x");

  std::function<double(double)> f_2_tmp = [](double x){ return f_2(x,4.); };
  double dx_f_2_FD = finiteDifferenceTangent(f_2_tmp, 3., epsilon);
  EXPECT_NEAR(f_2_x.execute(3, 4),dx_f_2_FD,tolerance*dx_f_2_FD);

  auto f_grad_exe = clad::gradient(f);
  double dx, dy;
  f_grad_exe.execute(3., 4., &dx, &dy);
  EXPECT_NEAR(f_x.execute(3, 4),dx,tolerance*dx);

  auto f_2_grad_exe = clad::gradient(f_2);
  //f_2_grad_exe.execute(3., 4., &dx, &dy);
  //EXPECT_NEAR(f_2_x.execute(3, 4),dx,tolerance*dx);
}
