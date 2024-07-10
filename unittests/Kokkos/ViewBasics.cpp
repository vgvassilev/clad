// Very basic Kokkos::View usage test that should work by all means
// inspired by
// https://github.com/kliegeois/clad/blob/kokkos-PR/unittests/Kokkos/view_access.cpp
// it has been modified to match gtest guidelines and improve readability

#include "ParallelAdd.h"
#include <Kokkos_Core.hpp>
#include "clad/Differentiator/Differentiator.h"
#include "gtest/gtest.h"

double f_basics(double x, double y) {
  const int N = 2;

  Kokkos::View<double* [N], Kokkos::HostSpace> a("a", N);
  Kokkos::View<double* [N], Kokkos::HostSpace> b("b", N);

  a(0, 0) = x;
  b(0, 0) = y * x;

  return a(0, 0) + a(0, 0) * b(0, 0) + b(0, 0);
}

TEST(ViewBasics, TestAccessForward) {
  // // check finite difference and forward mode similarity
  // const double eps = 1e-5;
  // const double tau = 1e-6; // tolerance

  // // TODO: uncomment this once it has been implemented
  // auto f_x = clad::differentiate(f_basics, "x");
  // for (double y = 3; y <= 5; y += 1) {
  //   std::function<double(double)> f_tmp = [y](double t){ return f_basics(t,
  //   y); }; for (double x = 3; x <= 5; x += 1) {
  //     double f_x_ex = f_x.execute(x, y);
  //     double dx_f_FD = finite_difference_tangent(f_tmp, x, eps);
  //     EXPECT_NEAR(f_x_ex, dx_f_FD, abs(tau*dx_f_FD));
  //   }
  // }
}

TEST(ViewBasics, TestAccessReverse) {
  // // check reverse mode and forward mode similarity
  // const double eps = 1e-5;
  // const double tau = 1e-6; // tolerance

  // // TODO: uncomment this once it has been implemented
  // auto f_grad_exe = clad::gradient(f_basics);
  // for (double y = 3; y <= 5; y += 1) {
  //   std::function<double(double)> f_tmp = [y](double t){ return f_basics(t,
  //   y); }; for (double x = 3; x <= 5; x += 1) {
  //     double dx_f_FD = finite_difference_tangent(f_tmp, x, eps);
  //     double dx, dy;
  //     f_grad_exe.execute(x, y, &dx, &dy);
  //     EXPECT_NEAR(dx_f_FD, dx, abs(tau*dx));
  //   }
  // }
}

double f_basics_deep_copy(double x, double y) {
  const int N = 2;

  Kokkos::View<double* [4], Kokkos::LayoutLeft, Kokkos::HostSpace> a("a", N);
  Kokkos::View<double* [4], Kokkos::LayoutLeft, Kokkos::HostSpace> b("b", N);

  Kokkos::deep_copy(a, 3 * x + y);
  b(0, 0) = x;
  Kokkos::deep_copy(b, a);

  b(0, 0) = b(0, 0) + a(0, 0) * b(0, 0);

  return a(0, 0); // derivative wrt x is constantly 3
}

TEST(ViewBasics, TestDeepCopyForward) {
  // // check finite difference and forward mode similarity
  // const double eps = 1e-5;
  // const double tau = 1e-6; // tolerance

  // // TODO: uncomment this once it has been implemented
  // auto f_x = clad::differentiate(f_basics_deep_copy, "x");
  // for (double y = 3; y <= 5; y += 1) {
  //   std::function<double(double)> f_tmp = [y](double t){ return
  //   f_basics_deep_copy(t, y);
  //   }; for (double x = 3; x <= 5; x += 1) {
  //     double f_x_ex = f_x.execute(x, y);
  //     double dx_f_FD = finite_difference_tangent(f_tmp, x, eps);
  //     EXPECT_NEAR(f_x_ex, dx_f_FD, abs(tau*dx_f_FD));
  //   }
  // }
}

TEST(ViewBasics, TestDeepCopyReverse) {
  // // check reverse mode and forward mode similarity
  // const double eps = 1e-5;
  // const double tau = 1e-6; // tolerance

  // // TODO: uncomment this once it has been implemented
  // auto f_grad_exe = clad::gradient(f_basics_deep_copy);
  // for (double y = 3; y <= 5; y += 1) {
  //   std::function<double(double)> f_tmp = [y](double t){ return
  //   f_basics_deep_copy(t, y);
  //   }; for (double x = 3; x <= 5; x += 1) {
  //     double dx_f_FD = finite_difference_tangent(f_tmp, x, eps);
  //     double dx, dy;
  //     f_grad_exe.execute(x, y, &dx, &dy);
  //     EXPECT_NEAR(dx_f_FD, dx, abs(tau*dx));
  //   }
  // }
}

double f_basics_deep_copy_2(double x, double y) {
  const int N = 2;

  Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace> a("a", N);
  Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace> b("b", N);

  Kokkos::deep_copy(a, 3 * y + x + 50);
  b(1) = x * y;
  Kokkos::deep_copy(b, a);

  b(1) = b(1) + a(0) * b(1);

  a(1) = x * x * x;
  a(0) += a(1);

  return a(0); // derivative of this wrt y is constantly 3
}

TEST(ViewBasics, TestDeepCopy2Forward) {
  // // check finite difference and forward mode similarity
  // const double eps = 1e-5;
  // const double tau = 1e-6; // tolerance

  // // TODO: uncomment this once it has been implemented
  // auto f_y = clad::differentiate(f_basics_deep_copy_2, "y");
  // for (double x = 3; x <= 5; x += 1) {
  //   std::function<double(double)> f_tmp = [x](double t){ return
  //   f_basics_deep_copy_2(x, t);
  //   }; for (double y = 3; y <= 5; y += 1) {
  //     double f_y_ex = f_y.execute(x, y);
  //     double dy_f_FD = finite_difference_tangent(f_tmp, y, eps);
  //     EXPECT_NEAR(f_y_ex, dy_f_FD, abs(tau*dy_f_FD));
  //   }
  // }
}

TEST(ViewBasics, TestDeepCopy2Reverse) {
  // // check reverse mode and forward mode similarity
  // const double eps = 1e-5;
  // const double tau = 1e-6; // tolerance

  // // TODO: uncomment this once it has been implemented
  // auto f_grad_exe = clad::gradient(f_basics_deep_copy_2);
  // for (double x = 3; x <= 5; x += 1) {
  //   std::function<double(double)> f_tmp = [x](double t){ return
  //   f_basics_deep_copy_2(x, t);
  //   }; for (double y = 3; y <= 5; y += 1) {
  //     double dy_f_FD = finite_difference_tangent(f_tmp, y, eps);
  //     double dx, dy;
  //     f_grad_exe.execute(x, y, &dx, &dy);
  //     EXPECT_NEAR(dy_f_FD, dy, abs(tau*dy));
  //   }
  // }
}