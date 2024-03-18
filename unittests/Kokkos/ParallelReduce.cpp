#include <Kokkos_Core.hpp>
#include "clad/Differentiator/Differentiator.h"
#include "gtest/gtest.h"
// #include "TestUtils.h"
#include "ParallelAdd.h"

TEST(ParallelReduce, HelloWorldLambdaLoopForward) {
  // // check finite difference and forward mode similarity
  // const double eps = 1e-5;
  // const double tau = 1e-6; // tolerance

  std::function<double(double)> _f = [](double x) {
    double res = 0.;
    Kokkos::parallel_reduce(
        "HelloWorld-forward", 5,
        KOKKOS_LAMBDA(const int& i, double& _res) { _res += x; }, res);
    // res = 5*x;
    return res;
  };

  // // TODO: uncomment this once it has been implemented
  // auto f_diff = clad::differentiate(_f, 0/*x*/);
  // for (double x = -2; x <= 2; x += 1) {
  //   double f_diff_ex = f_diff.execute(x);
  //   double dx_f_FD = finite_difference_tangent(_f, x, eps);
  //   EXPECT_NEAR(f_diff_ex, dx_f_FD, abs(tau*dx_f_FD));
  // }
}

TEST(ParallelReduce, HelloWorldLambdaLoopReverse) {
  // // check finite difference and reverse mode similarity
  // const double eps = 1e-5;
  // const double tau = 1e-6; // tolerance

  std::function<double(double)> _f = [](double x) {
    double res = 0.;
    Kokkos::parallel_reduce(
        "HelloWorld-reverse", 5,
        KOKKOS_LAMBDA(const int& i, double& _res) { _res += x; }, res);
    // res = 5*x;
    return res;
  };

  // // TODO: uncomment this once it has been implemented
  // auto f_grad = clad::gradient(_f);
  // for (double x = -2; x <= 2; x += 1) {
  //   double dx_f_FD = finite_difference_tangent(_f, x, eps);
  //   double dx;
  //   f_grad.execute(x, &dx);
  //   EXPECT_NEAR(dx_f_FD, dx, abs(tau*dx));
  // }
}

double parallel_polynomial_reduce(double x) {
  using Policy = Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>;
  Kokkos::View<double[1], Kokkos::HostSpace> res("res");
  res(0) = 0;
  Kokkos::parallel_reduce(
      "polycalc", Policy(0, 5),
      KOKKOS_LAMBDA(const int& i, double& _res) {
        _res += pow(x, i + 1) / (i + 1);
      },
      res(0));
  // for (int i = 0; i < 6; ++i) {
  //   res(0) += pow(x, i+1)/(i+1);
  // }
  return res(0);
}

TEST(ParallelReduce, ParallelPolynomialForward) {
  // // check true derivative and forward mode similarity
  // const double tau = 1e-5; // tolerance

  // // TODO: uncomment this once it has been implemented
  // auto f_diff = clad::differentiate(parallel_polynomial_reduce, "x");
  // for (double x = -2; x <= 2; x += 1) {
  //   double f_diff_ex = f_diff.execute(x);
  //   double dx_f_true = parallel_polynomial_true_derivative(x);
  //   EXPECT_NEAR(f_diff_ex, dx_f_true, abs(tau*dx_f_true));
  // }
}

TEST(ParallelReduce, ParallelPolynomialReverse) {
  // // check true derivative and reverse mode similarity
  // const double tau = 1e-5; // tolerance

  // // TODO: uncomment this once it has been implemented
  // auto f_grad = clad::gradient(parallel_polynomial_reduce);
  // for (double x = -2; x <= 2; x += 1) {
  //   double dx_f_true = parallel_polynomial_true_derivative(x);
  //   double dx = 0;
  //   f_grad.execute(x, &dx);
  //   EXPECT_NEAR(dx_f_true, dx, abs(tau*dx));
  // }
}