#include <Kokkos_Core.hpp>
#include "clad/Differentiator/Differentiator.h"
#include "gtest/gtest.h"
#include "TestUtils.hpp"
#include "parallel_sum.hpp"

TEST(ParallelFor, HelloWorldLambdaLoopForward) {
  // check finite difference and forward mode similarity

  // const double eps = 1e-5;
  // const double tau = 1e-6; // tolerance

  std::function<double(double)> _f = [](double x) {
    double res = 0.;
    Kokkos::parallel_for("HelloWorld-1", 5, [&res, x](const int i){ res = x*x; });
    // res = x*x;
    return res;
  };

  // TODO: uncomment this once it has been implemented
  // auto f_diff = clad::differentiate(_f, 0/*x*/);
  // for (double x = -2; x <= 2; x += 1) {
  //   double f_diff_ex = f_diff.execute(x);
  //   double dx_f_FD = finite_difference_tangent(_f, x, eps);
  //   EXPECT_NEAR(f_diff_ex, dx_f_FD, abs(tau*dx_f_FD));  
  // }
}

TEST(ParallelFor, HelloWorldLambdaLoopReverse) {
  // check finite difference and reverse mode similarity

  // const double eps = 1e-5;
  // const double tau = 1e-6; // tolerance

  std::function<double(double)> _f = [](double x) {
    double res = 0.;
    Kokkos::parallel_for("HelloWorld-2", 5, [&res, x](const int i){ res = x*x; });
    // res = x*x;
    return res;
  };

  // TODO: uncomment this once it has been implemented
  // auto f_grad = clad::gradient(_f);
  // for (double x = -2; x <= 2; x += 1) {
  //   double dx_f_FD = finite_difference_tangent(_f, x, eps);
  //   double dx;
  //   f_grad.execute(x, &dx);
  //   EXPECT_NEAR(dx_f_FD, dx, abs(tau*dx));
  // }
}

double parallel_polynomial(double x) {
  Kokkos::View <double[1]> res("res");
  res(0) = 0;
  Kokkos::parallel_for("polycalc", 5, KOKKOS_LAMBDA(const int i) {
      res(0) += pow(x, i+1)/(i+1);
  });
  // for (int i = 0; i < 6; ++i) { // standard C++ alternative
  //   res(0) += pow(x, i+1)/(i+1);
  // }
  return res(0);
}

double parallel_polynomial_true_derivative(double x) {
  double res = 0;
  double x_c = 1;
  for (unsigned i = 0; i < 6; ++i) {
      res += x_c;
      x_c *= x;
  }
  return res;
}

TEST(ParallelFor, ParallelPolynomialForward) {
  // check true derivative and forward mode similarity

  // const double tau = 1e-5; // tolerance

  // TODO: uncomment this once it has been implemented
  // auto f_diff = clad::differentiate(parallel_polynomial, "x");
  // for (double x = -2; x <= 2; x += 1) {
  //   double f_diff_ex = f_diff.execute(x);
  //   double dx_f_true = parallel_polynomial_true_derivative(x);
  //   EXPECT_NEAR(f_diff_ex, dx_f_true, abs(tau*dx_f_true)); 
  // }
}

TEST(ParallelFor, ParallelPolynomialReverse) {
  // check true derivative and reverse mode similarity
  
  // const double tau = 1e-5; // tolerance

  // TODO: uncomment this once it has been implemented
  // auto f_grad = clad::gradient(parallel_polynomial);
  // for (double x = -2; x <= 2; x += 1) {
  //   double dx_f_true = parallel_polynomial_true_derivative(x);
  //   double dx = 0;
  //   f_grad.execute(x, &dx);
  //   EXPECT_NEAR(dx_f_true, dx, abs(tau*dx));
  // }
}