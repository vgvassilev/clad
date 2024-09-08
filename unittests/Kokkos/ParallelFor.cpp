#include <Kokkos_Core.hpp>
#include "clad/Differentiator/Differentiator.h"
#include "clad/Differentiator/KokkosBuiltins.h"
#include "gtest/gtest.h"
// #include "TestUtils.h"

TEST(ParallelFor, HelloWorldLambdaLoopForward) {
  // // check finite difference and forward mode similarity
  // const double eps = 1e-5;
  // const double tau = 1e-6; // tolerance

  std::function<double(double)> _f = [](double x) {
    using Policy = Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>;
    double res[5] = {0};
    Kokkos::parallel_for("HelloWorld-forward", Policy(0, 5),
                         [&res, x](const int i) { res[i] = x * x; });
    // all elements of res should be the same, so return one of them arbitrarily
    return res[2];
  };

  // // TODO: uncomment this once it has been implemented
  // auto f_diff = clad::differentiate(_f, 0/*x*/);
  // for (double x = -2; x <= 2; x += 1) {
  //   double f_diff_ex = f_diff.execute(x);
  //   double dx_f_FD = finite_difference_tangent(_f, x, eps);
  //   EXPECT_NEAR(f_diff_ex, dx_f_FD, abs(tau*dx_f_FD));
  // }
}

TEST(ParallelFor, HelloWorldLambdaLoopReverse) {
  // // check finite difference and reverse mode similarity
  // const double eps = 1e-5;
  // const double tau = 1e-6; // tolerance

  std::function<double(double)> _f = [](double x) {
    using Policy = Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>;
    double res[5] = {0};
    Kokkos::parallel_for("HelloWorld-reverse", Policy(0, 5),
                         [&res, x](const int i) { res[i] = x * x; });
    // all elements of res should be the same, so return one of them arbitrarily
    return res[2];
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

double parallel_polynomial_for(double x) { // data races
  using Policy = Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>;
  Kokkos::View<double[1], Kokkos::HostSpace> res("res");
  res(0) = 0;
  Kokkos::parallel_for(
      "polycalc", Policy(0, 5),
      KOKKOS_LAMBDA(const int i) { res(0) += pow(x, i + 1) / (i + 1); });
  // for (int i = 0; i < 6; ++i) {
  //   res(0) += pow(x, i+1)/(i+1);
  // }
  return res(0);
}

TEST(ParallelFor, ParallelPolynomialForward) {
  // // check true derivative and forward mode similarity
  // const double tau = 1e-5; // tolerance

  // // TODO: uncomment this once it has been implemented
  // auto f_diff = clad::differentiate(parallel_polynomial_for, "x");
  // for (double x = -2; x <= 2; x += 1) {
  //   double f_diff_ex = f_diff.execute(x);
  //   double dx_f_true = parallel_polynomial_true_derivative(x);
  //   EXPECT_NEAR(f_diff_ex, dx_f_true, abs(tau*dx_f_true));
  // }
}

TEST(ParallelFor, ParallelPolynomialReverse) {
  // // check true derivative and reverse mode similarity
  // const double tau = 1e-5; // tolerance

  // // TODO: uncomment this once it has been implemented
  // auto f_grad = clad::gradient(parallel_polynomial_for);
  // for (double x = -2; x <= 2; x += 1) {
  //   double dx_f_true = parallel_polynomial_true_derivative(x);
  //   double dx = 0;
  //   f_grad.execute(x, &dx);
  //   EXPECT_NEAR(dx_f_true, dx, abs(tau*dx));
  // }
}

template <typename View> struct Foo {
  View& res;
  double& x;

  Foo(View& _res, double& _x) : res(_res), x(_x) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i) const { res(i) = x * i; }
};

double parallel_for_functor_simplest_case_fence(double x) {
  Kokkos::View<double[5], Kokkos::HostSpace> res("res");

  // Kokkos::fence("named fence"); // Does not work on some versions of Kokkos.

  Foo<Kokkos::View<double[5], Kokkos::HostSpace>> f(res, x);

  f(0); // FIXME: this is a workaround to put Foo::operator() into the
        // differentiation plan. This needs to be solved in clad.

  Kokkos::parallel_for(5, f);
  Kokkos::fence();

  return res(3);
}

double parallel_for_functor_simplest_case_intpol(double x) {
  Kokkos::View<double[5], Kokkos::HostSpace> res("res");

  Foo<Kokkos::View<double[5], Kokkos::HostSpace>> f(res, x);

  f(0); // FIXME: this is a workaround to put Foo::operator() into the
        // differentiation plan. This needs to be solved in clad.

  Kokkos::parallel_for("polynomial", 5, f);
  Kokkos::parallel_for(5, f);

  return res(3);
}

double parallel_for_functor_simplest_case_rangepol(double x) {
  Kokkos::View<double[5], Kokkos::HostSpace> res("res");

  Foo<Kokkos::View<double[5], Kokkos::HostSpace>> f(res, x);

  f(0); // FIXME: this is a workaround to put Foo::operator() into the
        // differentiation plan. This needs to be solved in clad.

  Kokkos::parallel_for(
      "polynomial",
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(1, 5), f);
  // Overwrite with another parallel_for (not named)
  Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(1, 5), f);

  return res(3);
}

template <typename View> struct Foo2 {
  View& res;
  double& x;

  Foo2(View& _res, double& _x) : res(_res), x(_x) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, const int j) const { res(i, j) = x * i * j; }
};

double parallel_for_functor_simplest_case_mdpol(double x) {
  Kokkos::View<double[5][5], Kokkos::HostSpace> res("res");

  Foo2<Kokkos::View<double[5][5], Kokkos::HostSpace>> f(res, x);

  f(0, 0); // FIXME: this is a workaround to put Foo::operator() into the
           // differentiation plan. This needs to be solved in clad.

  Kokkos::parallel_for(
      "polynomial",
      Kokkos::MDRangePolicy<
          Kokkos::Rank<2, Kokkos::Iterate::Right, Kokkos::Iterate::Left>>(
          {1, 1}, {5, 5}, {1, 1}),
      f);

  return res(3, 4);
}

double parallel_for_functor_simplest_case_mdpol_space_and_anon(double x) {
  Kokkos::View<double[5][5], Kokkos::HostSpace> res("res");

  Foo2<Kokkos::View<double[5][5], Kokkos::HostSpace>> f(res, x);

  f(0, 0); // FIXME: this is a workaround to put Foo::operator() into the
           // differentiation plan. This needs to be solved in clad.

  Kokkos::parallel_for(
      "polynomial",
      Kokkos::MDRangePolicy<
          Kokkos::DefaultHostExecutionSpace,
          Kokkos::Rank<2, Kokkos::Iterate::Right, Kokkos::Iterate::Left>>(
          {1, 1}, {5, 5}, {1, 1}),
      f);
  // Overwrite with another parallel_for (not named)
  Kokkos::parallel_for(
      Kokkos::MDRangePolicy<
          Kokkos::DefaultHostExecutionSpace,
          Kokkos::Rank<2, Kokkos::Iterate::Right, Kokkos::Iterate::Left>>(
          {1, 1}, {5, 5}, {1, 1}),
      f);

  return res(3, 4);
}

TEST(ParallelFor, FunctorSimplestCases) {
  const double eps = 1e-8;

  auto df0 = clad::differentiate(parallel_for_functor_simplest_case_fence, 0);
  for (double x = 3; x <= 5; x += 1)
    EXPECT_NEAR(df0.execute(x), 3, eps);

  auto df1 = clad::differentiate(parallel_for_functor_simplest_case_intpol, 0);
  for (double x = 3; x <= 5; x += 1)
    EXPECT_NEAR(df1.execute(x), 3, eps);

  auto df2 =
      clad::differentiate(parallel_for_functor_simplest_case_rangepol, 0);
  for (double x = 3; x <= 5; x += 1)
    EXPECT_NEAR(df2.execute(x), 3, eps);

  auto df3 = clad::differentiate(parallel_for_functor_simplest_case_mdpol, 0);
  for (double x = 3; x <= 5; x += 1)
    EXPECT_NEAR(df3.execute(x), 12, eps);

  auto df4 = clad::differentiate(
      parallel_for_functor_simplest_case_mdpol_space_and_anon, 0);
  for (double x = 3; x <= 5; x += 1)
    EXPECT_NEAR(df4.execute(x), 12, eps);
}