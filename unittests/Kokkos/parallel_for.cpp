#include <Kokkos_Core.hpp>

#include "gtest/gtest.h"

#include "clad/Differentiator/Differentiator.h"

#include "parallel_sum.hpp"

struct hello_world_pow2 {
  double x = 0.;
  // double result = 0.;
  KOKKOS_INLINE_FUNCTION void operator()(const int i) const {
    // result = x * x;
  }
};

template <typename ViewtypeX>
typename ViewtypeX::value_type f_multilevel(ViewtypeX x) {
  typename ViewtypeX::value_type sum;

  ViewtypeX y("y", x.extent(0));

  Kokkos::parallel_for( x.extent(0), KOKKOS_LAMBDA ( const size_t j0) {
    x(j0) = 3*x(j0);
  });

  Kokkos::parallel_for( x.extent(0)-1, KOKKOS_LAMBDA ( const size_t j1) {
    if (j1 != x.extent(0)-1)
      y(j1+1) = 2.6*x(j1)*x(j1);
    else
      y(j1) = 2.6*x(0)*x(0);
  });

  const int n_max = 10;
  const int n = x.extent(0) > n_max ? n_max : x.extent(0);

  auto y_n_rows = Kokkos::subview( y, Kokkos::make_pair(0, n));
  kokkos_builtin_derivative::parallel_sum(sum, y_n_rows);
  return sum;
}

TEST(parallel_for, HelloWorldFunctor) {
  hello_world_pow2 hw;
  hw.x = 2;
  Kokkos::parallel_for("HelloWorld", 15, hw);
  // EXPECT_EQ();
  //  FIXME: Add the calls to clad::differentiate/gradient...
}

TEST(parallel_for, multilevelG) {
  //auto f_multilevel_grad_exe = clad::gradient(f_multilevel<Kokkos::View<double *>>);
}