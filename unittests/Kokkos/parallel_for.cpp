#include <Kokkos_Core.hpp>

#include "gtest/gtest.h"

#include "clad/Differentiator/Differentiator.h"

struct hello_world_pow2 {
  double x = 0.;
  // double result = 0.;
  KOKKOS_INLINE_FUNCTION void operator()(const int i) const {
    // result = x * x;
  }
};

TEST(parallel_for, HelloWorldFunctor) {
  hello_world_pow2 hw;
  hw.x = 2;
  Kokkos::parallel_for("HelloWorld", 15, hw);
  // EXPECT_EQ();
  //  FIXME: Add the calls to clad::differentiate/gradient...
}
