#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  int res = RUN_ALL_TESTS();
  Kokkos::finalize();
  return res;
}
