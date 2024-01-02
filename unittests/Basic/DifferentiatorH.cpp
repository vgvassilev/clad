#include "clad/Differentiator/Differentiator.h"

#include "gtest/gtest.h"

TEST(DifferentiatorH, GetLength) {
  EXPECT_TRUE(clad::GetLength("") == 0);
  EXPECT_TRUE(clad::GetLength("abc") == 3);
}
