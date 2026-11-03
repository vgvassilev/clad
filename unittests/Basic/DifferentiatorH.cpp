#include "clad/Differentiator/Differentiator.h"

#include "gtest/gtest.h"

#include <utility>

TEST(DifferentiatorH, GetLength) {
  EXPECT_TRUE(clad::GetLength("") == 0);
  EXPECT_TRUE(clad::GetLength("abc") == 3);
}

// A restore_tracker that holds more records than fit in its inline storage
// keeps them on the heap. A copy of it, and a move of that copy, must still
// restore every value.
TEST(DifferentiatorH, RestoreTrackerCopiesGrownRecords) {
  static double pad[256];
  double a = 1;
  double b = 2;
  clad::restore_tracker t;
  for (int i = 0; i < 256; i += 2)
    t.store(pad[i]);
  t.store(a);
  t.store(b);
  a = 7;
  b = 8;
  clad::restore_tracker copy = t;
  clad::restore_tracker moved = std::move(copy);
  moved.restore();
  EXPECT_EQ(a, 1);
  EXPECT_EQ(b, 2);

  // A tracker that fits in its inline storage is moved by copying the
  // records.
  double c = 3;
  clad::restore_tracker small;
  small.store(c);
  c = 9;
  clad::restore_tracker smallMoved = std::move(small);
  smallMoved.restore();
  EXPECT_EQ(c, 3);
}
