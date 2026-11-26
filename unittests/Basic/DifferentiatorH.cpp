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

TEST(DifferentiatorH, DropAndTakeArgsOutOfBounds) {
  using Fn = void (*)(int, double, float);
  static_assert(clad::arity_of<Fn>::value == 3, "");

  // N < arity
  using First2 = clad::TakeNFirstArgs_t<2, Fn>;
  static_assert(std::is_same<First2, clad::list<int, double>>::value, "");
  using Drop2 = clad::DropArgs_t<2, Fn>;
  static_assert(std::is_same<Drop2, clad::list<float>>::value, "");

  // N == arity
  using First3 = clad::TakeNFirstArgs_t<3, Fn>;
  static_assert(std::is_same<First3, clad::list<int, double, float>>::value, "");
  using Drop3 = clad::DropArgs_t<3, Fn>;
  static_assert(std::is_same<Drop3, clad::list<>>::value, "");

  // N > arity: must not fail template instantiation or exceed tuple bounds
  using First5 = clad::TakeNFirstArgs_t<5, Fn>;
  static_assert(std::is_same<First5, clad::list<int, double, float>>::value, "");
  using Drop5 = clad::DropArgs_t<5, Fn>;
  static_assert(std::is_same<Drop5, clad::list<>>::value, "");

  // N == 0
  using First0 = clad::TakeNFirstArgs_t<0, Fn>;
  static_assert(std::is_same<First0, clad::list<>>::value, "");
  using Drop0 = clad::DropArgs_t<0, Fn>;
  static_assert(std::is_same<Drop0, clad::list<int, double, float>>::value, "");
}

