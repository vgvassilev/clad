#include "clad/Differentiator/Differentiator.h"

#include "gtest/gtest.h"

#include <cstddef>

// A tape small enough for a handful of pushes to cross several slabs: four
// elements inline, four per slab.
using SmallTape = clad::tape<int, 4, 4>;
using SmallTapeMT = clad::tape<int, 4, 4, /*is_multithread=*/true>;

template <typename Tape> static void fill(Tape& t, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i)
    clad::push(t, static_cast<int>(i));
}

TEST(Tape, IndexAcrossSlabs) {
  SmallTape t = {};
  fill(t, 23); // inline 0..3, slabs 4..7, 8..11, 12..15, 16..19, 20..22
  for (std::size_t i = 0; i < 23; ++i)
    EXPECT_EQ(t[i], static_cast<int>(i)) << "at " << i;
  // The same, read backwards, the way a reverse sweep reads a record.
  for (std::size_t i = 23; i-- > 0;)
    EXPECT_EQ(t[i], static_cast<int>(i)) << "at " << i;
  // And out of order, so that the lookup is reached from every side.
  const std::size_t order[] = {22, 4, 13, 12, 21, 5, 0, 17, 8, 19, 3, 11};
  for (std::size_t i : order)
    EXPECT_EQ(t[i], static_cast<int>(i)) << "at " << i;
}

// The const subscript reaches the same lookup as the non-const one, cursor
// included; there is no separate walk for it to get wrong.
TEST(Tape, IndexThroughConst) {
  SmallTape t = {};
  fill(t, 23);
  const SmallTape& c = t;
  for (std::size_t i = 23; i-- > 0;)
    EXPECT_EQ(c[i], static_cast<int>(i)) << "at " << i;
}

TEST(Tape, IndexAfterPop) {
  SmallTape t = {};
  fill(t, 23);
  EXPECT_EQ(t[22], 22); // the lookup remembers the last slab
  // Pop down into the third slab: the last two slabs are freed.
  for (int i = 0; i < 12; ++i)
    clad::pop(t);
  EXPECT_EQ(t.size(), 11u);
  for (std::size_t i = 11; i-- > 0;)
    EXPECT_EQ(t[i], static_cast<int>(i)) << "at " << i;
  // Grow again past where the freed slabs were.
  for (int i = 11; i < 30; ++i)
    clad::push(t, i);
  for (std::size_t i = 0; i < 30; ++i)
    EXPECT_EQ(t[i], static_cast<int>(i)) << "at " << i;
}

TEST(Tape, IteratorFollowsPushOrder) {
  SmallTape t = {};
  fill(t, 23);
  int expected = 0;
  for (int v : t)
    EXPECT_EQ(v, expected++);
  EXPECT_EQ(expected, 23);
}

TEST(Tape, MultithreadedTapeIndexes) {
  // The multithreaded instantiation keeps no cursor; the lookup must still
  // reach every slab from an end.
  SmallTapeMT t = {};
  fill(t, 23);
  for (std::size_t i = 23; i-- > 0;)
    EXPECT_EQ(t[i], static_cast<int>(i)) << "at " << i;
}

TEST(Tape, DefaultSizesAcrossSlabs) {
  clad::tape<double> t = {};
  const std::size_t n = 64 + 3 * 1024 + 5;
  for (std::size_t i = 0; i < n; ++i)
    clad::push(t, static_cast<double>(i));
  for (std::size_t i = n; i-- > 0;)
    EXPECT_EQ(t[i], static_cast<double>(i)) << "at " << i;
  EXPECT_EQ(t[64], 64.);
  EXPECT_EQ(t[64 + 1024], 64. + 1024);
  EXPECT_EQ(t[n - 1], static_cast<double>(n - 1));
}

TEST(Tape, NoMutexUnlessMultithreaded) {
  // The lock lives in the multithreaded instantiation only.
  EXPECT_LT(sizeof(clad::tape<double>),
            sizeof(clad::tape<double, 64, 1024, /*is_multithread=*/true>));
}
