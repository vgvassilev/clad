#include "clad/Differentiator/ScopeExit.h"

#include "gtest/gtest.h"

#include <memory>
#include <utility>

namespace {

int returnWithGuard(int& Calls) {
  auto Guard = clad_compat::makeScopeExit([&] { ++Calls; });
  return 7;
}

TEST(ScopeExit, RunsOnceAtScopeExit) {
  int Calls = 0;
  {
    auto Guard = clad_compat::makeScopeExit([&] { ++Calls; });
    EXPECT_EQ(Calls, 0);
  }
  EXPECT_EQ(Calls, 1);
}

TEST(ScopeExit, RunsOnEarlyReturn) {
  int Calls = 0;
  EXPECT_EQ(returnWithGuard(Calls), 7);
  EXPECT_EQ(Calls, 1);
}

TEST(ScopeExit, ReleaseDisarmsCleanup) {
  int Calls = 0;
  {
    auto Guard = clad_compat::makeScopeExit([&] { ++Calls; });
    Guard.release();
  }
  EXPECT_EQ(Calls, 0);
}

TEST(ScopeExit, MoveTransfersCleanupOwnership) {
  int Calls = 0;
  {
    auto Original = clad_compat::makeScopeExit([&] { ++Calls; });
    auto Moved = std::move(Original);
    EXPECT_EQ(Calls, 0);
  }
  EXPECT_EQ(Calls, 1);
}

TEST(ScopeExit, OwnsMoveOnlyCallable) {
  int Calls = 0;
  {
    auto Value = std::make_unique<int>(1);
    auto Guard = clad_compat::makeScopeExit(
        [Value = std::move(Value), &Calls] { Calls += *Value; });
    EXPECT_EQ(Value, nullptr);
    EXPECT_EQ(Calls, 0);
  }
  EXPECT_EQ(Calls, 1);
}

} // namespace