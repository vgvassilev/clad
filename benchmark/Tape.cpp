#include "benchmark/benchmark.h"

#include "clad/Differentiator/Differentiator.h"

#include <cstddef>

// A tape read back by index, the way a reverse sweep reads a loop's record
// once the record is indexed by the loop counter rather than popped. The
// sizes cross the inline buffer and then several slabs, so the cost of
// reaching an element in a slab far from either end is what is measured.

static void fill(clad::tape<double>& t, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i)
    clad::push(t, static_cast<double>(i));
}

// Backwards, from the last element to the first.
static void BM_TapeIndexedReverse(benchmark::State& state) {
  const std::size_t n = state.range(0);
  clad::tape<double> t = {};
  fill(t, n);
  for (auto _ : state) {
    double s = 0;
    for (std::size_t i = n; i-- > 0;)
      s += t[i];
    benchmark::DoNotOptimize(s);
  }
  state.SetItemsProcessed(state.iterations() * n);
}
BENCHMARK(BM_TapeIndexedReverse)->RangeMultiplier(8)->Range(1 << 12, 1 << 18);

// Forwards, from the first element to the last.
static void BM_TapeIndexedForward(benchmark::State& state) {
  const std::size_t n = state.range(0);
  clad::tape<double> t = {};
  fill(t, n);
  for (auto _ : state) {
    double s = 0;
    for (std::size_t i = 0; i < n; ++i)
      s += t[i];
    benchmark::DoNotOptimize(s);
  }
  state.SetItemsProcessed(state.iterations() * n);
}
BENCHMARK(BM_TapeIndexedForward)->RangeMultiplier(8)->Range(1 << 12, 1 << 18);

// Through the iterator, which reads by index underneath.
static void BM_TapeIterate(benchmark::State& state) {
  const std::size_t n = state.range(0);
  clad::tape<double> t = {};
  fill(t, n);
  for (auto _ : state) {
    double s = 0;
    for (double v : t)
      s += v;
    benchmark::DoNotOptimize(s);
  }
  state.SetItemsProcessed(state.iterations() * n);
}
BENCHMARK(BM_TapeIterate)->RangeMultiplier(8)->Range(1 << 12, 1 << 18);

// Push then pop every element: the access pattern the generated code uses
// today, as the reference the indexed reads are held against.
static void BM_TapePushPop(benchmark::State& state) {
  const std::size_t n = state.range(0);
  for (auto _ : state) {
    clad::tape<double> t = {};
    fill(t, n);
    double s = 0;
    for (std::size_t i = 0; i < n; ++i)
      s += clad::pop(t);
    benchmark::DoNotOptimize(s);
  }
  state.SetItemsProcessed(state.iterations() * n);
}
BENCHMARK(BM_TapePushPop)->RangeMultiplier(8)->Range(1 << 12, 1 << 18);

BENCHMARK_MAIN();
