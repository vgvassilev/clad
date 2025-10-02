#include "benchmark/benchmark.h"

#include "clad/Differentiator/Differentiator.h"

#include <cstddef>
#include <cstdio>
#include <thread>
#include <vector>

// Benchmarking overhead of locking in a single threaded environment
template <bool ThreadSafe>
static void BM_TapeLockOverhead(benchmark::State& state) {
  int block = state.range(0);

  for (auto _ : state) {
    clad::tape<double, 64, 1024, ThreadSafe> t;
    for (int i = 0; i < block; i++)
      clad::push(t, 1.0);
    for (int i = 0; i < block; i++)
      clad::pop(t);
  }
}

// Without locking mechanism
BENCHMARK_TEMPLATE(BM_TapeLockOverhead, false)
    ->RangeMultiplier(2)
    ->Range(0, 4096)
    ->Name("BM_TapeLockOverhead_NoLock");

// With locking mechanism
BENCHMARK_TEMPLATE(BM_TapeLockOverhead, true)
    ->RangeMultiplier(2)
    ->Range(0, 4096)
    ->Name("BM_TapeLockOverhead_Lock");

template <typename T>
void concurrent_push(T x, size_t n_threads, size_t pushes_per_thread) {
  clad::tape<T, 64, 1024, true> t = {};
  std::vector<std::thread> threads;

  for (size_t i = 0; i < n_threads; ++i) {
    threads.emplace_back([&]() {
      for (size_t j = 0; j < pushes_per_thread; ++j)
        clad::push<T>(t, x);
    });
  }

  for (auto& thread : threads)
    thread.join();

  size_t expected = n_threads * pushes_per_thread;
  size_t actual = t.size();
  if (expected != actual)
    printf("error: expected size %zu, actual size %zu\n", expected, actual);
}

// Benchmarking and testing thread safety with different configurations in
// multithreaded environment
static void BM_TapeThreadSafety(benchmark::State& state) {
  size_t n_threads = state.range(0);
  size_t pushes_per_thread = state.range(1);
  for (auto _ : state)
    concurrent_push<double>(/*x=*/1.0, /*n_threads=*/n_threads,
                            /*pushes_per_thread=*/pushes_per_thread);
}

BENCHMARK(BM_TapeThreadSafety)
    ->Args({1, 1000})
    ->Args({4, 1000})
    ->Args({8, 1000})
    ->Args({8, 1000})
    ->Args({16, 1000});

BENCHMARK_MAIN();