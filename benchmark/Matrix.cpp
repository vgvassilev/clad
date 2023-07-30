#include "benchmark/benchmark.h"

#include "clad/Differentiator/Differentiator.h"

// Benchmarking clad::matrix initialization as identity matrix.
static void BM_MatrixIdentity(benchmark::State& state) {
  unsigned N = state.range(0);
  for (auto _ : state) {
    clad::matrix<double> m = clad::identity_matrix<double>(N, N);
  }
}

BENCHMARK(BM_MatrixIdentity)
    ->RangeMultiplier(2)
    ->Range(4, 256);

// Benchmarking scalar operation on a row of clad::matrix.
static void BM_MatrixRowOp(benchmark::State& state) {
    unsigned N = state.range(0);
    clad::matrix<double> m(N, N);
    unsigned mid = N / 2;
    for (auto _ : state) {
        benchmark::DoNotOptimize(
            m[mid] *= 2
        );
    }
}

BENCHMARK(BM_MatrixRowOp)
    ->RangeMultiplier(2)
    ->Range(4, 256);

// Benchmarking sum of two rows of a clad::matrix.
static void BM_MatrixRowSum(benchmark::State& state) {
    unsigned N = state.range(0);
    clad::matrix<double> m(N, N);
    clad::array<double> res(N);
    unsigned mid = N / 2;
    for (auto _ : state) {
        benchmark::DoNotOptimize(
            res = m[0] + m[mid]
        );
    }
}

BENCHMARK(BM_MatrixRowSum)
    ->RangeMultiplier(2)
    ->Range(4, 256);

// Benchmarking sum of all rows of a clad::matrix.
static void BM_MatrixCompleteRowSum(benchmark::State& state) {
    unsigned N = state.range(0);
    clad::matrix<double> m(N, N);
    clad::array<double> res(N);
    for (auto _ : state) {  
      for (unsigned i = 0; i < N; ++i) {
          benchmark::DoNotOptimize(res += m[i]);
      }
    }
}

BENCHMARK(BM_MatrixRowSum)
    ->RangeMultiplier(2)
    ->Range(4, 256);

BENCHMARK_MAIN();
