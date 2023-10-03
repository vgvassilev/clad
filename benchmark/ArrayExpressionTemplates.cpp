#include "benchmark/benchmark.h"

#include "clad/Differentiator/Differentiator.h"

// Benchmark the expression x*y + y*z + z*x between clad arrays,
// this is to compare the performance of expression templates.
// We will evaluate the expression on using four different methods:
// 1. Using operations on clad arrays - this will use expression templates.
// 2. Using clad arrays but creating temporaries manually.
// 3. Using loops on clad arrays.
// 4. Using loops on native arrays.

// Benchmark expression templates.
static void BM_ExpressionTemplates(benchmark::State& state) {
  constexpr int n = 1000;
  clad::array<double> x(n);
  clad::array<double> y(n);
  clad::array<double> z(n);
  for (int i = 0; i < n; ++i) {
    x[i] = i + 1;
    y[i] = i + 2;
    z[i] = i + 3;
  }

  clad::array<double> res(n);
  for (auto _ : state)
    benchmark::DoNotOptimize(res = x * y + y * z + z * x);
}
BENCHMARK(BM_ExpressionTemplates);

// Benchmark manually creating temporaries.
static void BM_ManualTemporaries(benchmark::State& state) {
  constexpr int n = 1000;
  clad::array<double> x(n);
  clad::array<double> y(n);
  clad::array<double> z(n);
  for (int i = 0; i < n; ++i) {
    x[i] = i + 1;
    y[i] = i + 2;
    z[i] = i + 3;
  }

  clad::array<double> res(n);
  for (auto _ : state) {
    clad::array<double> temp1 = x * y;
    clad::array<double> temp2 = y * z;
    clad::array<double> temp3 = z * x;
    clad::array<double> temp4 = temp1 + temp2;
    benchmark::DoNotOptimize(res = temp4 + temp3);
  }
}
BENCHMARK(BM_ManualTemporaries);

// Benchmark loops on clad arrays.
static void BM_LoopsOnCladArrays(benchmark::State& state) {
  constexpr int n = 1000;
  clad::array<double> x(n);
  clad::array<double> y(n);
  clad::array<double> z(n);
  for (int i = 0; i < n; ++i) {
    x[i] = i + 1;
    y[i] = i + 2;
    z[i] = i + 3;
  }

  clad::array<double> res(n);
  for (auto _ : state) {
    for (int i = 0; i < n; ++i) {
      benchmark::DoNotOptimize(res[i] =
                                   x[i] * y[i] + y[i] * z[i] + z[i] * x[i]);
    }
  }
}
BENCHMARK(BM_LoopsOnCladArrays);

// Benchmark loops on native arrays.
static void BM_LoopsOnNativeArrays(benchmark::State& state) {
  constexpr int n = 1000;
  double* x = new double[n];
  double* y = new double[n];
  double* z = new double[n];
  for (int i = 0; i < n; ++i) {
    x[i] = i + 1;
    y[i] = i + 2;
    z[i] = i + 3;
  }

  double* res = new double[n];
  for (auto _ : state) {
    for (int i = 0; i < n; ++i) {
      benchmark::DoNotOptimize(res[i] =
                                   x[i] * y[i] + y[i] * z[i] + z[i] * x[i]);
    }
  }

  delete[] x;
  delete[] y;
  delete[] z;
  delete[] res;
}
BENCHMARK(BM_LoopsOnNativeArrays);

// Define our main.
BENCHMARK_MAIN();