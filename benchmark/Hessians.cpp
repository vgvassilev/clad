#include "benchmark/benchmark.h"

#include "clad/Differentiator/Differentiator.h"

#include "BenchmarkedFunctions.h"

// Benchmark Hessian diagonal sum computation, by computing the
// entire computation.
static void BM_HessianCompleteComputation(benchmark::State& state) {
  auto dfdx2 = clad::hessian(weightedSum, "p[0:1],w[0:1]");
  double p[] = {1, 2};
  double w[] = {3, 4};
  unsigned long long sum = 0;
  double hessianMatrix[16] = {};
  for (unsigned i = 0; i < 16; i++)
    hessianMatrix[i] = 0.0;
  for (auto _ : state) {
    dfdx2.execute(p, w, 3, hessianMatrix);
    for (int i = 0; i < 4; i++)
      // Sum the diagonal of the Hessian matrix.
      benchmark::DoNotOptimize(sum += hessianMatrix[i * 4 + i]);
  }
}
BENCHMARK(BM_HessianCompleteComputation);

// Benchmark Hessian diagonal sum computation, by computing only
// the diagonal elements.
static void BM_HessianDiagonalComputation(benchmark::State& state) {
  auto dfdx2 =
      clad::hessian<clad::opts::diagonal_only>(weightedSum, "p[0:1],w[0:1]");
  double p[] = {1, 2};
  double w[] = {3, 4};
  unsigned long long sum = 0;
  double diagonalHessian[4] = {};
  for (unsigned i = 0; i < 4; i++)
    diagonalHessian[i] = 0.0;
  for (auto _ : state) {
    dfdx2.execute(p, w, 3, diagonalHessian);
    for (int i = 0; i < 4; i++)
      // Sum the diagonal of the Hessian matrix.
      benchmark::DoNotOptimize(sum += diagonalHessian[i]);
  }
}
BENCHMARK(BM_HessianDiagonalComputation);

// Define our main.
BENCHMARK_MAIN();
