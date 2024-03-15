#include "benchmark/benchmark.h"

#include "clad/Differentiator/Differentiator.h"

#include "BenchmarkedFunctions.h"

// Benchmark forward mode for weighted sum.
static void BM_ForwardModeWeightedSum(benchmark::State& state) {
  auto dp0 = clad::differentiate(weightedSum, "p[0]");
  auto dp1 = clad::differentiate(weightedSum, "p[1]");
  auto dp2 = clad::differentiate(weightedSum, "p[2]");
  auto dp3 = clad::differentiate(weightedSum, "p[3]");
  auto dp4 = clad::differentiate(weightedSum, "p[4]");

  auto dw0 = clad::differentiate(weightedSum, "w[0]");
  auto dw1 = clad::differentiate(weightedSum, "w[1]");
  auto dw2 = clad::differentiate(weightedSum, "w[2]");
  auto dw3 = clad::differentiate(weightedSum, "w[3]");
  auto dw4 = clad::differentiate(weightedSum, "w[4]");

  constexpr int n = 5;
  double inputs[n];
  double weights[n];
  for (int i = 0; i < n; ++i) {
    inputs[i] = i + 1;
    weights[i] = 1.0 / (double)(i + 1);
  }

  double sum = 0;
  for (auto _ : state) {
    benchmark::DoNotOptimize(
        sum +=
        dp0.execute(inputs, weights, n) + dp1.execute(inputs, weights, n) +
        dp2.execute(inputs, weights, n) + dp3.execute(inputs, weights, n) +
        dp4.execute(inputs, weights, n) + dw0.execute(inputs, weights, n) +
        dw1.execute(inputs, weights, n) + dw2.execute(inputs, weights, n) +
        dw3.execute(inputs, weights, n) + dw4.execute(inputs, weights, n));
  }
}
BENCHMARK(BM_ForwardModeWeightedSum);

// Benchmark reverse mode for weighted sum.
static void BM_ReverseModeWeightedSum(benchmark::State& state) {
  auto grad = clad::gradient(weightedSum, "p, w");
  constexpr int n = 5;

  double inputs[n];
  double weights[n];
  for (int i = 0; i < n; ++i) {
    inputs[i] = i + 1;
    weights[i] = 1.0 / (double)(i + 1);
  }

  double dinp[n];
  double dweights[n];

  double sum = 0;
  for (auto _ : state) {
    grad.execute(inputs, weights, n, dinp, dweights);
    for (int i = 0; i < n; ++i) {
      sum += dinp[i] + dweights[i];
      dinp[i] = 0;
      dweights[i] = 0;
    }
  }
}
BENCHMARK(BM_ReverseModeWeightedSum);

// Benchmark vector forward mode for weighted sum.
static void BM_VectorForwardModeWeightedSum(benchmark::State& state) {
  auto vm_grad =
      clad::differentiate<clad::opts::vector_mode>(weightedSum, "p, w");
  constexpr int n = 5;

  double inputs[n];
  double weights[n];
  for (int i = 0; i < n; ++i) {
    inputs[i] = i + 1;
    weights[i] = 1.0 / (double)(i + 1);
  }

  double dinp[n];
  double dweights[n];
  clad::array_ref<double> dinp_ref(dinp, n);
  clad::array_ref<double> dweights_ref(dweights, n);

  double sum = 0;
  for (auto _ : state) {
    vm_grad.execute(inputs, weights, n, dinp_ref, dweights_ref);
    for (int i = 0; i < n; ++i) {
      sum += dinp[i] + dweights[i];
      dinp[i] = 0;
      dweights[i] = 0;
    }
  }
}
BENCHMARK(BM_VectorForwardModeWeightedSum);

// Define our main.
BENCHMARK_MAIN();
