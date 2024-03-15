#include "benchmark/benchmark.h"
#include "clad/Differentiator/Differentiator.h"

#include "BenchmarkedFunctions.h"

// Compare the execution of Reverse Mode Enzyme and Clad.

static void
BM_ReverseModeAddArrayAndMultiplyWithScalarsExecute(benchmark::State& state) {
  auto grad = clad::gradient(addArrayAndMultiplyWithScalars);
  double x = 5, y = 6;
  double dx = 0, dy = 0;
  int n = 5;
  int dn = 0;
  double arr[5] = {1, 2, 3, 4, 5};
  double darr[5] = {0};
  for (auto _ : state) {
    grad.execute(arr, x, y, 5, darr, &dx, &dy, &dn);
    dx = 0;
    dy = 0;
    for (int i = 0; i < n; i++) {
      darr[i] = 0;
    }
  }
}

BENCHMARK(BM_ReverseModeAddArrayAndMultiplyWithScalarsExecute);

static void BM_VectorForwardModeAddArrayAndMultiplyWithScalarsExecute(
    benchmark::State& state) {
  auto grad = clad::differentiate<clad::opts::vector_mode>(
      addArrayAndMultiplyWithScalars);
  double x = 5, y = 6;
  double dx = 0, dy = 0;
  int n = 5;
  int dn = 0;
  double arr[5] = {1, 2, 3, 4, 5};
  double darr[5] = {0};
  for (auto _ : state) {
    grad.execute(arr, x, y, 5, darr, &dx, &dy, &dn);
    dx = 0;
    dy = 0;
    for (int i = 0; i < n; i++)
      darr[i] = 0;
  }
}

BENCHMARK(BM_VectorForwardModeAddArrayAndMultiplyWithScalarsExecute);

static void BM_ReverseModeAddArrayAndMultiplyWithScalarsExecuteEnzyme(
    benchmark::State& state) {
  auto grad =
      clad::gradient<clad::opts::use_enzyme>(addArrayAndMultiplyWithScalars);
  double x = 5, y = 6;
  double dx = 0, dy = 0;
  int n = 5;
  int dn = 0;
  double arr[5] = {1, 2, 3, 4, 5};
  double darr[5] = {0};
  for (auto _ : state) {
    grad.execute(arr, x, y, 5, darr, &dx, &dy, &dn);
    dx = 0;
    dy = 0;
    for (int i = 0; i < n; i++) {
      darr[i] = 0;
    }
  }
}

BENCHMARK(BM_ReverseModeAddArrayAndMultiplyWithScalarsExecuteEnzyme);

static void BM_ReverseModeSumExecute(benchmark::State& state) {
  auto grad = clad::gradient(sum);
  double inputs[] = {1, 2, 3, 4, 5};
  double result[5] = {};
  for (auto _ : state) {
    grad.execute(inputs, /*dim*/ 5, result);
    for (int i = 0; i < 5; i++) {
      result[i] = 0;
    }
  }
}
BENCHMARK(BM_ReverseModeSumExecute);

static void BM_VectorForwardModeSumExecute(benchmark::State& state) {
  auto grad = clad::differentiate<clad::opts::vector_mode>(sum, "p");
  double inputs[] = {1, 2, 3, 4, 5};
  double result[5] = {};
  for (auto _ : state) {
    grad.execute(inputs, /*dim*/ 5, result);
    for (int i = 0; i < 5; i++)
      result[i] = 0;
  }
}
BENCHMARK(BM_VectorForwardModeSumExecute);

static void BM_ReverseModeSumExecuteWithEnzyme(benchmark::State& state) {
  auto grad = clad::gradient<clad::opts::use_enzyme>(sum);
  double inputs[] = {1, 2, 3, 4, 5};
  double result[5] = {};
  for (auto _ : state) {
    grad.execute(inputs, /*dim*/ 5, result);
    for (int i = 0; i < 5; i++) {
      result[i] = 0;
    }
  }
}
BENCHMARK(BM_ReverseModeSumExecuteWithEnzyme);

static void BM_ReverseModeProductExecute(benchmark::State& state) {
  auto grad = clad::gradient(product);
  double inputs[] = {1, 2, 3, 4, 5};
  double result[5] = {};
  for (auto _ : state) {
    grad.execute(inputs, /*dim*/ 5, result);
    for (int i = 0; i < 5; i++) {
      result[i] = 0;
    }
  }
}
BENCHMARK(BM_ReverseModeProductExecute);

static void BM_VectorForwardModeProductExecute(benchmark::State& state) {
  auto grad = clad::differentiate<clad::opts::vector_mode>(product, "p");
  double inputs[] = {1, 2, 3, 4, 5};
  double result[5] = {};
  for (auto _ : state) {
    grad.execute(inputs, /*dim*/ 5, result);
    for (int i = 0; i < 5; i++)
      result[i] = 0;
  }
}
BENCHMARK(BM_VectorForwardModeProductExecute);

static void BM_ReverseModeProductExecuteEnzyme(benchmark::State& state) {
  auto grad = clad::gradient<clad::opts::use_enzyme>(product);
  double inputs[] = {1, 2, 3, 4, 5};
  double result[5] = {};
  for (auto _ : state) {
    grad.execute(inputs, /*dim*/ 5, result);
    for (int i = 0; i < 5; i++) {
      result[i] = 0;
    }
  }
}
BENCHMARK(BM_ReverseModeProductExecuteEnzyme);

static void BM_ReverseGaus(benchmark::State& state) {
  auto dfdp_grad = clad::gradient(gaus);
  double x[] = {1, 1, 1, 1, 1};
  double p[] = {1, 2, 3, 4, 5};
  int dim = 5;

  double dx[5];
  double dp[5];
  double ds;
  int ddim;

  for (auto _ : state) {
    dfdp_grad.execute(x, p, /*sigma*/ 2, dim, dx, dp, &ds, &ddim);
    for (int i = 0; i < dim; i++) {
      dx[i] = 0; // clear for the next benchmark iteration
      dp[i] = 0; // clear for the next benchmark iteration
    }
    ds = 0;
    ddim = 0;
  }
}

BENCHMARK(BM_ReverseGaus);

static void BM_ReverseGausEnzyme(benchmark::State& state) {
  auto dfdp_grad = clad::gradient<clad::opts::use_enzyme>(gaus);
  double x[] = {1, 1, 1, 1, 1};
  double p[] = {1, 2, 3, 4, 5};
  int dim = 5;

  double dx[5];
  double dp[5];
  double ds;
  int ddim;

  for (auto _ : state) {
    dfdp_grad.execute(x, p, /*sigma*/ 2, dim, dx, dp, &ds, &ddim);
    for (int i = 0; i < dim; i++) {
      dx[i] = 0; // clear for the next benchmark iteration
      dp[i] = 0; // clear for the next benchmark iteration
    }
    ds = 0;
    ddim = 0;
  }
}

BENCHMARK(BM_ReverseGausEnzyme);

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

  double sum = 0;
  for (auto _ : state) {
    vm_grad.execute(inputs, weights, n, dinp, dweights);
    for (int i = 0; i < n; ++i) {
      sum += dinp[i] + dweights[i];
      dinp[i] = 0;
      dweights[i] = 0;
    }
  }
}

BENCHMARK(BM_VectorForwardModeWeightedSum);

// Benchmark reverse mode for weighted sum with Enzyme.
static void BM_ReverseModeWeightedSumEnzyme(benchmark::State& state) {
  auto grad = clad::gradient<clad::opts::use_enzyme>(weightedSum, "p, w");
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

BENCHMARK(BM_ReverseModeWeightedSumEnzyme);

// Define our main.
BENCHMARK_MAIN();
