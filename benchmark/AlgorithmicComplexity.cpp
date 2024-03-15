#include "benchmark/benchmark.h"

#include "clad/Differentiator/Differentiator.h"

#undef CLAD_NO_NUM_DIFF
#include "clad/Differentiator/NumericalDiff.h" // for the numerical comparison

#include "BenchmarkedFunctions.h"

// Compare the execution of forward, reverse and numerical diff.
// FIXME: Make the benchmark work with a range of inputs. That's currently
// problematic for reverse mode.

// inline double gaus(double* x, double* p /*means*/, double sigma, int dim);
static void BM_NumericGausP(benchmark::State& state) {
  using namespace numerical_diff;
  long double sum = 0;
  double x[] = {1, 1, 1, 1, 1};
  double p[] = {1, 2, 3, 4, 5};
  double dx[5] = {0, 0, 0, 0, 0};
  double dp[5] = {0, 0, 0, 0, 0};
  clad::tape<clad::array_ref<double>> results = {};
  int dim = 5;
  results.emplace_back(dx, dim);
  results.emplace_back(dp, dim);
  for (auto _ : state) {
    central_difference(gaus, results, /*printErrors*/ false, x, p,
                       /*sigma*/ 2., /*dim*/ dim);

    for (int i = 0; i < dim; i++) {
      benchmark::DoNotOptimize(sum += dp[i]);
      dp[i] = 0; // clear for the next benchmark iteration
    }
  }
}
// FIXME: Add the right interface to numerical_diff and enable the BM.
//BENCHMARK(BM_NumericGausP);

static void BM_ForwardGausP(benchmark::State& state) {
  auto dfdp0 = clad::differentiate(gaus, "p[0]");
  auto dfdp1 = clad::differentiate(gaus, "p[1]");
  auto dfdp2 = clad::differentiate(gaus, "p[2]");
  auto dfdp3 = clad::differentiate(gaus, "p[3]");
  auto dfdp4 = clad::differentiate(gaus, "p[4]");
  long double sum = 0;
  double x[] = {1, 1, 1, 1, 1};
  double p[] = {1, 2, 3, 4, 5};
  for (auto _ : state) {
    benchmark::DoNotOptimize(
        sum += dfdp0.execute(x, p, /*sigma*/ 2., /*dim*/ 5) +
               dfdp1.execute(x, p, /*sigma*/ 2., /*dim*/ 5) +
               dfdp2.execute(x, p, /*sigma*/ 2., /*dim*/ 5) +
               dfdp3.execute(x, p, /*sigma*/ 2., /*dim*/ 5) +
               dfdp4.execute(x, p, /*sigma*/ 2., /*dim*/ 5));
  }
}
BENCHMARK(BM_ForwardGausP);

static void BM_ReverseGausP(benchmark::State& state) {
  auto dfdp_grad = clad::gradient(gaus, "p");
  double x[] = {1, 1, 1, 1, 1};
  double p[] = {1, 2, 3, 4, 5};
  long double sum = 0;
  int dim = 5;
  double result[5] = {};
  for (auto _ : state) {
    dfdp_grad.execute(x, p, /*sigma*/ 2, dim, result);
    for (int i = 0; i < dim; i++) {
      benchmark::DoNotOptimize(sum += result[i]);
      result[i] = 0; // clear for the next benchmark iteration
    }
  }
}
BENCHMARK(BM_ReverseGausP);

// Define our main.
BENCHMARK_MAIN();
