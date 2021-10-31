#include "benchmark/benchmark.h"

// FIXME: If we move this before benchmark.h we have tons of errors due to a bug
#include "clad/Differentiator/Differentiator.h"

#include "BenchmarkedFunctions.h"

// Benchmark calling via CladFunction::execute
static void BM_ForwardModePow2Execute(benchmark::State &state) {
  auto dfdx = clad::differentiate(pow2, "x");
  unsigned long long sum = 0;
  for (auto _ : state) {
    benchmark::DoNotOptimize(sum += dfdx.execute(1));
  }
}
BENCHMARK(BM_ForwardModePow2Execute);

// Benchmark calling via a forward declaration.
double pow2_darg0(double);
static void BM_ForwardModePow2FwdDecl(benchmark::State &state) {
  auto dfdx = clad::differentiate(pow2, "x");
  (void) dfdx;
  unsigned long long sum = 0;
  for (auto _ : state) {
    benchmark::DoNotOptimize(sum += pow2_darg0(1));
  }
}
BENCHMARK(BM_ForwardModePow2FwdDecl);

// Benchmark calling the gradient via CladFunction::execute.
inline void sum_grad_0(double*, int, clad::array_ref<double>);
static void BM_ReverseModeSumFwdDecl(benchmark::State &state) {
  auto grad = clad::gradient(sum, "p");
  (void) grad;
  double inputs[] = {1, 2, 3, 4, 5};
  double result[3] = {};
  unsigned long long sum = 0;
  for (auto _ : state) {
    sum_grad_0(inputs,/*dim*/ 3, result);
    benchmark::DoNotOptimize(sum += result[0] + result[1] + result[2]);
  }
}
BENCHMARK(BM_ReverseModeSumFwdDecl);


// Benchmark calling the gradient via CladFunction::execute.
static void BM_ReverseModeSumExecute(benchmark::State &state) {
  auto grad = clad::gradient(sum, "p");
  double inputs[] = {1, 2, 3, 4, 5};
  double result[3] = {};
  unsigned long long sum = 0;
  for (auto _ : state) {
    grad.execute(inputs,/*dim*/ 3, result);
    benchmark::DoNotOptimize(sum += result[0] + result[1] + result[2]);
  }
}
BENCHMARK(BM_ReverseModeSumExecute);

// Define our main.
BENCHMARK_MAIN();
