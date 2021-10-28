#include "benchmark/benchmark.h"

// FIXME: If we move this before benchmark.h we have tons of errors due to a bug
#include "clad/Differentiator/Differentiator.h"

double pow2(double x) { return x * x; }

static void BM_SumCladExecute(benchmark::State &state) {
  auto dfdx = clad::differentiate(pow2, "x");
  unsigned long long sum = 0;
  for (auto _ : state) {
    benchmark::DoNotOptimize(sum += dfdx.execute(1));
  }
}
BENCHMARK(BM_SumCladExecute);

double pow2_darg0(double);
static void BM_SumCladFwdDecl(benchmark::State &state) {
  auto dfdx = clad::differentiate(pow2, "x");
  unsigned long long sum = 0;
  for (auto _ : state) {
    benchmark::DoNotOptimize(sum += pow2_darg0(1));
  }
}
BENCHMARK(BM_SumCladFwdDecl);

// Define our main.
BENCHMARK_MAIN();
