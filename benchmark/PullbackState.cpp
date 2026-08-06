#include "benchmark/benchmark.h"

#include "clad/Differentiator/BuiltinDerivatives.h"
#include "clad/Differentiator/Differentiator.h"

// clad::pullback_state<clad::no_state> must be a zero-cost abstraction: an
// empty payload, its per-call carrier, and the threading of that carrier into
// both the reverse_forw and the pullback have to fold away entirely at -O1+.
// These two benchmarks differentiate the same custom-derivative op -- one whose
// reverse_forw/pullback carry no state, one whose reverse_forw/pullback carry
// an empty pullback_state<no_state> -- and must report the same time within
// noise. (At -O2 clad emits byte-identical assembly for the two gradients.)
//
// The custom derivatives take precedence over the primal bodies, so clad
// routes through the reverse_forw for both variants, making the only
// difference the empty carrier. (The primals still need definitions: taking
// f_plain/f_state's address in clad::gradient odr-uses them, and their bodies
// call the ops.)

// Baseline: reverse_forw / pullback carry no state.
void op_plain(double* p) { *p = *p * *p; }
namespace clad::custom_derivatives {
inline void op_plain_reverse_forw(double* p, double* /*d_p*/) { *p = *p * *p; }
// NOLINTNEXTLINE(readability-non-const-parameter): clad matches a custom
// derivative against the primal's exact parameter types; const here breaks it.
inline void op_plain_pullback(double* p, double* d_p) { *d_p += 2.0 * *p; }
} // namespace clad::custom_derivatives
double f_plain(double x) {
  double v = x;
  op_plain(&v);
  return v;
}

// Same op, but the reverse_forw / pullback carry pullback_state<no_state>.
void op_state(double* p) { *p = *p * *p; }
namespace clad::custom_derivatives {
inline void op_state_reverse_forw(double* p, double* /*d_p*/,
                                  clad::pullback_state<clad::no_state>&) {
  *p = *p * *p;
}
// NOLINTNEXTLINE(readability-non-const-parameter): see op_plain_pullback.
inline void op_state_pullback(double* p, double* d_p,
                              clad::pullback_state<clad::no_state>) {
  *d_p += 2.0 * *p;
}
} // namespace clad::custom_derivatives
double f_state(double x) {
  double v = x;
  op_state(&v);
  return v;
}

static void BM_PullbackNoStateOverhead_Plain(benchmark::State& state) {
  auto grad = clad::gradient(f_plain, "x");
  double dx = 0;
  for (auto _ : state) {
    dx = 0;
    grad.execute(2.0, &dx);
    benchmark::DoNotOptimize(dx);
  }
}
BENCHMARK(BM_PullbackNoStateOverhead_Plain);

static void BM_PullbackNoStateOverhead_State(benchmark::State& state) {
  auto grad = clad::gradient(f_state, "x");
  double dx = 0;
  for (auto _ : state) {
    dx = 0;
    grad.execute(2.0, &dx);
    benchmark::DoNotOptimize(dx);
  }
}
BENCHMARK(BM_PullbackNoStateOverhead_State);

// Define our main.
BENCHMARK_MAIN();
