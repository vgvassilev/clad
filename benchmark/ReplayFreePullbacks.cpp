#include "benchmark/benchmark.h"

#include "clad/Differentiator/BuiltinDerivatives.h"
#include "clad/Differentiator/Differentiator.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

// The reverse-mode protocol's cost, on the smallest kernel that reproduces it.
//
// `objective` below has the shape that makes reverse-mode expensive: a mutating
// leaf callee (`qtimesx`) whose pullback needs an input buffer (`xc`) that the
// caller overwrites on the next iteration. clad handles that today by pairing a
// `restore_tracker` snapshot at the call site with a primal replay inside the
// pullback -- one snapshot, two restores and one replay per call per iteration.
//
// `objective_grad_replay_free` is the same gradient in the shape a replay-free
// clad would emit: the forward sweep runs the primal once and records the value
// the pullback consumes into a `clad::pullback_state` payload; the reverse
// sweep consumes it and never re-executes the primal. No tracker, no restores,
// no replay. It is hand-written here only so the target is measurable before
// the codegen exists. BM_ReverseProtocol_Agreement below holds the two to 1e-10
// of each other, so the ratio measures the protocol and not the math.
//
// Keep both benchmarks in sync when the generated shape changes: the point of
// this file is the *ratio*, which is the exit gate for the replay-free work.

constexpr int D = 8;
constexpr int K = 4;
constexpr int N = 512;
constexpr int LSZ = D * (D - 1) / 2;

// Raw arrays walked through pointers are what this file measures: it is the
// shape clad's gradient is handed through CladFunction::execute, and a bounds-
// checked container would not be the same code. The checks below say so once
// rather than on every line of every kernel.
// NOLINTBEGIN(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays,cppcoreguidelines-pro-bounds-pointer-arithmetic,cppcoreguidelines-pro-bounds-constant-array-index,cppcoreguidelines-pro-bounds-array-to-pointer-decay)

static void subtract(const double* a, const double* b, double* out) {
  for (int i = 0; i < D; i++)
    out[i] = a[i] - b[i];
}

static void qtimesx(const double* Qd, const double* L, const double* x,
                    double* out) {
  for (int i = 0; i < D; i++)
    out[i] = Qd[i] * x[i];
  int p = 0;
  for (int i = 0; i < D; i++)
    for (int j = i + 1; j < D; j++, p++)
      out[j] = out[j] + (L[p] * x[i]);
}

static double sqnorm(const double* v) {
  double s = 0;
  for (int i = 0; i < D; i++)
    s += v[i] * v[i];
  return s;
}

static double objective(const double* mu, const double* Qd, const double* L,
                        const double* x) {
  double total = 0;
  double xc[D];
  double qx[D];
  for (int ix = 0; ix < N; ix++)
    for (int ik = 0; ik < K; ik++) {
      subtract(&x[ix * D], &mu[ik * D], xc);
      qtimesx(&Qd[ik * D], &L[ik * LSZ], xc, qx);
      total += sqnorm(qx);
    }
  return total;
}

/// What the compiler would synthesize from `objective`'s TBR marks: the one
/// value its callees' pullbacks need and the caller clobbers.
struct objective_state {
  clad::tape<double> t_xc;
};

#ifdef _MSC_VER
#define CLAD_BM_NOINLINE __declspec(noinline)
#else
#define CLAD_BM_NOINLINE __attribute__((noinline))
#endif

// Must be reached out of line. clad's gradient is called through
// `CladFunction::execute`, so its pointer arguments carry no provenance and the
// optimizer cannot promote adjoint accumulators out of the inner loops. Letting
// this reference inline into the benchmark loop hands it that advantage and
// overstates the ratio by ~4.5x -- the whole point of the number is that both
// sides are opaque.
CLAD_BM_NOINLINE static void
objective_grad_replay_free(const double* mu, const double* Qd, const double* L,
                           const double* x, double* d_mu, double* d_Qd,
                           double* d_L) {
  clad::pullback_state<objective_state> st;
  double xc[D];
  double qx[D];
  double d_xc[D];
  double d_qx[D];

  for (int ix = 0; ix < N; ix++)
    for (int ik = 0; ik < K; ik++) {
      subtract(&x[ix * D], &mu[ik * D], xc);
      for (double e : xc)
        clad::push(st.data.t_xc, e);
      qtimesx(&Qd[ik * D], &L[ik * LSZ], xc, qx);
    }

  for (int ix = N - 1; ix >= 0; ix--)
    for (int ik = K - 1; ik >= 0; ik--) {
      for (int i = D - 1; i >= 0; i--)
        xc[i] = clad::pop(st.data.t_xc);
      // qx is a cheap function of the recorded xc, so recompute rather than
      // record it -- the store-vs-recompute choice a cost model would make.
      qtimesx(&Qd[ik * D], &L[ik * LSZ], xc, qx);
      for (int i = 0; i < D; i++)
        d_qx[i] = 2.0 * qx[i];
      for (int i = 0; i < D; i++)
        d_xc[i] = Qd[(ik * D) + i] * d_qx[i];
      int p = 0;
      for (int i = 0; i < D; i++) {
        // The adjoint of x[i] is invariant in the j loop. Accumulating it in
        // memory would leave a store->load round-trip in the recurrence that
        // the optimizer cannot promote, because the neighbouring d_L store may
        // alias it.
        double acc = d_xc[i];
        for (int j = i + 1; j < D; j++, p++) {
          d_L[(ik * LSZ) + p] += d_qx[j] * xc[i];
          acc += L[(ik * LSZ) + p] * d_qx[j];
        }
        d_xc[i] = acc;
      }
      for (int i = 0; i < D; i++)
        d_Qd[(ik * D) + i] += d_qx[i] * xc[i];
      for (int i = 0; i < D; i++)
        d_mu[(ik * D) + i] -= d_xc[i];
    }
}
// NOLINTEND(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays,cppcoreguidelines-pro-bounds-pointer-arithmetic,cppcoreguidelines-pro-bounds-constant-array-index,cppcoreguidelines-pro-bounds-array-to-pointer-decay)

namespace {
struct Inputs {
  std::vector<double> mu;
  std::vector<double> Qd;
  std::vector<double> L;
  std::vector<double> x;
  Inputs() : mu(K * D), Qd(K * D), L(K * LSZ), x(N * D) {
    std::mt19937 gen(7);
    std::normal_distribution<double> dist(0, 1);
    for (auto* v : {&mu, &Qd, &L, &x})
      for (double& e : *v)
        e = 0.3 * dist(gen);
  }
};
} // namespace

static const Inputs& inputs() {
  static const Inputs I;
  return I;
}

static void BM_ReverseProtocol_CladToday(benchmark::State& state) {
  const Inputs& in = inputs();
  auto grad = clad::gradient(objective, /*args=*/"mu,Qd,L");
  std::vector<double> d_mu(K * D);
  std::vector<double> d_Qd(K * D);
  std::vector<double> d_L(K * LSZ);
  for (auto _ : state) {
    std::fill(d_mu.begin(), d_mu.end(), 0.0);
    std::fill(d_Qd.begin(), d_Qd.end(), 0.0);
    std::fill(d_L.begin(), d_L.end(), 0.0);
    grad.execute(in.mu.data(), in.Qd.data(), in.L.data(), in.x.data(),
                 d_mu.data(), d_Qd.data(), d_L.data());
    benchmark::DoNotOptimize(d_mu.data());
    benchmark::ClobberMemory();
  }
}
BENCHMARK(BM_ReverseProtocol_CladToday);

static void BM_ReverseProtocol_ReplayFree(benchmark::State& state) {
  const Inputs& in = inputs();
  std::vector<double> d_mu(K * D);
  std::vector<double> d_Qd(K * D);
  std::vector<double> d_L(K * LSZ);
  for (auto _ : state) {
    std::fill(d_mu.begin(), d_mu.end(), 0.0);
    std::fill(d_Qd.begin(), d_Qd.end(), 0.0);
    std::fill(d_L.begin(), d_L.end(), 0.0);
    objective_grad_replay_free(in.mu.data(), in.Qd.data(), in.L.data(),
                               in.x.data(), d_mu.data(), d_Qd.data(),
                               d_L.data());
    benchmark::DoNotOptimize(d_mu.data());
    benchmark::ClobberMemory();
  }
}
BENCHMARK(BM_ReverseProtocol_ReplayFree);

// A ratio between two gradients that disagree measures nothing, so check them
// against each other once before reporting.
static void BM_ReverseProtocol_Agreement(benchmark::State& state) {
  const Inputs& in = inputs();
  auto grad = clad::gradient(objective, /*args=*/"mu,Qd,L");
  std::vector<double> c_mu(K * D);
  std::vector<double> c_Qd(K * D);
  std::vector<double> c_L(K * LSZ);
  std::vector<double> r_mu(K * D);
  std::vector<double> r_Qd(K * D);
  std::vector<double> r_L(K * LSZ);
  grad.execute(in.mu.data(), in.Qd.data(), in.L.data(), in.x.data(),
               c_mu.data(), c_Qd.data(), c_L.data());
  objective_grad_replay_free(in.mu.data(), in.Qd.data(), in.L.data(),
                             in.x.data(), r_mu.data(), r_Qd.data(), r_L.data());
  double worst = 0;
  auto compare = [&worst](const std::vector<double>& a,
                          const std::vector<double>& b) {
    for (size_t i = 0; i < a.size(); i++)
      worst = std::max(worst,
                       std::abs(a[i] - b[i]) / std::max(1e-8, std::abs(b[i])));
  };
  compare(r_mu, c_mu);
  compare(r_Qd, c_Qd);
  compare(r_L, c_L);
  if (worst > 1e-10)
    state.SkipWithError("replay-free gradient disagrees with clad's");
  for (auto _ : state)
    benchmark::DoNotOptimize(worst);
}
BENCHMARK(BM_ReverseProtocol_Agreement);

// google/benchmark's main, whose expansion this file does not own.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays,cppcoreguidelines-pro-type-reinterpret-cast)
BENCHMARK_MAIN();
