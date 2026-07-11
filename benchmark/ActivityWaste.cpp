#include "benchmark/benchmark.h"

#include "clad/Differentiator/Differentiator.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
double dummy_out = 0; // Sink with external linkage: keeps the passive and dead
                      // work live at -O2 and out of reach of constant folding.

namespace {

struct MemoryManager : public benchmark::MemoryManager {
  std::size_t CurNumAllocs = 0;
  std::size_t CurNumDeallocs = 0;
  std::size_t CurMaxBytesUsed = 0;
  void Start() override {
    CurNumAllocs = 0;
    CurNumDeallocs = 0;
    CurMaxBytesUsed = 0;
  }
  void Stop(Result& result) override {
    result.num_allocs = static_cast<std::int64_t>(CurNumAllocs);
    result.max_bytes_used = static_cast<std::int64_t>(CurMaxBytesUsed);
  }
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
auto mm = std::make_unique<MemoryManager>();

struct InstrumentationRegistrer {
  InstrumentationRegistrer() { benchmark::RegisterMemoryManager(mm.get()); }
  ~InstrumentationRegistrer() { benchmark::RegisterMemoryManager(nullptr); }
  InstrumentationRegistrer(const InstrumentationRegistrer&) = delete;
  InstrumentationRegistrer& operator=(const InstrumentationRegistrer&) = delete;
  InstrumentationRegistrer(InstrumentationRegistrer&&) = delete;
  InstrumentationRegistrer& operator=(InstrumentationRegistrer&&) = delete;
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
InstrumentationRegistrer MemMgrRegister;

/// Zeroes the allocation counters on entry and publishes them as per-iteration
/// benchmark counters on exit. Buffers must be constructed *before* this, so
/// that setup allocations are not attributed to the measured region.
class AddBMCounterRAII {
  MemoryManager& m_MemMgr;
  benchmark::State& m_State;

public:
  AddBMCounterRAII(MemoryManager& m, benchmark::State& s)
      : m_MemMgr(m), m_State(s) {
    m.CurNumAllocs = 0;
    m.CurNumDeallocs = 0;
    m.CurMaxBytesUsed = 0;
  }
  AddBMCounterRAII(const AddBMCounterRAII&) = delete;
  AddBMCounterRAII& operator=(const AddBMCounterRAII&) = delete;
  AddBMCounterRAII(AddBMCounterRAII&&) = delete;
  AddBMCounterRAII& operator=(AddBMCounterRAII&&) = delete;
  ~AddBMCounterRAII() {
    const auto it = m_State.iterations();
    m_State.counters["AllocN"] =
        static_cast<double>(m_MemMgr.CurNumAllocs) / static_cast<double>(it);
    m_State.counters["DellocN"] =
        static_cast<double>(m_MemMgr.CurNumDeallocs) / static_cast<double>(it);
    m_State.counters["AllocBytes"] =
        static_cast<double>(m_MemMgr.CurMaxBytesUsed) / static_cast<double>(it);
  }
};

/// Upper bound for dead_array_chain's scratch buffer. Must stay >= the largest
/// value BM_DeadArrayChain passes to Range().
constexpr int kMaxN = 10000;

} // namespace

// The global allocation operators have to reach the C allocator directly:
// routing through ::operator new would recurse.
// NOLINTBEGIN(cppcoreguidelines-no-malloc,cppcoreguidelines-owning-memory)
void* operator new(std::size_t size) {
  if (mm) {
    mm->CurNumAllocs++;
    mm->CurMaxBytesUsed += size;
  }
  return malloc(size);
}
void operator delete(void* p) noexcept {
  if (mm)
    mm->CurNumDeallocs++;
  free(p);
}
void operator delete(void* p, std::size_t) noexcept {
  if (mm)
    mm->CurNumDeallocs++;
  free(p);
}
// NOLINTEND(cppcoreguidelines-no-malloc,cppcoreguidelines-owning-memory)

// The kernels below take `const double*` because that is the signature
// shape clad::gradient and the reverse-mode tape actually exercise, and
// dead_array_chain's scratch buffer is a raw stack array for a related
// reason: a std::vector would allocate inside the measured region and
// pollute the very counters this file reports. Neither can satisfy the
// bounds-safety checks without defeating the measurement.
// clang-analyzer-deadcode.DeadStores is silenced for the same span: it
// fires on the '_' in google-benchmark's canonical `for (auto _ : state)`
// loop in all six wrappers, and that variable is deliberately never read.
// NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic,cppcoreguidelines-pro-bounds-constant-array-index,clang-analyzer-deadcode.DeadStores)

//===----------------------------------------------------------------------===//
// Positive control.
//
// Every value recorded here is genuinely needed to compute dr/dx, so no
// activity analysis may prune any of it. It exists so that a zero in any case
// below can be read as "no waste" rather than "the counters stopped working" --
// for instance if a future change routes the tape through operator new[], which
// this file does not intercept.
//===----------------------------------------------------------------------===//
double positive_control(const double* x, int n) {
  double r = 0;
  for (int i = 0; i < n; ++i) {
    double t = x[i] * x[i];
    r += t * t;
  }
  return r;
}

static void BM_PositiveControl(benchmark::State& state) {
  const auto n = static_cast<int>(state.range(0));
  std::vector<double> x(n, 0.5);
  std::vector<double> dx(n, 0.0);
  auto grad = clad::gradient(positive_control, "x");
  {
    AddBMCounterRAII c(*mm, state);
    for (auto _ : state) {
      for (double& d : dx)
        d = 0;
      grad.execute(x.data(), n, dx.data());
      benchmark::DoNotOptimize(dx.data());
    }
  }
}
BENCHMARK(BM_PositiveControl)->RangeMultiplier(4)->Range(64, 4096);

//===----------------------------------------------------------------------===//
// Case 1: passive work behind a call boundary.
//
// inner_accumulate's loop is a non-linear recurrence in s, so every s is taped,
// yet s never reaches the return value -- it only reaches dummy_out. Without
// interprocedural activity information the caller cannot know that, so the
// entire chain is recorded.
//===----------------------------------------------------------------------===//
__attribute__((noinline)) double inner_accumulate(double x, int work) {
  double s = 1.0;
  for (int i = 0; i < work; ++i)
    s = 0.25 * s * s + 0.5; // Converges to 2 - sqrt(2); never folds, never
                            // overflows.
  dummy_out += s;
  return x;
}

double interproc_passive(const double* x, int n, int work) {
  double r = 0;
  for (int i = 0; i < n; ++i)
    r += inner_accumulate(x[i], work);
  return r;
}

static void BM_InterprocPassive(benchmark::State& state) {
  constexpr int n = 50;
  const auto work = static_cast<int>(state.range(0));
  std::array<double, n> x{};
  std::array<double, n> dx{};
  x.fill(0.5);
  auto grad = clad::gradient(interproc_passive, "x");
  {
    AddBMCounterRAII c(*mm, state);
    for (auto _ : state) {
      dx.fill(0.0);
      grad.execute(x.data(), n, work, dx.data());
      benchmark::DoNotOptimize(dx.data());
    }
  }
}
BENCHMARK(BM_InterprocPassive)->RangeMultiplier(5)->Range(100, 5000);

//===----------------------------------------------------------------------===//
// Case 2: one dead field inside an otherwise active struct.
//
// p.Active1 and p.Active2 reach the return value; p.Dead only reaches
// dummy_out. Field-insensitive activity information has to keep the whole
// object, so p.Dead's non-linear chain is taped.
//===----------------------------------------------------------------------===//
struct Particle {
  double Active1;
  double Active2;
  double Dead;
};

double dead_struct_field(const double* x, int n) {
  double r = 0;
  for (int i = 0; i < n; ++i) {
    // All three fields are assigned before any read. Zero-initialising would
    // add an unrelated init expression to the differentiated code.
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    Particle p;
    p.Active1 = x[i];
    p.Active2 = x[i] * x[i];
    p.Dead = 1.0;
    for (int j = 0; j < 2; ++j)
      p.Dead = p.Dead * p.Dead + x[i];
    r += p.Active1 + p.Active2;
    dummy_out += p.Dead;
  }
  return r;
}

static void BM_DeadStructField(benchmark::State& state) {
  const auto n = static_cast<int>(state.range(0));
  std::vector<double> x(n);
  std::vector<double> dx(n, 0.0);
  for (int i = 0; i < n; ++i)
    x[i] = 0.5 + 0.001 * static_cast<double>(i);
  auto grad = clad::gradient(dead_struct_field, "x");
  {
    AddBMCounterRAII c(*mm, state);
    for (auto _ : state) {
      for (double& d : dx)
        d = 0;
      grad.execute(x.data(), n, dx.data());
      benchmark::DoNotOptimize(dx.data());
    }
  }
}
BENCHMARK(BM_DeadStructField)->RangeMultiplier(4)->Range(64, 4096);

//===----------------------------------------------------------------------===//
// Case 3: a dead chain through a scratch array.
//
// Only x[0] reaches the return value, so every value the chain tapes is waste.
// The buffer is function-local, which makes it passive by construction: it is
// not part of the differentiated signature, so no caller can request its
// derivative and turn the chain active.
//===----------------------------------------------------------------------===//
double dead_array_chain(const double* x, int n) {
  // Fixed-size local buffer, deliberately not zero-initialised: indices >= n
  // are never read, and memsetting 80 KB on every timed call would swamp the
  // measurement.
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables,cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  double scratch[kMaxN];
  for (int i = 0; i < n; ++i)
    scratch[i] = x[i] * x[i];
  // The analyzer also models an n <= 0 call in which the fill loop never
  // runs and scratch stays uninitialised; the harness only passes n in
  // [1000, 10000] and BM_DeadArrayChain bounds n by kMaxN.
  for (int k = 0; k < n - 1; ++k)
    // NOLINTNEXTLINE(clang-analyzer-core.UndefinedBinaryOperatorResult)
    scratch[k] = scratch[k + 1] * scratch[k + 1];
  double ret = x[0] * x[0];
  // NOLINTNEXTLINE(clang-analyzer-core.uninitialized.Assign)
  dummy_out += scratch[0];
  return ret;
}

static void BM_DeadArrayChain(benchmark::State& state) {
  const auto n = static_cast<int>(state.range(0));
  if (n > kMaxN) {
    state.SkipWithError("n exceeds kMaxN: raise kMaxN or lower Range()");
    return;
  }
  std::vector<double> x(n);
  std::vector<double> dx(n, 0.0);
  for (int i = 0; i < n; ++i)
    x[i] = 0.5 + 0.001 * static_cast<double>(i);
  auto grad = clad::gradient(dead_array_chain, "x");
  {
    AddBMCounterRAII c(*mm, state);
    for (auto _ : state) {
      for (double& d : dx)
        d = 0;
      grad.execute(x.data(), n, dx.data());
      benchmark::DoNotOptimize(dx.data());
    }
  }
}
BENCHMARK(BM_DeadArrayChain)->RangeMultiplier(10)->Range(1000, 10000);

//===----------------------------------------------------------------------===//
// Case 4: work guarded by a runtime flag.
//
// t depends on x, so it is varied, but it only reaches dummy_out and never the
// return value. The adjoint of the whole guarded block is therefore waste --
// something only a usefulness analysis can see, not a variedness one.
//===----------------------------------------------------------------------===//
double branch_guarded(const double* x, int n, int flag) {
  double r = x[0] * x[0];
  if (flag) {
    double t = 0;
    // Bounded on purpose: this converges to 2 - sqrt(3) for x[i] == 0.5,
    // whereas t = t * t + x[i] has no real fixed point and reaches inf within
    // about twenty iterations.
    for (int i = 0; i < n; ++i)
      t = 0.25 * t * t + 0.5 * x[i];
    dummy_out += t;
  }
  return r;
}

static void BM_BranchGuarded(benchmark::State& state) {
  const auto n = static_cast<int>(state.range(0));
  std::vector<double> x(n, 0.5);
  std::vector<double> dx(n, 0.0);
  auto grad = clad::gradient(branch_guarded, "x");
  {
    AddBMCounterRAII c(*mm, state);
    for (auto _ : state) {
      for (double& d : dx)
        d = 0;
      grad.execute(x.data(), n, /*flag=*/1, dx.data());
      benchmark::DoNotOptimize(dx.data());
    }
  }
}
BENCHMARK(BM_BranchGuarded)->RangeMultiplier(4)->Range(64, 4096);

//===----------------------------------------------------------------------===//
// Case 5: useful but not varied.
//
// x1..x5 are seeded from constants, so nothing in the chain depends on x and
// d(x + x5)/dx is exactly 1. x5 does reach the return value, so usefulness
// alone has to keep the chain -- only variedness can prune it. The values reach
// a fixed point after four iterations (0.25, 0.0625, 0.00390625,
// 1.52587890625e-05) and neither overflow nor underflow, but the tape still
// records one entry per assignment per rep, which is what this measures.
//===----------------------------------------------------------------------===//
double useful_not_varied(double x, int reps) {
  double x1 = 0.5;
  double x2 = 0.5;
  double x3 = 0.5;
  double x4 = 0.5;
  double x5 = 0.5;
  for (int i = 0; i < reps; ++i) {
    x5 = x4 * x4;
    x4 = x3 * x3;
    x3 = x2 * x2;
    x2 = x1 * x1;
  }
  return x + x5;
}

static void BM_UsefulNotVaried(benchmark::State& state) {
  const auto reps = static_cast<int>(state.range(0));
  auto grad = clad::gradient(useful_not_varied, "x");
  {
    AddBMCounterRAII c(*mm, state);
    for (auto _ : state) {
      double dx = 0;
      grad.execute(0.5, reps, &dx);
      benchmark::DoNotOptimize(dx);
    }
  }
}
BENCHMARK(BM_UsefulNotVaried)->RangeMultiplier(10)->Range(100, 10000);

// NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic,cppcoreguidelines-pro-bounds-constant-array-index,clang-analyzer-deadcode.DeadStores)

BENCHMARK_MAIN();
