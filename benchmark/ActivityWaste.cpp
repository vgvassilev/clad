#include "benchmark/benchmark.h"

#include "ActivityWasteKernels.h"
#include "clad/Differentiator/Differentiator.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

// External sink; keep its writes observable outside this translation unit.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables,misc-use-internal-linkage)
double dummy_out = 0;

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

// Measure per-call allocations, excluding buffer setup and teardown.
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
    // Snapshot before publishing: counter-map insertion can allocate.
    const auto allocations = m_MemMgr.CurNumAllocs;
    const auto deallocations = m_MemMgr.CurNumDeallocs;
    const auto bytes = m_MemMgr.CurMaxBytesUsed;
    m_State.counters["AllocN"] =
        static_cast<double>(allocations) / static_cast<double>(it);
    m_State.counters["DellocN"] =
        static_cast<double>(deallocations) / static_cast<double>(it);
    m_State.counters["AllocBytes"] =
        static_cast<double>(bytes) / static_cast<double>(it);
  }
};

} // namespace

// Use the C allocator to avoid recursing into these allocation hooks.
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

// Google Benchmark intentionally leaves the range-for variable unused.
// NOLINTBEGIN(clang-analyzer-deadcode.DeadStores)

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
      dummy_out = 0;
      grad.execute(x.data(), n, work, dx.data());
      benchmark::DoNotOptimize(dx.data());
      benchmark::DoNotOptimize(dummy_out);
    }
  }
}
BENCHMARK(BM_InterprocPassive)->RangeMultiplier(5)->Range(100, 5000);

static void BM_DeadStructField(benchmark::State& state) {
  const auto n = static_cast<int>(state.range(0));
  std::vector<double> x(n);
  std::vector<double> dx(n, 0.0);
  for (int i = 0; i < n; ++i)
    x[i] = 0.5 + (0.001 * static_cast<double>(i));
  auto grad = clad::gradient(dead_struct_field, "x");
  {
    AddBMCounterRAII c(*mm, state);
    for (auto _ : state) {
      for (double& d : dx)
        d = 0;
      dummy_out = 0;
      grad.execute(x.data(), n, dx.data());
      benchmark::DoNotOptimize(dx.data());
      benchmark::DoNotOptimize(dummy_out);
    }
  }
}
BENCHMARK(BM_DeadStructField)->RangeMultiplier(4)->Range(64, 4096);

static void BM_DeadArrayChain(benchmark::State& state) {
  const auto n = static_cast<int>(state.range(0));
  if (n < 1 || n > kMaxN) {
    state.SkipWithError("n must be in [1, kMaxN]");
    return;
  }
  std::vector<double> x(n);
  std::vector<double> dx(n, 0.0);
  for (int i = 0; i < n; ++i)
    x[i] = 0.5 + (0.001 * static_cast<double>(i));
  auto grad = clad::gradient(dead_array_chain, "x");
  {
    AddBMCounterRAII c(*mm, state);
    for (auto _ : state) {
      for (double& d : dx)
        d = 0;
      dummy_out = 0;
      grad.execute(x.data(), n, dx.data());
      benchmark::DoNotOptimize(dx.data());
      benchmark::DoNotOptimize(dummy_out);
    }
  }
}
BENCHMARK(BM_DeadArrayChain)->RangeMultiplier(10)->Range(1000, 10000);

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
      dummy_out = 0;
      grad.execute(x.data(), n, /*flag=*/1, dx.data());
      benchmark::DoNotOptimize(dx.data());
      benchmark::DoNotOptimize(dummy_out);
    }
  }
}
BENCHMARK(BM_BranchGuarded)->RangeMultiplier(4)->Range(64, 4096);

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

// NOLINTEND(clang-analyzer-deadcode.DeadStores)

BENCHMARK_MAIN();
