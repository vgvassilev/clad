#include "benchmark/benchmark.h"

#include "clad/Differentiator/Differentiator.h"

namespace {
  struct MemoryManager : public benchmark::MemoryManager {
    size_t cur_num_allocs = 0;
    size_t cur_num_deallocs = 0;
    size_t cur_max_bytes_used = 0;
    void Start() override {
      cur_num_allocs = 0;
      cur_num_deallocs = 0;
      cur_max_bytes_used = 0;
    }
    void Stop(Result* result) override {
      result->num_allocs = cur_num_allocs;
      result->max_bytes_used = cur_max_bytes_used;
    }
  };
  static auto mm = std::unique_ptr<MemoryManager>(new MemoryManager());
  static struct InstrumentationRegistrer {
    InstrumentationRegistrer() { benchmark::RegisterMemoryManager(mm.get()); }
    ~InstrumentationRegistrer() { benchmark::RegisterMemoryManager(nullptr); }
  } __mem_mgr_register;

  class AddBMCounterRAII {
    MemoryManager& MemMgr;
    benchmark::State& State;

  public:
    AddBMCounterRAII(MemoryManager& mm, benchmark::State& state)
        : MemMgr(mm), State(state) {
      mm.cur_num_allocs = 0;
      mm.cur_max_bytes_used = 0;
    }
    ~AddBMCounterRAII() { pop(); }

    void pop() {
      State.counters["AllocN"] = MemMgr.cur_num_allocs;
      State.counters["DellocN"] = MemMgr.cur_num_deallocs;
      State.counters["AllocBytes"] = MemMgr.cur_max_bytes_used;
    }
  };
} // namespace

void* operator new(size_t size) {
  if (mm) {
    mm->cur_num_allocs++;
    mm->cur_max_bytes_used += size;
  }
  void* p = malloc(size);
  return p;
}

void operator delete(void* p) noexcept {
  if (mm)
    mm->cur_num_deallocs++;
  free(p);
}

template <typename T> void func(clad::tape<T>& t, T x, int n) {
  for (int i = 0; i < n; i++)
    clad::push<T>(t, x);

  for (int i = 0; i < n; i++)
    clad::pop<T>(t);
}

static void BM_TapeMemory(benchmark::State& state) {
  int block = state.range(0);
  AddBMCounterRAII MemCounters(*mm.get(), state);
  clad::tape<double> t;
  for (auto _ : state) {
    func<double>(t, 1, block * 2 + 1);
  }
}
BENCHMARK(BM_TapeMemory)->RangeMultiplier(2)->Range(0, 4096)->Iterations(1);

#include "BenchmarkedFunctions.h"

static void BM_ReverseGausMemoryP(benchmark::State& state) {
  auto dfdp_grad = clad::gradient(gaus, "p");
  unsigned dim = state.range(0);
  double* x = new double[dim]();
  double* p = new double[dim]();
  double* result = new double[dim]();
  for (unsigned i = 0; i < dim; i++) {
    result[i] = 0;
    x[i] = 1;
    p[i] = i;
  }
  AddBMCounterRAII MemCounters(*mm.get(), state);
  for (auto _ : state) {
    dfdp_grad.execute(x, p, /*sigma*/ 2, dim, result);
  }
}
BENCHMARK(BM_ReverseGausMemoryP)
    ->RangeMultiplier(2)
    ->Range(0, 4096)
    ->Iterations(1);

// Define our main.
BENCHMARK_MAIN();
