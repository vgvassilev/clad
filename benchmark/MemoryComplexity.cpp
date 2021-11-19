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
  mm->cur_num_allocs = 0;
  mm->cur_max_bytes_used = 0;
  clad::tape<double> t;
  for (auto _ : state) {
    func<double>(t, 1, block * 2 + 1);
  }
  state.counters["AllocN"] = mm->cur_num_allocs;
  state.counters["DellocN"] = mm->cur_num_deallocs;
  state.counters["AllocBytes"] = mm->cur_max_bytes_used;
}

BENCHMARK(BM_TapeMemory)->RangeMultiplier(2)->Range(0, 4096)->Iterations(1);

// Define our main.
BENCHMARK_MAIN();
