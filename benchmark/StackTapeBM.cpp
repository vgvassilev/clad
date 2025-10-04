#include "benchmark/benchmark.h"

#include "StackTape.h"

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
    mm.cur_num_deallocs = 0;
    mm.cur_max_bytes_used = 0;
  }
  ~AddBMCounterRAII() { pop(); }

  void pop() {
    const unsigned long iterations = State.iterations();
    State.counters["AllocN"] = MemMgr.cur_num_allocs / iterations;
    State.counters["DellocN"] = MemMgr.cur_num_deallocs / iterations;
    State.counters["AllocBytes"] = MemMgr.cur_max_bytes_used / iterations;
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

template <typename T> void func(clad::stack_tape::tape<T>& t, T x, int n) {
  for (int i = 0; i < n; i++)
    clad::stack_tape::push<T>(t, x);

  for (int i = 0; i < n; i++)
    benchmark::DoNotOptimize(clad::stack_tape::pop<T>(t));
}

static void BM_TapeMemory(benchmark::State& state) {
  int block = state.range(0);
  AddBMCounterRAII MemCounters(*mm.get(), state);
  for (auto _ : state) {
    clad::stack_tape::tape<double> t;
    func<double>(t, 1, block * 2 + 1);
  }
}
BENCHMARK(BM_TapeMemory)->RangeMultiplier(2)->Range(0, 12000);

// Define our main.
BENCHMARK_MAIN();
