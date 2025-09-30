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

template <typename T, std::size_t SBO_SIZE = 64, std::size_t SLAB_SIZE = 1024>
void func(clad::tape<T, SBO_SIZE, SLAB_SIZE>& t, T x, int n) {
  for (int i = 0; i < n; i++)
    clad::push<T, SBO_SIZE, SLAB_SIZE>(t, x);

  for (int i = 0; i < n; i++)
    clad::pop<T, SBO_SIZE, SLAB_SIZE>(t);
}

static void BM_TapeMemory(benchmark::State& state) {
  int block = state.range(0);
  AddBMCounterRAII MemCounters(*mm.get(), state);
  for (auto _ : state) {
    clad::tape<double> t;
    func<double>(t, 1, block * 2 + 1);
  }
}
BENCHMARK(BM_TapeMemory)->RangeMultiplier(2)->Range(0, 4096);

template <std::size_t SBO_SIZE, std::size_t SLAB_SIZE>
static void BM_TapeMemory_Templated(benchmark::State& state) {
  int block = state.range(0);
  AddBMCounterRAII MemCounters(*mm.get(), state);
  for (auto _ : state) {
    clad::tape<double, SBO_SIZE, SLAB_SIZE> t;
    func<double, SBO_SIZE, SLAB_SIZE>(t, 1, block * 2 + 1);
  }
}

#define REGISTER_TAPE_BENCHMARK(sbo, slab)                                     \
  BENCHMARK_TEMPLATE(BM_TapeMemory_Templated, sbo, slab)                       \
      ->RangeMultiplier(2)                                                     \
      ->Range(0, 4096)                                                         \
      ->Name("BM_TapeMemory/SBO_" #sbo "_SLAB_" #slab)

REGISTER_TAPE_BENCHMARK(64, 1024);
REGISTER_TAPE_BENCHMARK(32, 512);

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
BENCHMARK(BM_ReverseGausMemoryP)->RangeMultiplier(2)->Range(0, 4096);

// Define our main.
BENCHMARK_MAIN();
