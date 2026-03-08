#include "benchmark/benchmark.h"

#include "clad/Differentiator/Differentiator.h"
#include "clad/Differentiator/Tape.h"
#include <cstddef>
#include <cstdint>

extern "C" {
void pushReal8(double x);
void popReal8(double* x);
}
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
    void Stop(Result& result) override {
      result.num_allocs = cur_num_allocs;
      result.max_bytes_used = cur_max_bytes_used;
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

template <typename T, std::size_t SBO_SIZE = 64, std::size_t SLAB_SIZE = 1024,
          bool DiskOffload = false>
void func(clad::tape_impl<T, SBO_SIZE, SLAB_SIZE, /*is_Multithread=*/false,
                          DiskOffload>& t,
          T x, int n) {
  for (int i = 0; i < n; i++)
    clad::push(t, x);

  for (int i = 0; i < n; i++) {
    benchmark::DoNotOptimize(t.back());
    t.pop_back();
  }
}

static void BM_TapeMemory(benchmark::State& state) {
  int block = state.range(0);
  AddBMCounterRAII MemCounters(*mm.get(), state);
  for (auto _ : state) {
    // Explicitly using false for DiskOffload to test baseline
    clad::tape_impl<double, 64, 1024, /*is_Multithread=*/false,
                    /*DiskOffload=*/false>
        t;
    func<double, 64, 1024, /*DiskOffload=*/false>(t, 1, block * 2 + 1);
  }
}
BENCHMARK(BM_TapeMemory)->RangeMultiplier(2)->Range(0, 4096);

template <std::size_t SBO_SIZE, std::size_t SLAB_SIZE>
static void BM_TapeMemory_Templated(benchmark::State& state) {
  int block = state.range(0);
  AddBMCounterRAII MemCounters(*mm.get(), state);
  for (auto _ : state) {
    clad::tape_impl<double, SBO_SIZE, SLAB_SIZE, /*is_Multithread=*/false,
                    /*DiskOffload=*/false>
        t;
    func<double, SBO_SIZE, SLAB_SIZE, /*DiskOffload=*/false>(t, 1,
                                                             block * 2 + 1);
  }
}

#define REGISTER_TAPE_BENCHMARK(sbo, slab)                                     \
  BENCHMARK_TEMPLATE(BM_TapeMemory_Templated, sbo, slab)                       \
      ->RangeMultiplier(2)                                                     \
      ->Range(0, 4096)                                                         \
      ->Name("BM_TapeMemory/SBO_" #sbo "_SLAB_" #slab)

REGISTER_TAPE_BENCHMARK(64, 1024);
REGISTER_TAPE_BENCHMARK(32, 512);

// This explicitly tests the case where DiskOffload = true
template <std::size_t SBO_SIZE, std::size_t SLAB_SIZE>
static void BM_Multilayer_Storage(benchmark::State& state) {
  int64_t block = state.range(0);
  AddBMCounterRAII MemCounters(*mm, state);
  for (auto _ : state) {
    // Set DiskOffload = true here
    clad::tape_impl<double, SBO_SIZE, SLAB_SIZE, /*is_Multithread=*/false,
                    /*DiskOffload=*/true>
        t;
    func<double, SBO_SIZE, SLAB_SIZE, /*DiskOffload=*/true>(t, 1,
                                                            block * 2 + 1);
  }
}

BENCHMARK_TEMPLATE(BM_Multilayer_Storage, 64, 1024)
    ->RangeMultiplier(2)
    ->Range(0, 4096)
    ->Name("BM_Multilayer_Storage/SBO_64_SLAB_1024_DISK");

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

const size_t TARGET_ELEMENTS = 20000;

static void BM_CrashTest_OS_Paging(benchmark::State& state) {
  AddBMCounterRAII MemCounters(*mm, state);
  for (auto _ : state) {
    clad::tape_impl<double, 64, 1024, /*is_Multithread=*/false,
                    /*DiskOffload=*/false>
        t;

    for (size_t i = 0; i < TARGET_ELEMENTS; ++i) {
      try {
        clad::push(t, 1.0);
      } catch (std::bad_alloc& e) {
        state.SkipWithError("OS ran out of memory!");
        break;
      }
    }
  }
}

BENCHMARK(BM_CrashTest_OS_Paging)->Iterations(1);

static void BM_CrashTest_Clad_Offload(benchmark::State& state) {
  AddBMCounterRAII MemCounters(*mm, state);
  for (auto _ : state) {
    clad::tape_impl<double, 64, 1310720, /*is_Multithread=*/false,
                    /*DiskOffload=*/true>
        t;

    for (size_t i = 0; i < TARGET_ELEMENTS; ++i)
      clad::push(t, 1.0);
  }
}
BENCHMARK(BM_CrashTest_Clad_Offload)->Iterations(1);

// Tapenade Comparison Benchmarks
struct Tapenade {
  void push(double val) { pushReal8(val); }
  void pop(double* val) { popReal8(val); }
};
struct CladDisk {
  clad::tape_impl<double, 64, 1024, /*is_multithread=*/false,
                  /*DiskOffload=*/true>
      tape;

  void push(double val) { tape.emplace_back(val); }
  void pop(double* val) {
    *val = tape.back();
    tape.pop_back();
  }
};
template <typename Strategy> static void BM_PushPop(benchmark::State& state) {
  int64_t n = state.range(0);
  Strategy s; // Instantiate the specific strategy (Clad or Tapenade)

  for (auto _ : state) {
    for (int64_t i = 0; i < n; ++i)
      s.push(1.0);
    for (int64_t i = 0; i < n; ++i) {
      double val;
      s.pop(&val);
      benchmark::DoNotOptimize(val);
    }
  }
}
BENCHMARK_TEMPLATE(BM_PushPop, Tapenade)
    ->RangeMultiplier(4)
    ->Range(1024, 262144)
    ->Name("BM_PushPop/TapenadeStack");

BENCHMARK_TEMPLATE(BM_PushPop, CladDisk)
    ->RangeMultiplier(4)
    ->Range(1024, 262144)
    ->Name("BM_PushPop/CladDiskStack");

BENCHMARK_MAIN();