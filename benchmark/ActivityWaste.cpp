#include "clad/Differentiator/Differentiator.h"
#include <cstddef>
#include <memory>
#include "benchmark/benchmark.h"
double dummy_out = 0; // Global to prevent Clang constant folding

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
auto mm = std::make_unique<MemoryManager>();
struct InstrumentationRegistrer {
  InstrumentationRegistrer() { benchmark::RegisterMemoryManager(mm.get()); }
  ~InstrumentationRegistrer() { benchmark::RegisterMemoryManager(nullptr); }
} MemMgrRegister;

class AddBMCounterRAII {
  MemoryManager& MemMgr;
  benchmark::State& State;

public:
  AddBMCounterRAII(MemoryManager& m, benchmark::State& s)
      : MemMgr(m), State(s) {
    m.cur_num_allocs = 0;
    m.cur_num_deallocs = 0;
    m.cur_max_bytes_used = 0;
  }
  ~AddBMCounterRAII() {
    const unsigned long it = State.iterations();
    State.counters["AllocN"] = MemMgr.cur_num_allocs / it;
    State.counters["DellocN"] = MemMgr.cur_num_deallocs / it;
    State.counters["AllocBytes"] = MemMgr.cur_max_bytes_used / it;
  }
};
} // namespace

void* operator new(size_t size) {
  if (mm) {
    mm->cur_num_allocs++;
    mm->cur_max_bytes_used += size;
  }
  return malloc(size);
}
void operator delete(void* p) noexcept {
  if (mm)
    mm->cur_num_deallocs++;
  free(p);
}

double positive_control(const double* x, int n) {
  double r = 0;
  for (int i = 0; i < n; ++i) {
    double t = x[i] * x[i];
    r += t * t;
  }
  return r;
}

static void BM_PositiveControl(benchmark::State& state) {
  int n = state.range(0);
  double* x = new double[n];
  double* dx = new double[n];
  for (int i = 0; i < n; ++i)
    x[i] = 0.5;
  auto grad = clad::gradient(positive_control, "x");
  {
    AddBMCounterRAII c(*mm.get(), state);
    for (auto _ : state) {
      for (int i = 0; i < n; ++i)
        dx[i] = 0;
      grad.execute(x, n, dx);
      benchmark::DoNotOptimize(dx);
    }
  }
  delete[] x;
  delete[] dx;
}
BENCHMARK(BM_PositiveControl)->RangeMultiplier(4)->Range(64, 4096);

__attribute__((noinline)) double inner_accumulate(double x, int work) {
  double s = 1.0;
  for (int i = 0; i < work; ++i)
    s = 0.25 * s * s + 0.5; // bounded in (0.5, 0.75); never folds or overflows
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
  const int n = 50;
  int work = state.range(0);
  double x[n], dx[n];
  for (int i = 0; i < n; ++i) {
    x[i] = 0.5;
    dx[i] = 0;
  }
  auto grad = clad::gradient(interproc_passive, "x");
  AddBMCounterRAII c(*mm.get(), state);
  for (auto _ : state) {
    for (int i = 0; i < n; ++i)
      dx[i] = 0;
    grad.execute(x, n, work, dx);
    benchmark::DoNotOptimize(dx);
  }
}
BENCHMARK(BM_InterprocPassive)->RangeMultiplier(5)->Range(100, 5000);

struct Particle {
  double active1, active2, dead;
};
double dead_struct_field(const double* x, int n) {
  double r = 0;
  for (int i = 0; i < n; ++i) {
    Particle p;
    p.active1 = x[i];
    p.active2 = x[i] * x[i];
    p.dead = 1.0;
    for (int j = 0; j < 2; ++j)
      p.dead = p.dead * p.dead + x[i];
    r += p.active1 + p.active2;
    dummy_out += p.dead;
  }
  return r;
}

static void BM_DeadStructField(benchmark::State& state) {
  int n = state.range(0);
  double* x = new double[n];
  double* dx = new double[n];
  for (int i = 0; i < n; ++i) {
    x[i] = 0.5 + 0.001 * i;
    dx[i] = 0;
  }
  auto grad = clad::gradient(dead_struct_field, "x");
  {
    AddBMCounterRAII c(*mm.get(), state);
    for (auto _ : state) {
      for (int i = 0; i < n; ++i)
        dx[i] = 0;
      grad.execute(x, n, dx);
      benchmark::DoNotOptimize(dx);
    }
  }
  delete[] x;
  delete[] dx;
}
BENCHMARK(BM_DeadStructField)->RangeMultiplier(4)->Range(64, 4096);

double dead_array_chain(const double* x, double* scratch, int n) {
  for (int i = 0; i < n; ++i)
    scratch[i] = x[i] * x[i];
  for (int k = 0; k < n - 1; ++k)
    scratch[k] = scratch[k + 1] * scratch[k + 1];
  double ret = x[0] * x[0];
  dummy_out += scratch[0];
  return ret;
}

static void BM_DeadArrayChain(benchmark::State& state) {
  int n = state.range(0);
  double* x = new double[n];
  double* dx = new double[n];
  double* scratch = new double[n];
  double* dscratch = new double[n];
  for (int i = 0; i < n; ++i) {
    x[i] = 0.5 + 0.001 * i;
    dx[i] = 0;
    scratch[i] = 0;
    dscratch[i] = 0;
  }
  auto grad = clad::gradient(dead_array_chain, "x, scratch");
  {
    AddBMCounterRAII c(*mm.get(), state);
    for (auto _ : state) {
      for (int i = 0; i < n; ++i) {
        dx[i] = 0;
        dscratch[i] = 0;
      }
      grad.execute(x, scratch, n, dx, dscratch);
      benchmark::DoNotOptimize(dx);
    }
  }
  delete[] x;
  delete[] dx;
  delete[] scratch;
  delete[] dscratch;
}
BENCHMARK(BM_DeadArrayChain)->RangeMultiplier(10)->Range(1000, 10000);

double branch_guarded(const double* x, int n, int flag) {
  double r = x[0] * x[0];
  if (flag) {
    double t = 0;
    for (int i = 0; i < n; ++i)
      t = t * t + x[i];
    dummy_out += t;
  }
  return r;
}

static void BM_BranchGuarded(benchmark::State& state) {
  int n = state.range(0);
  double* x = new double[n];
  double* dx = new double[n];
  for (int i = 0; i < n; ++i) {
    x[i] = 0.5;
    dx[i] = 0;
  }
  auto grad = clad::gradient(branch_guarded, "x");
  {
    AddBMCounterRAII c(*mm.get(), state);
    for (auto _ : state) {
      for (int i = 0; i < n; ++i)
        dx[i] = 0;
      grad.execute(x, n, /*flag=*/1, dx); // executes and scales with n
      benchmark::DoNotOptimize(dx);
    }
  }
  delete[] x;
  delete[] dx;
}
BENCHMARK(BM_BranchGuarded)->RangeMultiplier(4)->Range(64, 4096);

double useful_not_varied(double x, int reps) {
  double x1 = 0.5, x2 = 0.5, x3 = 0.5, x4 = 0.5, x5 = 0.5;
  for (int i = 0; i < reps; ++i) {
    x5 = x4 * x4;
    x4 = x3 * x3;
    x3 = x2 * x2;
    x2 = x1 * x1;
  }
  return x + x5;
}

static void BM_UsefulNotVaried(benchmark::State& state) {
  int reps = state.range(0);
  auto grad = clad::gradient(useful_not_varied, "x");
  AddBMCounterRAII c(*mm.get(), state);
  for (auto _ : state) {
    double dx = 0;
    grad.execute(0.5, reps, &dx);
    benchmark::DoNotOptimize(dx);
  }
}
BENCHMARK(BM_UsefulNotVaried)->RangeMultiplier(10)->Range(100, 10000);

BENCHMARK_MAIN();
