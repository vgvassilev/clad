#ifndef BENCHMARK_ACTIVITY_WASTE_KERNELS_H
#define BENCHMARK_ACTIVITY_WASTE_KERNELS_H

// Each executable supplies an external sink for work outside the return value.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
extern double dummy_out;
constexpr int kMaxN = 10000;

// Exercise pointer and raw-array differentiation without heap buffer setup.
// NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic,cppcoreguidelines-pro-bounds-constant-array-index)

// Active nonlinear work to check that tape allocations are observed.
[[maybe_unused]] static double positive_control(const double* x, int n) {
  double r = 0;
  for (int i = 0; i < n; ++i) {
    double t = x[i] * x[i];
    r += t * t;
  }
  return r;
}

// The callee recurrence feeds only the sink; the return is x.
static __attribute__((noinline)) double inner_accumulate(double x, int work) {
  double s = 1.0;
  for (int i = 0; i < work; ++i)
    s = (0.25 * s * s) + 0.5;
  dummy_out += s;
  return x;
}

[[maybe_unused]] static double interproc_passive(const double* x, int n,
                                                 int work) {
  double r = 0;
  for (int i = 0; i < n; ++i)
    r += inner_accumulate(x[i], work);
  return r;
}

// Only Active1 and Active2 contribute to the return value.
struct Particle {
  double Active1;
  double Active2;
  double Dead;
};

[[maybe_unused]] static double dead_struct_field(const double* x, int n) {
  double r = 0;
  for (int i = 0; i < n; ++i) {
    // Every field is assigned before use.
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    Particle p;
    p.Active1 = x[i];
    p.Active2 = x[i] * x[i];
    p.Dead = 1.0;
    for (int j = 0; j < 2; ++j)
      p.Dead = (p.Dead * p.Dead) + x[i];
    r += p.Active1 + p.Active2;
    dummy_out += p.Dead;
  }
  return r;
}

// The scratch chain feeds only the sink; the return depends on x[0].
[[maybe_unused]] static double dead_array_chain(const double* x, int n) {
  // Initialize only the used prefix; require 1 <= n <= kMaxN.
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables,cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  double scratch[kMaxN];
  for (int i = 0; i < n; ++i)
    scratch[i] = x[i] * x[i];
  // The positive-size precondition makes these scratch reads initialized.
  for (int k = 0; k < n - 1; ++k)
    // NOLINTNEXTLINE(clang-analyzer-core.UndefinedBinaryOperatorResult)
    scratch[k] = scratch[k + 1] * scratch[k + 1];
  double ret = x[0] * x[0];
  // NOLINTNEXTLINE(clang-analyzer-core.uninitialized.Assign)
  dummy_out += scratch[0];
  return ret;
}

// The guarded recurrence feeds only the sink, not the return value.
[[maybe_unused]] static double branch_guarded(const double* x, int n,
                                              int flag) {
  double r = x[0] * x[0];
  if (flag) {
    double t = 0;
    for (int i = 0; i < n; ++i)
      t = (0.25 * t * t) + (0.5 * x[i]);
    dummy_out += t;
  }
  return r;
}

// The constant chain contributes to the return, but not its derivative.
[[maybe_unused]] static double useful_not_varied(double x, int reps) {
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

// NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic,cppcoreguidelines-pro-bounds-constant-array-index)

#endif
