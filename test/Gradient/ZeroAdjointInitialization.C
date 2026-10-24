// RUN: %cladclang %s -I%S/../../include -Xclang -verify -o %t 2>&1 | %filecheck %s
// RUN: %t | %filecheck_exec %s
// RUN: %cladclang -std=c++14 -Wno-c++17-extensions %s -I%S/../../include -Xclang -verify -o %t14
// RUN: %t14 | %filecheck_exec %s
// RUN: %cladclang -std=c++20 %s -I%S/../../include -Xclang -verify -o %t20
// RUN: %t20 | %filecheck_exec %s
// RUN: %cladclang %s -I%S/../../include -Xclang -verify -Xclang -plugin-arg-clad -Xclang -disable-tbr -o %t.no-tbr
// RUN: %t.no-tbr | %filecheck_exec %s
// RUN: %cladclang %s -I%S/../../include -Xclang -verify -Xclang -plugin-arg-clad -Xclang -enable-va -o %t.va
// RUN: %t.va | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"

extern "C" int printf(const char*, ...);

// A zero adjoint needs runtime structure and cannot be constructed with {}.
struct Config {
  double factor;
  int extent;
  explicit Config(double f, int n = 7) : factor(f), extent(n) {}
  Config* operator&() { return this; }
  const Config* operator&() const = delete;
};

const Config config(2.);
struct AggregateConfig {
  double factor;
  int extent;
};
const AggregateConfig aggregateConfig{2., 11};
int touches = 0;
int badZeroInputs = 0;
int badSeeds = 0;

__attribute__((annotate("non_differentiable"))) void bump() { ++touches; }

__attribute__((annotate("non_differentiable"))) AggregateConfig
operator+(const AggregateConfig& value, int) {
  return value;
}

namespace clad {
// The zero_like argument becomes an lvalue after storing the primal xvalue.
Config zero_like(const Config&&) = delete;
Config zero_like(const Config& value) {
  if (clad_addressof(value) != clad_addressof(config))
    ++badZeroInputs;
  return Config(0., value.extent);
}

AggregateConfig zero_like(const AggregateConfig& value) {
  if (value.extent != 11)
    ++badZeroInputs;
  return {0., value.extent};
}
} // namespace clad

struct Result {
  double value = 0.;
  Result() = default;
  Result(double x, Config c) : value(x * c.factor) {}
};

namespace clad {
namespace custom_derivatives {
namespace class_functions {
ValueAndAdjoint<Result, Result>
constructor_reverse_forw(Tag<Result>, double x, Config c, double, Config dc) {
  if (dc.factor != 0. || dc.extent != c.extent)
    ++badSeeds;
  return {Result(x, c), Result(0., c)};
}

void constructor_pullback(double, Config c, Result* dr, double* dx, Config*) {
  *dx += c.factor * dr->value;
}
} // namespace class_functions
} // namespace custom_derivatives
} // namespace clad

double global_config(double x) {
  Result r(x, config);
  return r.value * r.value;
}

// CHECK-LABEL: void global_config_grad(double x, double *_d_x) {
// CHECK: constructor_reverse_forw(clad::Tag<Result>(), x, config, 0., clad::zero_like(config))
// CHECK: Config _r{{[0-9]+}} = clad::zero_like(config);

double evaluated_config(double x) {
  Result r(x, (bump(), config));
  return r.value * r.value;
}

double loop_config(double x) {
  double result = 0.;
  for (int i = 0; i < 3; ++i) {
    Result r(x, (bump(), config));
    result += r.value * r.value;
  }
  return result;
}

double branch_config(double x) {
  if (x > 0) {
    Result r(x, (bump(), config));
    return r.value * r.value;
  }
  return x;
}

double xvalue_config(double x) {
  Result r(x, static_cast<const Config&&>((bump(), config)));
  return r.value * r.value;
}

double* local_aggregate_config(double* p) {
  AggregateConfig c = aggregateConfig + 0;
  if (c.extent != 11)
    __builtin_trap();
  return p;
}

// CHECK-LABEL: local_aggregate_config_reverse_forw(
// CHECK: AggregateConfig c = aggregateConfig + 0;
// CHECK-NEXT: AggregateConfig _d_c = clad::zero_like(c);

double aggregate_config(double x) { return *local_aggregate_config(&x); }

int main() {
  double dx = 0.;
  clad::gradient(global_config).execute(3., &dx);
  printf("global %.1f\n", dx); // CHECK-EXEC: global 24.0

  dx = 0.;
  touches = 0;
  clad::gradient(evaluated_config).execute(3., &dx);
  printf("evaluated %.1f %d\n", dx, touches); // CHECK-EXEC-NEXT: evaluated 24.0 1

  dx = 0.;
  touches = 0;
  clad::gradient(loop_config).execute(3., &dx);
  printf("loop %.1f %d\n", dx, touches); // CHECK-EXEC-NEXT: loop 72.0 3

  dx = 0.;
  touches = 0;
  auto branch = clad::gradient(branch_config);
  branch.execute(3., &dx);
  printf("branch %.1f %d\n", dx, touches); // CHECK-EXEC-NEXT: branch 24.0 1
  dx = 0.;
  touches = 0;
  branch.execute(-1., &dx);
  printf("untaken %.1f %d\n", dx, touches); // CHECK-EXEC-NEXT: untaken 1.0 0

  dx = 0.;
  touches = 0;
  clad::gradient(xvalue_config).execute(3., &dx);
  printf("xvalue %.1f %d\n", dx, touches); // CHECK-EXEC-NEXT: xvalue 24.0 1

  dx = 0.;
  clad::gradient(aggregate_config).execute(3., &dx);
  printf("aggregate %.1f\n", dx); // CHECK-EXEC-NEXT: aggregate 1.0
  printf("invalid %d %d\n", badZeroInputs, badSeeds); // CHECK-EXEC-NEXT: invalid 0 0
}

// expected-no-diagnostics
