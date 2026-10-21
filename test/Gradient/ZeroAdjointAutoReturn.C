// RUN: %cladclang %s -I%S/../../include -Xclang -verify -o %t
// RUN: %t | %filecheck_exec %s
// RUN: %cladclang -std=c++14 -Wno-c++17-extensions %s -I%S/../../include -Xclang -verify -o %t14
// RUN: %t14 | %filecheck_exec %s
// RUN: %cladclang -std=c++20 %s -I%S/../../include -Xclang -verify -o %t20
// RUN: %t20 | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"

extern "C" int printf(const char*, ...);

struct MemoryValue {
  double value;
  double* storage;
  int extent;
};

int zeroLikeCalls = 0;
int badSeeds = 0;

double read_memory_value(const MemoryValue& value) { return value.value; }

namespace clad {
// The template's return type has not been deduced before Clad finds it.
template <typename T> auto zero_like(const T& value) {
  ++zeroLikeCalls;
  return MemoryValue{0., nullptr, value.extent};
}

namespace custom_derivatives {
void read_memory_value_pullback(const MemoryValue&, double d_output,
                                MemoryValue* d_value) {
  if (d_value->extent != 11)
    ++badSeeds;
  d_value->value += d_output;
}
} // namespace custom_derivatives
} // namespace clad

double loss(MemoryValue value, double x) {
  value.value = x * x;
  return read_memory_value(value);
}

int main() {
  double d_x = 0.;
  clad::gradient(loss, "x").execute(MemoryValue{0., nullptr, 11}, 3., &d_x);
  printf("%.1f %d %d\n", d_x, zeroLikeCalls, badSeeds);
  // CHECK-EXEC: 6.0 1 0
}

// expected-no-diagnostics
