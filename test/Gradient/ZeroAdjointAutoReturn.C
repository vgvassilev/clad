// RUN: %cladclang %s -I%S/../../include -Xclang -verify -o %t
// RUN: %t | %filecheck_exec %s
// RUN: %cladclang -std=c++14 -Wno-c++17-extensions %s -I%S/../../include -Xclang -verify -o %t14
// RUN: %t14 | %filecheck_exec %s
// RUN: %cladclang -std=c++20 %s -I%S/../../include -Xclang -verify -o %t20
// RUN: %t20 | %filecheck_exec %s
// RUN: %cladclang -DMISSING_ZERO_LIKE_DEFINITION %s -I%S/../../include -Xclang -verify -o %t.missing
// RUN: %t.missing | %filecheck %s --check-prefix=CHECK-FALLBACK

#include "clad/Differentiator/Differentiator.h"

extern "C" int printf(const char*, ...);

struct MemoryValue {
  double value;
  double* storage;
  int extent;
};

int zeroLikeCalls = 0;
int seedExtent = -1;

double read_memory_value(const MemoryValue& value) { return value.value; }

namespace clad {
#ifdef MISSING_ZERO_LIKE_DEFINITION
// A declaration alone cannot supply a deduced return type. Use zero init.
template <typename T> auto zero_like(const T& value);
#else
// The template's return type has not been deduced before Clad finds it.
template <typename T> auto zero_like(const T& value) {
  ++zeroLikeCalls;
  return MemoryValue{0., nullptr, value.extent};
}
#endif

namespace custom_derivatives {
void read_memory_value_pullback(const MemoryValue&, double d_output,
                                MemoryValue* d_value) {
  seedExtent = d_value->extent;
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
  printf("%.1f %d %d\n", d_x, zeroLikeCalls, seedExtent);
  // CHECK-EXEC: 6.0 1 11
  // CHECK-FALLBACK: 6.0 0 0
}

// expected-no-diagnostics
