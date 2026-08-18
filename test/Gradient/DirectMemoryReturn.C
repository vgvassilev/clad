// RUN: %cladclang %s -I%S/../../include -o %t 2>&1 | %filecheck %s
// RUN: %t | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"

#include <cstdio>

struct MemoryValue {
  double value;
  double* storage;
};

int primalCalls = 0;
int zeroLikeCalls = 0;

MemoryValue make_memory_value(double input) {
  ++primalCalls;
  return {input * input, nullptr};
}

namespace clad {

MemoryValue zero_like(const MemoryValue& /*value*/) {
  ++zeroLikeCalls;
  return {0.0, nullptr};
}

} // namespace clad

namespace clad::custom_derivatives {

void make_memory_value_pullback(double input, MemoryValue d_output,
                                double* d_input) {
  *d_input += 2.0 * input * d_output.value;
}

} // namespace clad::custom_derivatives

MemoryValue direct_memory_value(double input) {
  return make_memory_value(input);
}

double memory_loss(double input) {
  auto result = direct_memory_value(input);
  return result.value;
}

// CHECK: clad::ValueAndAdjoint<MemoryValue, MemoryValue> direct_memory_value_reverse_forw(double input, double _d_input) {
// CHECK-NEXT:     MemoryValue _t0 = make_memory_value(input);
// CHECK-NEXT:     MemoryValue _r0 = clad::zero_like(_t0);
// CHECK-NEXT:     return {{.*}}_t0{{.*}}_r0{{.*}};
// CHECK: }

int main() {
  auto gradient = clad::gradient(memory_loss);
  double d_input = 0.0;
  gradient.execute(3.0, &d_input);
  std::printf("%.1f\n", d_input); // CHECK-EXEC: 6.0
  std::printf("%d %d\n", primalCalls,
              zeroLikeCalls); // CHECK-EXEC-NEXT: 2 2
}
