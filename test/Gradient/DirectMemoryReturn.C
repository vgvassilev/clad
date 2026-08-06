// RUN: %cladclang %s -I%S/../../include -o %t 2>&1 | %filecheck %s
// RUN: %t | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"

#include <cstdio>

struct MemoryValue {
  double value;
  double* storage;
};

MemoryValue make_memory_value(double input) {
  return {input * input, nullptr};
}

namespace clad::custom_derivatives {

clad::ValueAndAdjoint<MemoryValue, MemoryValue>
make_memory_value_reverse_forw(double input,
                               [[maybe_unused]] double d_input) {
  return {make_memory_value(input), {0.0, nullptr}};
}

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
// CHECK:     clad::ValueAndAdjoint<MemoryValue, MemoryValue> _t0 = clad::custom_derivatives::make_memory_value_reverse_forw(input, 0.);
// CHECK:     return {_t0.value, _t0.adjoint};
// CHECK: }

int main() {
  auto gradient = clad::gradient(memory_loss);
  double d_input = 0.0;
  gradient.execute(3.0, &d_input);
  std::printf("%.1f\n", d_input); // CHECK-EXEC: 6.0
}
