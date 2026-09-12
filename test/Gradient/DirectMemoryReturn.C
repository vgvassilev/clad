// RUN: %cladclang %s -I%S/../../include -o %t 2>&1 | %filecheck %s
// RUN: %t | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"

extern "C" int printf(const char*, ...);

struct MemoryValue {
  double value;
  double* storage;
};

struct FallbackMemoryValue {
  double value;
  double* storage;
};

int primalCalls = 0;
int zeroLikeCalls = 0;

MemoryValue make_memory_value(double input) {
  ++primalCalls;
  return {input * input, nullptr};
}

FallbackMemoryValue make_fallback_memory_value(double input) {
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

void make_fallback_memory_value_pullback(double input,
                                         FallbackMemoryValue d_output,
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

double fallback_memory_loss(double input) {
  double result = 0.0;
  for (int i = 0; i < 2; ++i)
    result += make_fallback_memory_value(input).value;
  return result;
}

double nonactive_memory_loss(double input) {
  return make_memory_value(1.0).value + input * 0.0;
}

// CHECK: clad::ValueAndAdjoint<MemoryValue, MemoryValue> direct_memory_value_reverse_forw(double input, double _d_input) {
// CHECK-NEXT:     MemoryValue _t0 = make_memory_value(input);
// CHECK-NEXT:     MemoryValue _r0 = clad::zero_like(_t0);
// CHECK-NEXT:     return {{.*}}_t0{{.*}}_r0{{.*}};
// CHECK: }
// CHECK: void nonactive_memory_loss_grad(double input, double *_d_input) {
// CHECK-NEXT:     {
// CHECK-NEXT:         MemoryValue _r0 = {0., nullptr};

int main() {
  auto gradient = clad::gradient(memory_loss);
  double d_input = 0.0;
  gradient.execute(3.0, &d_input);
  printf("%.1f\n", d_input); // CHECK-EXEC: 6.0
  printf("%d %d\n", primalCalls,
         zeroLikeCalls); // CHECK-EXEC-NEXT: 2 2

  double fallbackDerivative = 0.0;
  auto fallbackGradient = clad::gradient(fallback_memory_loss);
  fallbackGradient.execute(3.0, &fallbackDerivative);
  printf("%.1f\n", fallbackDerivative); // CHECK-EXEC-NEXT: 12.0

  double nonactiveDerivative = 0.0;
  auto nonactiveGradient = clad::gradient(nonactive_memory_loss);
  nonactiveGradient.execute(3.0, &nonactiveDerivative);
  printf("%.1f\n", nonactiveDerivative); // CHECK-EXEC-NEXT: 0.0
}
