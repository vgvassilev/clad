// RUN: %cladclang %s -I%S/../../include -o %t 2>&1 | %filecheck %s
// RUN: %t | %filecheck_exec %s
// RUN: %cladclang %s -DDELETED_ZERO_LIKE -I%S/../../include -o %t 2>&1 | %filecheck %s
// RUN: %t | %filecheck_exec %s
// RUN: %cladclang %s -DAMBIGUOUS_ZERO_LIKE -I%S/../../include -o %t 2>&1 | %filecheck %s
// RUN: %t | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"

extern "C" int printf(const char*, ...);

struct MemoryValue {
  double value;
  double* storage;
  int extent;
};

struct FallbackMemoryValue {
  double value;
  double* storage;
};

int primalCalls = 0;
int zeroLikeCalls = 0;
int mutableZeroLikeCalls = 0;

MemoryValue make_memory_value(double input) {
  ++primalCalls;
  return {input * input, nullptr, 1};
}

const MemoryValue make_const_memory_value(double input) {
  return make_memory_value(input);
}

FallbackMemoryValue make_fallback_memory_value(double input) {
  return {input * input, nullptr};
}

MemoryValue operator+(MemoryValue lhs, double rhs) {
  return {lhs.value + rhs, nullptr, lhs.extent};
}

FallbackMemoryValue operator+(FallbackMemoryValue lhs, double rhs) {
  return {lhs.value + rhs, nullptr};
}

MemoryValue explicit_memory_value(double input) {
  return make_memory_value(input);
}

double read_memory_value(const MemoryValue& value) { return value.value; }

namespace clad {

MemoryValue zero_like(const MemoryValue& value) {
  ++zeroLikeCalls;
  return {0.0, nullptr, value.extent};
}

MemoryValue zero_like(MemoryValue& value) {
  ++mutableZeroLikeCalls;
  return zero_like(static_cast<const MemoryValue&>(value));
}

// Overload probing must model the stored lvalue, not the original prvalue.
MemoryValue zero_like(MemoryValue&&) = delete;

// Missing, deleted and ambiguous customizations all keep the same fallback.
#if defined(DELETED_ZERO_LIKE)
FallbackMemoryValue zero_like(const FallbackMemoryValue&) = delete;
#elif defined(AMBIGUOUS_ZERO_LIKE)
FallbackMemoryValue zero_like(const FallbackMemoryValue&, int = 0);
FallbackMemoryValue zero_like(const FallbackMemoryValue&, long = 0);
#endif

} // namespace clad

namespace clad::custom_derivatives {

void make_memory_value_pullback(double input, MemoryValue d_output,
                                double* d_input) {
  *d_input += 2.0 * input * d_output.value;
}

void make_const_memory_value_pullback(double input, MemoryValue d_output,
                                      double* d_input) {
  *d_input += 2.0 * input * d_output.value;
}

void make_fallback_memory_value_pullback(double input,
                                       FallbackMemoryValue d_output,
                                       double* d_input) {
  *d_input += 2.0 * input * d_output.value;
}

void operator_plus_pullback(MemoryValue lhs, double rhs, MemoryValue d_output,
                            MemoryValue* d_lhs, double* d_rhs) {
  d_lhs->value += d_output.value;
  *d_rhs += d_output.value;
}

void operator_plus_pullback(FallbackMemoryValue lhs, double rhs,
                            FallbackMemoryValue d_output,
                            FallbackMemoryValue* d_lhs, double* d_rhs) {
  d_lhs->value += d_output.value;
  *d_rhs += d_output.value;
}

clad::ValueAndAdjoint<MemoryValue, MemoryValue>
explicit_memory_value_reverse_forw(double input, double /*d_input*/) {
  return {explicit_memory_value(input), {0.0, nullptr, 1}};
}

void explicit_memory_value_pullback(double input, MemoryValue d_output,
                                   double* d_input) {
  *d_input += 2.0 * input * d_output.value;
}

void read_memory_value_pullback(const MemoryValue& value, double d_output,
                               MemoryValue* d_value) {
  // The caller must be able to accumulate into a correctly shaped adjoint,
  // even when the value came from a constant factory.
  if (d_value->extent != value.extent)
    __builtin_trap();
  d_value->value += d_output;
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

MemoryValue constant_memory_value(double input) {
  return make_memory_value(1.0);
}

double constant_memory_loss(double input) {
  auto result = constant_memory_value(input);
  return read_memory_value(result);
}

MemoryValue operator_memory_value(double input) {
  return MemoryValue{input, nullptr, 1} + input;
}

double operator_memory_loss(double input) {
  auto result = operator_memory_value(input);
  return result.value;
}

double loop_memory_loss(double input) {
  double result = 0.0;
  for (int i = 0; i < 2; ++i)
    result += make_memory_value(input).value;
  return result;
}

double explicit_memory_loss(double input) {
  auto result = explicit_memory_value(input);
  return result.value;
}

double fallback_operator_loss(double input) {
  return (FallbackMemoryValue{input, nullptr} + input).value;
}

double const_memory_loss(double input) {
  auto value = make_const_memory_value(input);
  return value.value;
}

// CHECK: clad::ValueAndAdjoint<MemoryValue, MemoryValue> direct_memory_value_reverse_forw(double input, double _d_input) {
// CHECK-NEXT:     MemoryValue _t0 = make_memory_value(input);
// CHECK-NEXT:     MemoryValue _r0 = clad::zero_like(_t0);
// CHECK-NEXT:     return {{.*}}_t0{{.*}}_r0{{.*}};
// CHECK: }
// CHECK: void fallback_memory_loss_grad(double input, double *_d_input) {
// CHECK-NEXT:     int _d_i = 0;
// CHECK-NEXT:     int i = 0;
// CHECK-NEXT:     double _d_result = 0.;
// CHECK-NEXT:     double result = 0.;
// CHECK-NEXT:     unsigned {{int|long|long long}} _t0;
// CHECK-NEXT:     for (i = 0; i < 2; ++i) {
// CHECK-NEXT:         result += make_fallback_memory_value(input).value;
// CHECK-NEXT:     }
// CHECK-NEXT:     _d_result += 1;
// CHECK-NEXT:     for (_t0 = 2{{U|UL|ULL}}; _t0; _t0--) {
// CHECK-NEXT:         FallbackMemoryValue _r0 = {0., nullptr};
// CHECK-NEXT:         _r0.value += _d_result;
// CHECK-NEXT:         double _r1 = 0.;
// CHECK-NEXT:         clad::custom_derivatives::make_fallback_memory_value_pullback(input, _r0, &_r1);
// CHECK-NEXT:         *_d_input += _r1;
// CHECK-NEXT:     }
// CHECK-NEXT: }
// CHECK: void nonactive_memory_loss_grad(double input, double *_d_input) {
// CHECK-NEXT:     {
// CHECK-NEXT:         MemoryValue _r0 = {0., nullptr, 0};

// A constant factory still supplies the result adjoint of a reverse_forw,
// even though its pullback has no work to do.
// CHECK: clad::ValueAndAdjoint<MemoryValue, MemoryValue> constant_memory_value_reverse_forw(double input, double _d_input) {
// CHECK-NEXT:     MemoryValue _t0 = make_memory_value(1.);
// CHECK-NEXT:     MemoryValue _r0 = clad::zero_like(_t0);
// CHECK-NEXT:     return {_t0, _r0};
// CHECK-NEXT: }
// CHECK: void constant_memory_value_pullback(double input, MemoryValue _d_y, double *_d_input) {
// CHECK-NEXT: }

// CHECK: clad::ValueAndAdjoint<MemoryValue, MemoryValue> operator_memory_value_reverse_forw(double input, double _d_input) {
// CHECK-NEXT:     MemoryValue _t0 = MemoryValue({input, nullptr, 1}) + input;
// CHECK-NEXT:     MemoryValue _r2 = clad::zero_like(_t0);
// CHECK-NEXT:     return {_t0, _r2};
// CHECK-NEXT: }

// CHECK: void loop_memory_loss_grad(double input, double *_d_input) {
// CHECK:     clad::tape<MemoryValue> _t1 = {};
// CHECK-NEXT:     clad::tape<MemoryValue> _r2 = {};
// CHECK:     for (i = 0; i < 2; ++i) {
// CHECK-NEXT:         clad::push(_t1, make_memory_value(input));
// CHECK-NEXT:         clad::push(_r2, clad::zero_like(clad::back(_t1)));
// CHECK-NEXT:         result += clad::back(_t1).value;
// CHECK-NEXT:     }

// CHECK: void explicit_memory_loss_grad(double input, double *_d_input) {
// CHECK-NEXT:     clad::ValueAndAdjoint<MemoryValue, MemoryValue> _t0 = clad::custom_derivatives::explicit_memory_value_reverse_forw(input, 0.);
// CHECK-NOT: zero_like
// CHECK:     clad::custom_derivatives::explicit_memory_value_pullback(input, _d_result,

// An operator without a viable zero_like keeps its existing null-adjoint path.
// CHECK: void fallback_operator_loss_grad(double input, double *_d_input) {
// CHECK-NEXT:     {
// CHECK-NEXT:         FallbackMemoryValue _r0 = {0., nullptr};
// CHECK-NEXT:         _r0.value += 1;
// CHECK-NEXT:         FallbackMemoryValue _r1 = {0., nullptr};
// CHECK-NEXT:         double _r2 = 0.;
// CHECK-NEXT:         clad::custom_derivatives::operator_plus_pullback(FallbackMemoryValue({input, nullptr}), input, _r0, &_r1, &_r2);
// CHECK-NEXT:         *_d_input += _r1.value;
// CHECK-NEXT:         *_d_input += _r2;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// The stored primal loses const, so lookup must select the mutable overload.
// CHECK: void const_memory_loss_grad(double input, double *_d_input) {
// CHECK-NEXT:     MemoryValue _t0 = make_const_memory_value(input);
// CHECK-NEXT:     MemoryValue _r1 = clad::zero_like(_t0);

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

  zeroLikeCalls = 0;
  double constantDerivative = 0.0;
  auto constantGradient = clad::gradient(constant_memory_loss);
  constantGradient.execute(3.0, &constantDerivative);
  printf("%.1f %d\n", constantDerivative,
         zeroLikeCalls); // CHECK-EXEC-NEXT: 0.0 1

  primalCalls = zeroLikeCalls = 0;
  double operatorDerivative = 0.0;
  auto operatorGradient = clad::gradient(operator_memory_loss);
  operatorGradient.execute(3.0, &operatorDerivative);
  printf("%.1f %d\n", operatorDerivative,
         zeroLikeCalls); // CHECK-EXEC-NEXT: 2.0 2

  primalCalls = zeroLikeCalls = 0;
  double loopDerivative = 0.0;
  auto loopGradient = clad::gradient(loop_memory_loss);
  loopGradient.execute(3.0, &loopDerivative);
  printf("%.1f %d %d\n", loopDerivative, primalCalls,
         zeroLikeCalls); // CHECK-EXEC-NEXT: 12.0 2 2

  zeroLikeCalls = 0;
  double explicitDerivative = 0.0;
  auto explicitGradient = clad::gradient(explicit_memory_loss);
  explicitGradient.execute(3.0, &explicitDerivative);
  printf("%.1f %d\n", explicitDerivative,
         zeroLikeCalls); // CHECK-EXEC-NEXT: 6.0 0

  double fallbackOperatorDerivative = 0.0;
  auto fallbackOperatorGradient = clad::gradient(fallback_operator_loss);
  fallbackOperatorGradient.execute(3.0, &fallbackOperatorDerivative);
  printf("%.1f\n", fallbackOperatorDerivative); // CHECK-EXEC-NEXT: 2.0

  primalCalls = zeroLikeCalls = mutableZeroLikeCalls = 0;
  double constDerivative = 0.0;
  auto constGradient = clad::gradient(const_memory_loss);
  constGradient.execute(3.0, &constDerivative);
  printf("%.1f %d %d %d\n", constDerivative, primalCalls, zeroLikeCalls,
         mutableZeroLikeCalls); // CHECK-EXEC-NEXT: 6.0 1 1 1
}
