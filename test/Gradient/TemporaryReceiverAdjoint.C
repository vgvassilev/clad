// RUN: %cladclang %s -I%S/../../include -o %t 2>&1 | %filecheck %s
// RUN: %t | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"

extern "C" int printf(const char*, ...);

struct Value {
  double value;
  double* storage;
  int extent;

  ~Value() {} // Keep the temporary-binding wrappers used by container types.

  double read() const { return value; }
  Value scale(double factor) const {
    return {value * factor, nullptr, extent};
  }
};

int zeroLikeCalls = 0;
int reverseForwCalls = 0;

Value make_value(double x) { return {x * x, nullptr, 7}; }
Value explicit_value(double x) { return make_value(x); }
Value operator*(Value lhs, Value rhs) {
  return {lhs.value * rhs.value, nullptr, lhs.extent};
}

namespace clad {
Value zero_like(const Value& value) {
  ++zeroLikeCalls;
  return {0., nullptr, value.extent};
}

namespace custom_derivatives {
void make_value_pullback(double x, Value d_output, double* d_x) {
  *d_x += 2 * x * d_output.value;
}

void operator_star_pullback(Value lhs, Value rhs, Value d_output, Value* d_lhs,
                           Value* d_rhs) {
  // Argument adjoints still need the structure supplied by zero_like.
  if (d_lhs->extent != lhs.extent || d_rhs->extent != rhs.extent)
    __builtin_trap();
  d_lhs->value += rhs.value * d_output.value;
  d_rhs->value += lhs.value * d_output.value;
}

ValueAndAdjoint<Value, Value> explicit_value_reverse_forw(double x, double) {
  ++reverseForwCalls;
  return {explicit_value(x), {0., nullptr, 7}};
}

void explicit_value_pullback(double x, Value d_output, double* d_x) {
  *d_x += 2 * x * d_output.value;
}

namespace class_functions {
void read_pullback(const Value*, double d_output, Value* d_this) {
  d_this->value += d_output;
}

void scale_pullback(const Value* self, double factor, Value d_output,
                    Value* d_this, double* d_factor) {
  d_this->value += factor * d_output.value;
  *d_factor += self->value * d_output.value;
}
} // namespace class_functions
} // namespace custom_derivatives
} // namespace clad

double temporary(double x) { return make_value(x).read(); }

// CHECK: void temporary_grad(double x, double *_d_x) {
// CHECK-NOT: zero_like
// CHECK: clad::custom_derivatives::class_functions::read_pullback
// CHECK: clad::custom_derivatives::make_value_pullback
// CHECK: }

double nested(double x) { return make_value(x).scale(x).read(); }

double arguments(double x) {
  return (make_value(x) * make_value(x)).read();
}

double local(double x) {
  auto value = make_value(x);
  return value.read();
}

double reference(double x) {
  auto value = make_value(x);
  Value& ref = value;
  return ref.read();
}

double loop(double x) {
  double result = 0.;
  for (int i = 0; i < 2; ++i)
    result += make_value(x).read();
  return result;
}

double explicit_forward(double x) { return explicit_value(x).read(); }

int main() {
  double d_x = 0.;
  clad::gradient(temporary).execute(3., &d_x);
  printf("temporary: %.0f, zeros: %d\n", d_x, zeroLikeCalls);
  // CHECK-EXEC: temporary: 6, zeros: 0

  d_x = 0.;
  zeroLikeCalls = 0;
  clad::gradient(nested).execute(3., &d_x);
  printf("nested: %.0f, zeros: %d\n", d_x, zeroLikeCalls);
  // CHECK-EXEC-NEXT: nested: 27, zeros: 0

  d_x = 0.;
  zeroLikeCalls = 0;
  clad::gradient(arguments).execute(3., &d_x);
  printf("arguments: %.0f, zeros: %d\n", d_x, zeroLikeCalls);
  // CHECK-EXEC-NEXT: arguments: 108, zeros: 2

  d_x = 0.;
  zeroLikeCalls = 0;
  clad::gradient(local).execute(3., &d_x);
  printf("local: %.0f, zeros: %d\n", d_x, zeroLikeCalls);
  // CHECK-EXEC-NEXT: local: 6, zeros: 1

  d_x = 0.;
  zeroLikeCalls = 0;
  clad::gradient(reference).execute(3., &d_x);
  printf("reference: %.0f, zeros: %d\n", d_x, zeroLikeCalls);
  // CHECK-EXEC-NEXT: reference: 6, zeros: 1

  d_x = 0.;
  zeroLikeCalls = 0;
  clad::gradient(loop).execute(3., &d_x);
  printf("loop: %.0f, zeros: %d\n", d_x, zeroLikeCalls);
  // CHECK-EXEC-NEXT: loop: 12, zeros: 0

  d_x = 0.;
  zeroLikeCalls = 0;
  clad::gradient(explicit_forward).execute(3., &d_x);
  printf("explicit: %.0f, zeros: %d, forwards: %d\n", d_x, zeroLikeCalls,
         reverseForwCalls);
  // CHECK-EXEC-NEXT: explicit: 6, zeros: 0, forwards: 1
}
