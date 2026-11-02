// RUN: %cladclang %s -I%S/../../include -Xclang -verify -o %t 2>&1 | %filecheck %s
// RUN: %t | %filecheck_exec %s
// RUN: %cladclang %s -I%S/../../include -Xclang -plugin-arg-clad -Xclang -fenable-analysis=activity -o %t.va
// RUN: %t.va | %filecheck_exec %s
// expected-no-diagnostics

#include "clad/Differentiator/Differentiator.h"

extern "C" int printf(const char*, ...);

struct Box {
  double value;
  double* storage;
};

struct Sample {
  double value;
  double scale(double weight) const { return value * weight; }
  double plain_scale(double weight) const { return value * weight; }
  double state_scale(double weight) const { return value * weight; }
  Box boxed_scale(double weight) const { return {value * weight, nullptr}; }
};

int data_pullbacks = 0;

double multiply(const Sample& data, double weight) {
  return data.value * weight;
}

namespace clad::custom_derivatives {
void multiply_pullback(const Sample& data, double weight, double seed,
                       Sample* d_data, double* d_weight) {
  ++data_pullbacks;
  d_data->value += weight * seed;
  *d_weight += data.value * seed;
}
void multiply_pullback(const Sample& data, double, double seed,
                       clad::no_adjoint, double* d_weight) {
  *d_weight += data.value * seed;
}
void multiply_pullback(const Sample&, double weight, double seed,
                       Sample* d_data, clad::no_adjoint) {
  ++data_pullbacks;
  d_data->value += weight * seed;
}
} // namespace clad::custom_derivatives

namespace clad::custom_derivatives::class_functions {
void scale_pullback(const Sample* data, double weight, double seed,
                    Sample* d_data, double* d_weight) {
  multiply_pullback(*data, weight, seed, d_data, d_weight);
}
void scale_pullback(const Sample* data, double, double seed,
                    clad::no_adjoint, double* d_weight) {
  *d_weight += data->value * seed;
}
double state_scale_reverse_forw(const Sample* data, double weight,
                               const Sample* d_data, double,
                               clad::pullback_state<double>& state) {
  // The older forward helper still requires a valid, initialized pointer.
  state.data = data->value + d_data->value;
  return data->state_scale(weight);
}
void state_scale_pullback(const Sample*, double, double seed, clad::no_adjoint,
                          double* d_weight, clad::pullback_state<double> state) {
  *d_weight += state.data * seed;
}
clad::ValueAndAdjoint<Box, Box>
boxed_scale_reverse_forw(const Sample* data, double weight,
                        const Sample* d_data, double) {
  return {data->boxed_scale(weight), {d_data->value, nullptr}};
}
void boxed_scale_pullback(const Sample* data, double, Box seed,
                          clad::no_adjoint, double* d_weight) {
  *d_weight += data->value * seed.value;
}
} // namespace clad::custom_derivatives::class_functions

double method_loss(const Sample& data, double weight) {
  return data.scale(weight);
}

double generated_method_loss(const Sample& data, double weight) {
  return data.plain_scale(weight);
}

double state_method_loss(const Sample& data, double weight) {
  double value = data.state_scale(weight);
  return value;
}

double loop_method_loss(const Sample& data, double weight) {
  double value = 0;
  for (int i = 0; i < 3; ++i) {
    Box result = data.boxed_scale(weight);
    value += result.value;
  }
  return value;
}

double scalar_product(double data, double weight) { return data * weight; }

namespace clad::custom_derivatives {
void scalar_product_pullback(double data, double, double seed,
                             clad::no_adjoint, double* d_weight) {
  *d_weight += data * seed;
}
} // namespace clad::custom_derivatives

double scalar_loss(double data, double weight) {
  return scalar_product(data, weight);
}

double buffered(const Sample& data, double weight) {
  return data.value * weight;
}

namespace clad::custom_derivatives {
double buffered_reverse_forw(const Sample& data, double weight, const Sample&,
                             double, clad::pullback_state<double>& state) {
  state.data = data.value;
  return data.value * weight;
}
void buffered_pullback(const Sample&, double, double seed,
                       clad::no_adjoint, double* d_weight,
                       clad::pullback_state<double> state) {
  *d_weight += state.data * seed;
}
} // namespace clad::custom_derivatives

double state_loss(const Sample& data, double weight) {
  double value = buffered(data, weight);
  return value;
}

double loss(const Sample& data, double weight) {
  return multiply(data, weight);
}

// The reverse request order must not reuse the other option's derivative.
double other_loss(const Sample& data, double weight) {
  return multiply(data, weight);
}

double layer(const Sample& data, double weight) {
  double hidden = weight * weight;
  return multiply(data, hidden);
}

double nested(const Sample& data, double weight) {
  return layer(data, weight);
}

// This copy becomes active inside the callee. The root's promise about its
// unselected parameter does not apply to the callee's mutable local copy.
double replace_copy(double x, double y) {
  y = x * x;
  return y;
}

double nested_copy(double data, double weight) {
  return replace_copy(weight, data);
}

int main() {
  Sample data{3};
  double dw = 0;
  auto normal = clad::gradient(loss, "weight");
  auto assumed = clad::gradient<clad::opts::assume_unselected_inactive>(
      loss, "weight");
  normal.execute(data, 2, &dw);
  printf("default: %.0f, data pullbacks: %d\n", dw, data_pullbacks);
  // CHECK-EXEC: default: 3, data pullbacks: 1
  dw = 0;
  data_pullbacks = 0;
  assumed.execute(data, 2, &dw);
  printf("assumed: %.0f, data pullbacks: %d\n", dw, data_pullbacks);
  // CHECK-EXEC: assumed: 3, data pullbacks: 0

  auto other_assumed =
      clad::gradient<clad::opts::assume_unselected_inactive>(other_loss, "weight");
  auto other_normal = clad::gradient(other_loss, "weight");
  dw = 0;
  other_assumed.execute(data, 2, &dw);
  printf("assumed first: %.0f, data pullbacks: %d\n", dw, data_pullbacks);
  // CHECK-EXEC: assumed first: 3, data pullbacks: 0
  dw = 0;
  other_normal.execute(data, 2, &dw);
  printf("default second: %.0f, data pullbacks: %d\n", dw, data_pullbacks);
  // CHECK-EXEC: default second: 3, data pullbacks: 1

  auto nested_grad = clad::gradient<clad::opts::assume_unselected_inactive>(
      nested, "weight");
  dw = 0;
  data_pullbacks = 0;
  nested_grad.execute(data, 2, &dw);
  printf("nested: %.0f, data pullbacks: %d\n", dw, data_pullbacks);
  // CHECK-EXEC: nested: 12, data pullbacks: 0

  auto copy_grad = clad::gradient<clad::opts::assume_unselected_inactive>(
      nested_copy, "weight");
  dw = 0;
  copy_grad.execute(3, 2, &dw);
  printf("mutable callee copy: %.0f\n", dw);
  // CHECK-EXEC: mutable callee copy: 4

  auto scalar_grad = clad::gradient<clad::opts::assume_unselected_inactive>(
      scalar_loss, "weight");
  dw = 0;
  scalar_grad.execute(3, 2, &dw);
  printf("scalar: %.0f, data pullbacks: %d\n", dw, data_pullbacks);
  // CHECK-EXEC: scalar: 3, data pullbacks: 0

  auto method_grad = clad::gradient<clad::opts::assume_unselected_inactive>(
      method_loss, "weight");
  dw = 0;
  method_grad.execute(data, 2, &dw);
  printf("method: %.0f, data pullbacks: %d\n", dw, data_pullbacks);
  // CHECK-EXEC: method: 3, data pullbacks: 0

  auto state_grad = clad::gradient<clad::opts::assume_unselected_inactive>(
      state_loss, "weight");
  dw = 0;
  state_grad.execute(data, 2, &dw);
  printf("state: %.0f, data pullbacks: %d\n", dw, data_pullbacks);
  // CHECK-EXEC: state: 3, data pullbacks: 0

  // Reversing the selection preserves the data output and omits the scalar.
  auto data_grad = clad::gradient<clad::opts::assume_unselected_inactive>(
      loss, "data");
  Sample d_data{};
  data_grad.execute(data, 2, &d_data);
  printf("selected data: %.0f, data pullbacks: %d\n", d_data.value,
         data_pullbacks);
  // CHECK-EXEC: selected data: 2, data pullbacks: 1

  auto generated_method =
      clad::gradient<clad::opts::assume_unselected_inactive>(
          generated_method_loss, "weight");
  dw = 0;
  generated_method.execute(data, 2, &dw);
  printf("generated method: %.0f\n", dw);
  // CHECK-EXEC: generated method: 3

  auto state_method = clad::gradient<clad::opts::assume_unselected_inactive>(
      state_method_loss, "weight");
  dw = 0;
  state_method.execute(data, 2, &dw);
  printf("state method: %.0f\n", dw);
  // CHECK-EXEC: state method: 3

  auto loop_method = clad::gradient<clad::opts::assume_unselected_inactive>(
      loop_method_loss, "weight");
  dw = 0;
  loop_method.execute(data, 2, &dw);
  printf("loop method: %.0f\n", dw);
  // CHECK-EXEC: loop method: 9
}

// CHECK-LABEL: void loss_grad_1_assume_inactive(
// CHECK-NOT: Sample _d_data
// CHECK: clad::custom_derivatives::multiply_pullback(data, weight, 1, {{(clad::)?}}no_adjoint(),
// CHECK-LABEL: void other_loss_grad_1_assume_inactive(
// CHECK-NOT: Sample _d_data
// CHECK: clad::custom_derivatives::multiply_pullback(data, weight, 1, {{(clad::)?}}no_adjoint(),
// CHECK-LABEL: void state_method_loss_grad_1_assume_inactive(
// CHECK: Sample [[RECEIVER:_r[0-9]+]] = Sample({0.});
// CHECK: state_scale_reverse_forw(&data, weight, &[[RECEIVER]],
// CHECK: state_scale_pullback(&data, weight, _d_value, clad::no_adjoint(),
