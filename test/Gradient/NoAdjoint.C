// RUN: %cladclang %s -I%S/../../include -o %t 2>&1 | %filecheck %s
// RUN: %t | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"

extern "C" int printf(const char*, ...);

int selected = 0;

double legacy(double data, double weight) { return data * weight; }
double compact(double data, double weight) { return data * weight; }
double generated(double data, double weight) { return data * weight; }
double generated_middle(double x, double data, double z) {
  return x * data + x * z;
}
double activity(const double& data, double weight) { return data * weight; }
template <class T> T product(const T& data, const T& weight) {
  return data * weight;
}
template <int N> double scaled_product(double data, double weight) {
  return N * data * weight;
}

namespace impl {
double array_product(const double* data, double weight) {
  return data[0] * weight;
}
} // namespace impl
namespace api {
using impl::array_product;
}

namespace clad::custom_derivatives::api {
// Follow the using-declaration's namespace, including for pointer inputs.
void array_product_pullback(const double* data, double, double seed,
                            clad::no_adjoint, double* d_weight) {
  selected = 6;
  *d_weight += data[0] * seed;
}
} // namespace clad::custom_derivatives::api

namespace clad::custom_derivatives {
// No nullable contract: both pointers are unconditionally dereferenced.
void legacy_pullback(double data, double weight, double seed,
                     double* d_data, double* d_weight) {
  selected = 1;
  *d_data += weight * seed;
  *d_weight += data * seed;
}
// An untagged compact overload cannot say which direction it implements.
// Even when it matches by type, use the full custom below instead.
void compact_pullback(double data, double, double seed, double* d_weight) {
  selected = -1;
  *d_weight += data * seed;
}
void compact_pullback(double data, double weight, double seed,
                      double* d_data, double* d_weight) {
  selected = 2;
  *d_data += weight * seed;
  *d_weight += data * seed;
}
void activity_pullback(const double& data, double, double seed,
                       double* d_weight) {
  selected = -1;
  *d_weight += data * seed;
}
void activity_pullback(const double& data, double weight, double seed,
                       double* d_data, double* d_weight) {
  selected = 7;
  *d_data += weight * seed;
  *d_weight += data * seed;
}
// The position-preserving marker must win over the full signature and the
// ambiguous compact overload.
template <class T>
void product_pullback(const T& data, const T& weight, T seed,
                      T* d_data, T* d_weight) {
  selected = 3;
  *d_data += weight * seed;
  *d_weight += data * seed;
}
template <class T>
void product_pullback(const T& data, const T&, T seed, T* d_weight) {
  selected = 4;
  *d_weight += data * seed;
}
template <class T>
void product_pullback(const T& data, const T&, T seed,
                      clad::no_adjoint, T* d_weight) {
  selected = 5;
  *d_weight += data * seed;
}
template <int N>
void scaled_product_pullback(::std::integral_constant<int, N>, double data,
                             double, double seed, clad::no_adjoint,
                             double* d_weight) {
  *d_weight += N * data * seed;
}
} // namespace clad::custom_derivatives

double legacy_loss(double data, double weight) { return legacy(data, weight); }
double compact_loss(double data, double weight) { return compact(data, weight); }
double generated_loss(double data, double weight) {
  return generated(data, weight);
}
double middle_loss(double x, double data, double z) {
  return generated_middle(x, data, z);
}
double activity_loss(double weight) {
  double data = 3;
  return activity(data, weight);
}
double template_loss(double data, double weight) { return product(data, weight); }
double array_loss(const double* data, double weight) {
  return api::array_product(data, weight);
}
double scaled_loss(double data, double weight) {
  return scaled_product<2>(data, weight) + scaled_product<3>(data, weight);
}

// A marker is useful independently of the request option when an existing
// non-differentiable parameter already has no adjoint.
double annotated_loss(CLAD_NONDIFFERENTIABLE double data, double weight) {
  return product(data, weight);
}

int main() {
  auto old = clad::gradient<clad::opts::assume_unselected_inactive>(legacy_loss, "weight");
  auto small = clad::gradient<clad::opts::assume_unselected_inactive>(compact_loss, "weight");
  auto small_data = clad::gradient<clad::opts::assume_unselected_inactive>(compact_loss, "data");
  auto generated_weight = clad::gradient<clad::opts::assume_unselected_inactive>(generated_loss, "weight");
  auto generated_data = clad::gradient<clad::opts::assume_unselected_inactive>(generated_loss, "data");
  auto middle = clad::gradient<clad::opts::assume_unselected_inactive>(middle_loss, "x,z");
  auto active = clad::gradient<clad::opts::enable_va>(activity_loss);
  auto tagged = clad::gradient<clad::opts::assume_unselected_inactive>(template_loss, "weight");
  auto full = clad::gradient(template_loss);
  auto annotated = clad::gradient(annotated_loss, "weight");
  auto array = clad::gradient<clad::opts::assume_unselected_inactive>(array_loss, "weight");
  auto scaled = clad::gradient<clad::opts::assume_unselected_inactive>(
      scaled_loss, "weight");
  double dw = 0;
  old.execute(3, 2, &dw);
  printf("legacy: %.0f, overload %d\n", dw, selected);
  // CHECK-EXEC: legacy: 3, overload 1
  dw = 0;
  small.execute(3, 2, &dw);
  printf("compact: %.0f, overload %d\n", dw, selected);
  // CHECK-EXEC: compact: 3, overload 2
  double dd = 0;
  small_data.execute(3, 2, &dd);
  printf("compact data: %.0f, overload %d\n", dd, selected);
  // CHECK-EXEC: compact data: 2, overload 2
  dd = dw = 0;
  generated_weight.execute(3, 2, &dw);
  generated_data.execute(3, 2, &dd);
  printf("generated: %.0f %.0f\n", dd, dw);
  // CHECK-EXEC: generated: 2 3
  double dx = 0, dz = 0;
  middle.execute(2, 3, 5, &dx, &dz);
  printf("middle marker: %.0f %.0f\n", dx, dz);
  // CHECK-EXEC: middle marker: 8 2
  dw = 0;
  active.execute(2, &dw);
  printf("activity: %.0f, overload %d\n", dw, selected);
  // CHECK-EXEC: activity: 3, overload 7
  dw = 0;
  tagged.execute(3, 2, &dw);
  printf("tagged template: %.0f, overload %d\n", dw, selected);
  // CHECK-EXEC: tagged template: 3, overload 5
  dd = 0;
  dw = 0;
  full.execute(3, 2, &dd, &dw);
  printf("full: %.0f %.0f, overload %d\n", dd, dw, selected);
  // CHECK-EXEC: full: 2 3, overload 3
  dw = 0;
  annotated.execute(3, 2, &dw);
  printf("annotated: %.0f, overload %d\n", dw, selected);
  // CHECK-EXEC: annotated: 3, overload 5
  dw = 0;
  double data[] = {3};
  array.execute(data, 2, &dw);
  printf("array: %.0f, overload %d\n", dw, selected);
  // CHECK-EXEC: array: 3, overload 6
  dw = 0;
  scaled.execute(3, 2, &dw);
  printf("template tags: %.0f\n", dw);
  // CHECK-EXEC: template tags: 15
}

// CHECK-LABEL: void legacy_loss_grad_1_assume_inactive(
// CHECK-NOT: nullptr
// CHECK: clad::custom_derivatives::legacy_pullback(data, weight, 1, &_r{{[0-9]+}}, &_r{{[0-9]+}});
// CHECK-LABEL: void compact_loss_grad_1_assume_inactive(
// CHECK: clad::custom_derivatives::compact_pullback(data, weight, 1, &_r{{[0-9]+}}, &_r{{[0-9]+}});
// CHECK-LABEL: void compact_loss_grad_0_assume_inactive(
// CHECK: clad::custom_derivatives::compact_pullback(data, weight, 1, &_r{{[0-9]+}}, &_r{{[0-9]+}});
// CHECK-LABEL: void generated_pullback_assume_inactive(double data, double weight, double _d_y, clad::no_adjoint, double *_d_weight)
// CHECK-LABEL: void generated_loss_grad_1_assume_inactive(
// CHECK: generated_pullback_assume_inactive(data, weight, 1, clad::no_adjoint(), &_r{{[0-9]+}});
// CHECK-LABEL: void generated_pullback_assume_inactive(double data, double weight, double _d_y, double *_d_data, clad::no_adjoint)
// CHECK-LABEL: void generated_loss_grad_0_assume_inactive(
// CHECK: generated_pullback_assume_inactive(data, weight, 1, &_r{{[0-9]+}}, clad::no_adjoint());
// CHECK-LABEL: void generated_middle_pullback_assume_inactive(double x, double data, double z, double _d_y, double *_d_x, clad::no_adjoint, double *_d_z)
// CHECK-LABEL: void middle_loss_grad_0_2_assume_inactive(
// CHECK: generated_middle_pullback_assume_inactive(x, data, z, 1, &_r{{[0-9]+}}, clad::no_adjoint(), &_r{{[0-9]+}});
// CHECK-LABEL: void template_loss_grad_1_assume_inactive(
// CHECK: clad::custom_derivatives::product_pullback(data, weight, 1, clad::no_adjoint(),
// CHECK-LABEL: void array_loss_grad_1_assume_inactive(
// CHECK-NOT: _d_data
// CHECK: clad::custom_derivatives::api::array_product_pullback(data, weight, 1, clad::no_adjoint(),
// CHECK-LABEL: void scaled_loss_grad_1_assume_inactive(
// CHECK-DAG: scaled_product_pullback(std::integral_constant<int, 2>(), data, weight, 1, clad::no_adjoint(),
// CHECK-DAG: scaled_product_pullback(std::integral_constant<int, 3>(), data, weight, 1, clad::no_adjoint(),
