// RUN: %cladclang %s -I%S/../../include -o %t 2>&1 | %filecheck %s
// RUN: %t | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"

#include <cstdio>

struct Ref {
  double* p;
  Ref(double& x) : p(&x) {}
  operator double&() const { return *p; }
};

struct IndexedRef {
  double* p;
  double& at(int i) const { return p[i]; }
};

int customPullbackCalls = 0;

namespace clad {
namespace custom_derivatives {
namespace class_functions {
clad::ValueAndAdjoint<double&, double&>
at_reverse_forw(const ::IndexedRef* x, int i, const ::IndexedRef* d_x,
                int /*d_i*/) {
  return {x->at(i), d_x->at(i)};
}

void at_pullback(const ::IndexedRef* /*x*/, int /*i*/, ::IndexedRef* /*d_x*/,
                 int* /*d_i*/) {
  ++customPullbackCalls;
}
} // namespace class_functions
} // namespace custom_derivatives
} // namespace clad

template <class T> double product(const T* x) {
  return x[0] * x[1] * x[2] * x[3];
}

double custom_ref_case(double* params) {
  Ref x[]{params[3], params[2], params[1], params[0]};
  return product(x);
}

double custom_derivative_case(double* params) {
  IndexedRef x{params};
  return x.at(1);
}

int main() {
  double params[]{1, 2, 3, 4};
  double dParams[4]{};
  auto grad = clad::gradient(custom_ref_case);
  grad.execute(params, dParams);
  std::printf("{%.0f, %.0f, %.0f, %.0f}\n", dParams[0], dParams[1],
              dParams[2], dParams[3]);

  double dCustomParams[4]{};
  auto customGrad = clad::gradient(custom_derivative_case);
  customGrad.execute(params, dCustomParams);
  std::printf("{%.0f, %.0f, %.0f, %.0f}\n", dCustomParams[0],
              dCustomParams[1], dCustomParams[2], dCustomParams[3]);
  std::printf("%d\n", customPullbackCalls);
}

// CHECK: void product_pullback(const Ref *x, double _d_y, Ref *_d_x) {
// CHECK:     clad::ValueAndAdjoint<double &, double &> _t3 = x[0].conversion_operator_reverse_forw(clad::Tag<double &>(), &_d_x[0]);
// CHECK:     clad::ValueAndAdjoint<double &, double &> _t4 = x[1].conversion_operator_reverse_forw(clad::Tag<double &>(), &_d_x[1]);
// CHECK:     clad::ValueAndAdjoint<double &, double &> _t5 = x[2].conversion_operator_reverse_forw(clad::Tag<double &>(), &_d_x[2]);
// CHECK:     clad::ValueAndAdjoint<double &, double &> _t6 = x[3].conversion_operator_reverse_forw(clad::Tag<double &>(), &_d_x[3]);
// CHECK:         x[0].conversion_operator_pullback(&_d_x[0]);
// CHECK:         x[1].conversion_operator_pullback(&_d_x[1]);
// CHECK:         x[2].conversion_operator_pullback(&_d_x[2]);
// CHECK:         x[3].conversion_operator_pullback(&_d_x[3]);

// CHECK: void custom_derivative_case_grad(double *params, double *_d_params) {
// CHECK:     clad::ValueAndAdjoint<double &, double &> _t0 = clad::custom_derivatives::class_functions::at_reverse_forw(&x, 1, &_d_x, 0);
// CHECK:         clad::custom_derivatives::class_functions::at_pullback(&x, 1, &_d_x, &_r0);

// CHECK-EXEC: {24, 12, 8, 6}
// CHECK-EXEC-NEXT: {0, 1, 0, 0}
// CHECK-EXEC-NEXT: 1
