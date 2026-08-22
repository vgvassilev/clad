// RUN: %cladclang -std=c++17 -I%S/../../include %s -o %t 2>&1 | %filecheck %s
// RUN: %t | %filecheck_exec %s

// A local alias of read-only data -- `const double* t = xlArr;` or
// `const double& a = xlArr[0];` -- has no tangent, since the const pointer
// parameter it aliases cannot have one either. Forward mode used to fabricate
// one anyway: a null pointer typed `const double*`, later dereferenced, or a
// reference bound to the literal 0. Both broke the Hessian, whose reverse pass
// then built a `double*` adjoint out of a `const double*` initializer, or
// bound a `double&` adjoint to a temporary.

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

double ptr_alias(double* params, double const* xlArr) {
  double const* t = xlArr;
  return params[0] * params[0] * t[0] + params[0] * params[1] * t[1];
}

// In the hessian's pushforward the const pointer parameter has a run-time
// tangent, so the alias inherits it -- nullable, with every read guarded.
// CHECK: clad::ValueAndPushforward<double, double> ptr_alias_pushforward(double *params, const double *xlArr, double *_d_params, const double *_d_xlArr) {
// CHECK-NEXT: const double *_d_t = _d_xlArr;
// CHECK-NEXT: const double *t = xlArr;

double ref_alias(double* params, double const* xlArr) {
  double const& a = xlArr[0];
  double const& b = 3.;
  return params[0] * params[0] * a * b;
}

// A reference alias cannot be reseated to a run-time tangent, so it gets
// none; uses of it derive to a plain 0.
// CHECK: clad::ValueAndPushforward<double, double> ref_alias_pushforward(double *params, const double *xlArr, double *_d_params, const double *_d_xlArr) {
// CHECK-NEXT: const double &a = xlArr[0];
// CHECK-NEXT: const double &b = 3.;

// The shape this was reported with: RooFit's code generation emits such
// aliases inside a checkpointed loop.
double loop_alias(double* params, double const* xlArr) {
  double sum = 0.;
#pragma clad checkpoint loop
  for (int i = 0; i < 2; i++) {
    unsigned int idx = i;
    double const* t = xlArr + 1 * idx;
    sum += params[0] * params[0] * t[0];
  }
  return sum;
}

// CHECK: clad::ValueAndPushforward<double, double> loop_alias_pushforward(double *params, const double *xlArr, double *_d_params, const double *_d_xlArr) {
// CHECK: unsigned int idx = i;
// CHECK-NEXT: const double *_d_t = (_d_xlArr ? _d_xlArr + idx : nullptr);
// CHECK-NEXT: const double *t = xlArr + 1 * idx;

// A `const double&` bound to a literal broke plain reverse mode as well.
double ref_to_literal(double* params) {
  double const& r = 5.;
  return params[0] * r;
}

// CHECK: void ref_to_literal_grad(double *params, double *_d_params) {
// CHECK-NEXT: const double &r = 5.;
// CHECK-NEXT: _d_params[0] += 1 * r;

// Outside the hessian there is no tangent to inherit: the alias gets none,
// and uses of it derive to a plain 0.
// CHECK: double ptr_alias_darg0_0(double *params, const double *xlArr) {
// CHECK-NEXT: const double *t = xlArr;

// The reverse pass keeps the alias and its tangent `const double*` and never
// builds an adjoint out of them.
// CHECK: void ptr_alias_pushforward_0_pullback(double *params, const double *xlArr, double *_d_params, const double *_d_xlArr, clad::ValueAndPushforward<double, double> _d_y, double *_d_params0) {
// CHECK-NEXT: const double *_d_t = _d_xlArr;
// CHECK-NEXT: const double *t = xlArr;

int main() {
  double params[2] = {2., 3.};
  double xlArr[2] = {5., 7.};

  double m[4] = {};
  clad::hessian(ptr_alias, "params[0:1]").execute(params, xlArr, m);
  printf("ptr_alias %.2f %.2f %.2f %.2f\n", m[0], m[1], m[2], m[3]);
  // CHECK-EXEC: ptr_alias 10.00 7.00 7.00 0.00

  double m2[1] = {};
  clad::hessian(ref_alias, "params[0]").execute(params, xlArr, m2);
  printf("ref_alias %.2f\n", m2[0]);
  // CHECK-EXEC: ref_alias 30.00

  double m3[1] = {};
  clad::hessian(loop_alias, "params[0]").execute(params, xlArr, m3);
  printf("loop_alias %.2f\n", m3[0]);
  // CHECK-EXEC: loop_alias 24.00

  double d = 0.;
  clad::gradient(ref_to_literal, "params[0]").execute(params, &d);
  printf("ref_to_literal %.2f\n", d);
  // CHECK-EXEC: ref_to_literal 5.00

  printf("ptr_alias_darg %.2f\n",
         clad::differentiate(ptr_alias, "params[0]").execute(params, xlArr));
  // CHECK-EXEC: ptr_alias_darg 41.00
}
