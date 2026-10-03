// RUN: %cladclang %s -DMODE=0 -I%S/../../include -o %t.scalar -Xclang -verify 2>&1 | %filecheck %s --check-prefix=SCALAR
// RUN: %t.scalar | %filecheck %s --check-prefix=SCALAR-EXEC
// RUN: %cladclang %s -DMODE=1 -I%S/../../include -o %t.vector -Xclang -verify 2>&1 | %filecheck %s --check-prefix=VECTOR
// RUN: %t.vector | %filecheck %s --check-prefix=VECTOR-EXEC
// expected-no-diagnostics

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

CLAD_NONDIFFERENTIABLE double opaque(double v) { return v; }
int calls = 0;

double set_reference(double& r, double x) {
  ++calls;
  r = x * x;
  return r;
}

double set_pointer(double* r, double x) {
  ++calls;
  *r = x * x;
  return x * x;
}

double direct_reference(double x) {
  double r = 0;
  opaque(set_reference(r, x));
  return r;
}

double direct_pointer(double x) {
  double r = 0;
  opaque(set_pointer(&r, x));
  return r;
}

double nested_reference(double x, bool enabled) {
  double r = 0;
  if (enabled && opaque(set_reference(r, x))) {}
  return r;
}

int main() {
#if MODE == 0
  auto Ref = clad::differentiate(direct_reference, "x");
  auto Ptr = clad::differentiate(direct_pointer, "x");
  auto Nested = clad::differentiate(nested_reference, "x");
#else
  auto Ref = clad::differentiate<clad::opts::vector_mode>(direct_reference, "x");
  auto Nested = clad::differentiate<clad::opts::vector_mode>(nested_reference, "x");
#endif
  double R = 0, P = 0, Off = 0, On = 0;
  calls = 0;
#if MODE == 0
  R = Ref.execute(2);
  P = Ptr.execute(2);
  Off = Nested.execute(2, false);
#else
  Ref.execute(2, &R);
  Nested.execute(2, false, &Off);
#endif
  int BeforeOn = calls;
#if MODE == 0
  On = Nested.execute(2, true);
#else
  Nested.execute(2, true, &On);
#endif
  printf("reference %.1f\n", R);
#if MODE == 0
  printf("pointer %.1f\n", P);
#endif
  printf("nested %.1f %.1f calls %d %d\n", Off, On, BeforeOn, calls);
  // SCALAR-EXEC: reference 4.0
  // SCALAR-EXEC-NEXT: pointer 4.0
  // SCALAR-EXEC-NEXT: nested 0.0 4.0 calls 2 3
  // VECTOR-EXEC: reference 4.0
  // VECTOR-EXEC-NEXT: nested 0.0 4.0 calls 1 2
}

// SCALAR-LABEL: inline clad::ValueAndPushforward<double, double> set_reference_pushforward(double &r, double x, double &_d_r, double _d_x) {
// SCALAR-NEXT:     ++calls;
// SCALAR-NEXT:     _d_r = _d_x * x + x * _d_x;
// SCALAR-NEXT:     r = x * x;
// SCALAR-NEXT:     return {r, _d_r};
// SCALAR-NEXT: }
// SCALAR-LABEL: double direct_reference_darg0(double x) {
// SCALAR-NEXT:     double _d_x = 1;
// SCALAR-NEXT:     double _d_r = 0;
// SCALAR-NEXT:     double r = 0;
// SCALAR-NEXT:     opaque([&]() -> double {
// SCALAR-NEXT:         clad::ValueAndPushforward<double, double> _t1 = set_reference_pushforward(r, x, _d_r, _d_x);
// SCALAR-NEXT:         return _t1.value;
// SCALAR-NEXT:     }());
// SCALAR-NEXT:     return _d_r;
// SCALAR-NEXT: }
// SCALAR-LABEL: inline clad::ValueAndPushforward<double, double> set_pointer_pushforward(double *r, double x, double *_d_r, double _d_x) {
// SCALAR-NEXT:     ++calls;
// SCALAR-NEXT:     *_d_r = _d_x * x + x * _d_x;
// SCALAR-NEXT:     *r = x * x;
// SCALAR-NEXT:     return {x * x, _d_x * x + x * _d_x};
// SCALAR-NEXT: }
// SCALAR-LABEL: double direct_pointer_darg0(double x) {
// SCALAR-NEXT:     double _d_x = 1;
// SCALAR-NEXT:     double _d_r = 0;
// SCALAR-NEXT:     double r = 0;
// SCALAR-NEXT:     opaque([&]() -> double {
// SCALAR-NEXT:         clad::ValueAndPushforward<double, double> _t1 = set_pointer_pushforward(&r, x, &_d_r, _d_x);
// SCALAR-NEXT:         return _t1.value;
// SCALAR-NEXT:     }());
// SCALAR-NEXT:     return _d_r;
// SCALAR-NEXT: }
// SCALAR-LABEL: double nested_reference_darg0(double x, bool enabled) {
// SCALAR-NEXT:     double _d_x = 1;
// SCALAR-NEXT:     bool _d_enabled = 0;
// SCALAR-NEXT:     double _d_r = 0;
// SCALAR-NEXT:     double r = 0;
// SCALAR-NEXT:     if (enabled && opaque([&]() -> double {
// SCALAR-NEXT:         clad::ValueAndPushforward<double, double> _t1 = set_reference_pushforward(r, x, _d_r, _d_x);
// SCALAR-NEXT:         return _t1.value;
// SCALAR-NEXT:     }())) {
// SCALAR-NEXT:     }
// SCALAR-NEXT:     return _d_r;
// SCALAR-NEXT: }

// VECTOR-LABEL: inline clad::ValueAndPushforward<double, clad::array<double> > set_reference_vector_pushforward(double &r, double x, clad::array<double> &_d_r, clad::array<double> _d_x) {
// VECTOR-NEXT:     unsigned long indepVarCount = _d_x.size();
// VECTOR-NEXT:     ++calls;
// VECTOR-NEXT:     _d_r = _d_x * x + x * _d_x;
// VECTOR-NEXT:     r = x * x;
// VECTOR-NEXT:     return {r, _d_r};
// VECTOR-NEXT: }
// VECTOR-LABEL: void direct_reference_dvec(double x, double *_d_x) {
// VECTOR-NEXT:     unsigned long indepVarCount = 1UL;
// VECTOR-NEXT:     clad::array<double> _d_vector_x = clad::one_hot_vector(indepVarCount, 0UL);
// VECTOR-NEXT:     clad::array<double> _d_vector_r(clad::zero_vector(indepVarCount));
// VECTOR-NEXT:     double r = 0;
// VECTOR-NEXT:     clad::zero_vector(indepVarCount);
// VECTOR-NEXT:     opaque([&]() -> double {
// VECTOR-NEXT:         clad::ValueAndPushforward<double, clad::array<double> > _t1 = set_reference_vector_pushforward(r, x, _d_vector_r, _d_vector_x);
// VECTOR-NEXT:         return _t1.value;
// VECTOR-NEXT:     }());
// VECTOR-NEXT:     {
// VECTOR-NEXT:         clad::array<double> _d_vector_return(_d_vector_r);
// VECTOR-NEXT:         *_d_x = _d_vector_return[0UL];
// VECTOR-NEXT:         return;
// VECTOR-NEXT:     }
// VECTOR-NEXT: }
// VECTOR-LABEL: void nested_reference_dvec_0(double x, bool enabled, double *_d_x) {
// VECTOR-NEXT:     unsigned long indepVarCount = 1UL;
// VECTOR-NEXT:     clad::array<double> _d_vector_x = clad::one_hot_vector(indepVarCount, 0UL);
// VECTOR-NEXT:     clad::array<bool> _d_vector_enabled = clad::zero_vector(indepVarCount);
// VECTOR-NEXT:     clad::array<double> _d_vector_r(clad::zero_vector(indepVarCount));
// VECTOR-NEXT:     double r = 0;
// VECTOR-NEXT:     if (enabled && ((clad::zero_vector(indepVarCount)) , opaque([&]() -> double {
// VECTOR-NEXT:         clad::ValueAndPushforward<double, clad::array<double> > _t1 = set_reference_vector_pushforward(r, x, _d_vector_r, _d_vector_x);
// VECTOR-NEXT:         return _t1.value;
// VECTOR-NEXT:     }()))) {
// VECTOR-NEXT:     }
// VECTOR-NEXT:     {
// VECTOR-NEXT:         clad::array<double> _d_vector_return(_d_vector_r);
// VECTOR-NEXT:         *_d_x = _d_vector_return[0UL];
// VECTOR-NEXT:         return;
// VECTOR-NEXT:     }
// VECTOR-NEXT: }
