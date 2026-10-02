// RUN: %cladclang %s -I%S/../../include -o %t -Xclang -verify 2>&1 | %filecheck %s
// RUN: %t | %filecheck_exec %s
// expected-no-diagnostics

#include "clad/Differentiator/Differentiator.h"
#include <cmath>
#include <cstdio>

double sin_fn(double x) { return std::sin(x); }
double exp_fn(double x) { return std::exp(x); }
double nested_fn(double x) { return 2 * std::sin(x * x); }

int main() {
  auto s3 = clad::differentiate<3>(sin_fn, "x");
  auto s4 = clad::differentiate<4>(sin_fn, "x");
  auto e3 = clad::differentiate<3>(exp_fn, "x");
  auto n3 = clad::differentiate<3>(nested_fn, "x");
  printf("sin %.6f %.6f\n", s3.execute(3.), s4.execute(3.));
  printf("exp %.6f\n", e3.execute(0.));
  printf("nested %.6f\n", n3.execute(1.));
  // CHECK-EXEC: sin 0.989992 0.141120
  // CHECK-EXEC-NEXT: exp 1.000000
  // CHECK-EXEC-NEXT: nested -28.840141
}

// Generated derivative bodies.
// CHECK-LABEL: double sin_fn_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     ValueAndPushforward<double, double> _t0 = clad::custom_derivatives::std::sin_pushforward(x, _d_x);
// CHECK-NEXT:     return _t0.pushforward;
// CHECK-NEXT: }

// CHECK-LABEL: double sin_fn_d2arg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     double _d__d_x = 0;
// CHECK-NEXT:     double _d_x0 = 1;
// CHECK-NEXT:     clad::ValueAndPushforward<ValueAndPushforward<double, double>, ValueAndPushforward<double, double> > _t0 = clad::custom_derivatives::std::sin_pushforward_pushforward(x, _d_x0, _d_x, _d__d_x);
// CHECK-NEXT:     ValueAndPushforward<double, double> _d__t0 = _t0.pushforward;
// CHECK-NEXT:     ValueAndPushforward<double, double> _t00 = _t0.value;
// CHECK-NEXT:     return _d__t0.pushforward;
// CHECK-NEXT: }

// CHECK-LABEL: double sin_fn_d3arg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     double _d__d_x = 0;
// CHECK-NEXT:     double _d_x0 = 1;
// CHECK-NEXT:     double _d__d__d_x = 0;
// CHECK-NEXT:     double _d__d_x0 = 0;
// CHECK-NEXT:     double _d__d_x00 = 0;
// CHECK-NEXT:     double _d_x00 = 1;
// CHECK-NEXT:     clad::ValueAndPushforward<clad::ValueAndPushforward<ValueAndPushforward<double, double>, ValueAndPushforward<double, double> >, clad::ValueAndPushforward<ValueAndPushforward<double, double>, ValueAndPushforward<double, double> > > _t0 = clad::custom_derivatives::std::sin_pushforward_pushforward_pushforward(x, _d_x00, _d_x0, _d__d_x0, _d_x, _d__d_x00, _d__d_x, _d__d__d_x);
// CHECK-NEXT:     clad::ValueAndPushforward<ValueAndPushforward<double, double>, ValueAndPushforward<double, double> > _d__t0 = _t0.pushforward;
// CHECK-NEXT:     clad::ValueAndPushforward<ValueAndPushforward<double, double>, ValueAndPushforward<double, double> > _t00 = _t0.value;
// CHECK-NEXT:     ValueAndPushforward<double, double> _d__d__t0 = _d__t0.pushforward;
// CHECK-NEXT:     ValueAndPushforward<double, double> _d__t00 = _t00.pushforward;
// CHECK-NEXT:     ValueAndPushforward<double, double> _d__t000 = _d__t0.value;
// CHECK-NEXT:     ValueAndPushforward<double, double> _t000 = _t00.value;
// CHECK-NEXT:     return _d__d__t0.pushforward;
// CHECK-NEXT: }

// CHECK-LABEL: double sin_fn_d4arg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     double _d__d_x = 0;
// CHECK-NEXT:     double _d_x0 = 1;
// CHECK-NEXT:     double _d__d__d_x = 0;
// CHECK-NEXT:     double _d__d_x0 = 0;
// CHECK-NEXT:     double _d__d_x00 = 0;
// CHECK-NEXT:     double _d_x00 = 1;
// CHECK-NEXT:     double _d__d__d__d_x = 0;
// CHECK-NEXT:     double _d__d__d_x0 = 0;
// CHECK-NEXT:     double _d__d__d_x00 = 0;
// CHECK-NEXT:     double _d__d_x01 = 0;
// CHECK-NEXT:     double _d__d__d_x000 = 0;
// CHECK-NEXT:     double _d__d_x000 = 0;
// CHECK-NEXT:     double _d__d_x001 = 0;
// CHECK-NEXT:     double _d_x000 = 1;
// CHECK-NEXT:     clad::ValueAndPushforward<clad::ValueAndPushforward<clad::ValueAndPushforward<ValueAndPushforward<double, double>, ValueAndPushforward<double, double> >, clad::ValueAndPushforward<ValueAndPushforward<double, double>, ValueAndPushforward<double, double> > >, clad::ValueAndPushforward<clad::ValueAndPushforward<ValueAndPushforward<double, double>, ValueAndPushforward<double, double> >, clad::ValueAndPushforward<ValueAndPushforward<double, double>, ValueAndPushforward<double, double> > > > _t0 = clad::custom_derivatives::std::sin_pushforward_pushforward_pushforward_pushforward(x, _d_x000, _d_x00, _d__d_x01, _d_x0, _d__d_x000, _d__d_x0, _d__d__d_x0, _d_x, _d__d_x001, _d__d_x00, _d__d__d_x00, _d__d_x, _d__d__d_x000, _d__d__d_x, _d__d__d__d_x);
// CHECK-NEXT:     clad::ValueAndPushforward<clad::ValueAndPushforward<ValueAndPushforward<double, double>, ValueAndPushforward<double, double> >, clad::ValueAndPushforward<ValueAndPushforward<double, double>, ValueAndPushforward<double, double> > > _d__t0 = _t0.pushforward;
// CHECK-NEXT:     clad::ValueAndPushforward<clad::ValueAndPushforward<ValueAndPushforward<double, double>, ValueAndPushforward<double, double> >, clad::ValueAndPushforward<ValueAndPushforward<double, double>, ValueAndPushforward<double, double> > > _t00 = _t0.value;
// CHECK-NEXT:     clad::ValueAndPushforward<ValueAndPushforward<double, double>, ValueAndPushforward<double, double> > _d__d__t0 = _d__t0.pushforward;
// CHECK-NEXT:     clad::ValueAndPushforward<ValueAndPushforward<double, double>, ValueAndPushforward<double, double> > _d__t00 = _t00.pushforward;
// CHECK-NEXT:     clad::ValueAndPushforward<ValueAndPushforward<double, double>, ValueAndPushforward<double, double> > _d__t000 = _d__t0.value;
// CHECK-NEXT:     clad::ValueAndPushforward<ValueAndPushforward<double, double>, ValueAndPushforward<double, double> > _t000 = _t00.value;
// CHECK-NEXT:     ValueAndPushforward<double, double> _d__d__d__t0 = _d__d__t0.pushforward;
// CHECK-NEXT:     ValueAndPushforward<double, double> _d__d__t00 = _d__t00.pushforward;
// CHECK-NEXT:     ValueAndPushforward<double, double> _d__d__t000 = _d__t000.pushforward;
// CHECK-NEXT:     ValueAndPushforward<double, double> _d__t001 = _t000.pushforward;
// CHECK-NEXT:     ValueAndPushforward<double, double> _d__d__t0000 = _d__d__t0.value;
// CHECK-NEXT:     ValueAndPushforward<double, double> _d__t0000 = _d__t00.value;
// CHECK-NEXT:     ValueAndPushforward<double, double> _d__t0001 = _d__t000.value;
// CHECK-NEXT:     ValueAndPushforward<double, double> _t0000 = _t000.value;
// CHECK-NEXT:     return _d__d__d__t0.pushforward;
// CHECK-NEXT: }

// CHECK-LABEL: double exp_fn_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     ValueAndPushforward<double, double> _t0 = clad::custom_derivatives::std::exp_pushforward(x, _d_x);
// CHECK-NEXT:     return _t0.pushforward;
// CHECK-NEXT: }

// CHECK-LABEL: double exp_fn_d2arg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     double _d__d_x = 0;
// CHECK-NEXT:     double _d_x0 = 1;
// CHECK-NEXT:     clad::ValueAndPushforward<ValueAndPushforward<double, double>, ValueAndPushforward<double, double> > _t0 = clad::custom_derivatives::std::exp_pushforward_pushforward(x, _d_x0, _d_x, _d__d_x);
// CHECK-NEXT:     ValueAndPushforward<double, double> _d__t0 = _t0.pushforward;
// CHECK-NEXT:     ValueAndPushforward<double, double> _t00 = _t0.value;
// CHECK-NEXT:     return _d__t0.pushforward;
// CHECK-NEXT: }

// CHECK-LABEL: double exp_fn_d3arg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     double _d__d_x = 0;
// CHECK-NEXT:     double _d_x0 = 1;
// CHECK-NEXT:     double _d__d__d_x = 0;
// CHECK-NEXT:     double _d__d_x0 = 0;
// CHECK-NEXT:     double _d__d_x00 = 0;
// CHECK-NEXT:     double _d_x00 = 1;
// CHECK-NEXT:     clad::ValueAndPushforward<clad::ValueAndPushforward<ValueAndPushforward<double, double>, ValueAndPushforward<double, double> >, clad::ValueAndPushforward<ValueAndPushforward<double, double>, ValueAndPushforward<double, double> > > _t0 = clad::custom_derivatives::std::exp_pushforward_pushforward_pushforward(x, _d_x00, _d_x0, _d__d_x0, _d_x, _d__d_x00, _d__d_x, _d__d__d_x);
// CHECK-NEXT:     clad::ValueAndPushforward<ValueAndPushforward<double, double>, ValueAndPushforward<double, double> > _d__t0 = _t0.pushforward;
// CHECK-NEXT:     clad::ValueAndPushforward<ValueAndPushforward<double, double>, ValueAndPushforward<double, double> > _t00 = _t0.value;
// CHECK-NEXT:     ValueAndPushforward<double, double> _d__d__t0 = _d__t0.pushforward;
// CHECK-NEXT:     ValueAndPushforward<double, double> _d__t00 = _t00.pushforward;
// CHECK-NEXT:     ValueAndPushforward<double, double> _d__t000 = _d__t0.value;
// CHECK-NEXT:     ValueAndPushforward<double, double> _t000 = _t00.value;
// CHECK-NEXT:     return _d__d__t0.pushforward;
// CHECK-NEXT: }

// CHECK-LABEL: double nested_fn_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     ValueAndPushforward<double, double> _t0 = clad::custom_derivatives::std::sin_pushforward(x * x, _d_x * x + x * _d_x);
// CHECK-NEXT:     return 2 * _t0.pushforward;
// CHECK-NEXT: }

// CHECK-LABEL: double nested_fn_d2arg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     double _d__d_x = 0;
// CHECK-NEXT:     double _d_x0 = 1;
// CHECK-NEXT:     clad::ValueAndPushforward<ValueAndPushforward<double, double>, ValueAndPushforward<double, double> > _t0 = clad::custom_derivatives::std::sin_pushforward_pushforward(x * x, _d_x0 * x + x * _d_x0, _d_x * x + x * _d_x, _d__d_x * x + _d_x0 * _d_x + _d_x * _d_x0 + x * _d__d_x);
// CHECK-NEXT:     ValueAndPushforward<double, double> _d__t0 = _t0.pushforward;
// CHECK-NEXT:     ValueAndPushforward<double, double> _t00 = _t0.value;
// CHECK-NEXT:     return 2 * _d__t0.pushforward;
// CHECK-NEXT: }

// CHECK-LABEL: double nested_fn_d3arg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     double _d__d_x = 0;
// CHECK-NEXT:     double _d_x0 = 1;
// CHECK-NEXT:     double _d__d__d_x = 0;
// CHECK-NEXT:     double _d__d_x0 = 0;
// CHECK-NEXT:     double _d__d_x00 = 0;
// CHECK-NEXT:     double _d_x00 = 1;
// CHECK-NEXT:     clad::ValueAndPushforward<clad::ValueAndPushforward<ValueAndPushforward<double, double>, ValueAndPushforward<double, double> >, clad::ValueAndPushforward<ValueAndPushforward<double, double>, ValueAndPushforward<double, double> > > _t0 = clad::custom_derivatives::std::sin_pushforward_pushforward_pushforward(x * x, _d_x00 * x + x * _d_x00, _d_x0 * x + x * _d_x0, _d__d_x0 * x + _d_x00 * _d_x0 + _d_x0 * _d_x00 + x * _d__d_x0, _d_x * x + x * _d_x, _d__d_x00 * x + _d_x00 * _d_x + _d_x * _d_x00 + x * _d__d_x00, _d__d_x * x + _d_x0 * _d_x + _d_x * _d_x0 + x * _d__d_x, _d__d__d_x * x + _d__d_x0 * _d_x + _d__d_x00 * _d_x0 + _d_x00 * _d__d_x + _d__d_x * _d_x00 + _d_x0 * _d__d_x00 + _d_x * _d__d_x0 + x * _d__d__d_x);
// CHECK-NEXT:     clad::ValueAndPushforward<ValueAndPushforward<double, double>, ValueAndPushforward<double, double> > _d__t0 = _t0.pushforward;
// CHECK-NEXT:     clad::ValueAndPushforward<ValueAndPushforward<double, double>, ValueAndPushforward<double, double> > _t00 = _t0.value;
// CHECK-NEXT:     ValueAndPushforward<double, double> _d__d__t0 = _d__t0.pushforward;
// CHECK-NEXT:     ValueAndPushforward<double, double> _d__t00 = _t00.pushforward;
// CHECK-NEXT:     ValueAndPushforward<double, double> _d__t000 = _d__t0.value;
// CHECK-NEXT:     ValueAndPushforward<double, double> _t000 = _t00.value;
// CHECK-NEXT:     return 2 * _d__d__t0.pushforward;
// CHECK-NEXT: }
