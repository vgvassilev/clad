// RUN: %cladclang %s -I%S/../../include -oNestedCalls.out 2>&1 | FileCheck %s
// RUN: ./NestedCalls.out | FileCheck -check-prefix=CHECK-EXEC %s

//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"
#include <cmath>

extern "C" int printf(const char* fmt, ...);

double sq(double x) { return x * x; }

// CHECK: clad::ValueAndPushforward<double, double> sq_pushforward(double x, double _d_x) {
// CHECK-NEXT:     return {x * x, _d_x * x + x * _d_x};
// CHECK-NEXT: }

double one(double x) { return sq(std::sin(x)) + sq(std::cos(x)); }

// CHECK: clad::ValueAndPushforward<double, double> one_pushforward(double x, double _d_x) {
// CHECK-NEXT:     ValueAndPushforward<double, double> _t0 = clad::custom_derivatives::sin_pushforward(x, _d_x);
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t1 = sq_pushforward(_t0.value, _t0.pushforward);
// CHECK-NEXT:     ValueAndPushforward<double, double> _t2 = clad::custom_derivatives::cos_pushforward(x, _d_x);
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t3 = sq_pushforward(_t2.value, _t2.pushforward);
// CHECK-NEXT:     return {_t1.value + _t3.value, _t1.pushforward + _t3.pushforward};
// CHECK-NEXT: }

double f(double x, double y) {
  double t = one(x);
  return t * y;
}
// CHECK: double f_darg0(double x, double y) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     double _d_y = 0;
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t0 = one_pushforward(x, _d_x);
// CHECK-NEXT:     double _d_t = _t0.pushforward;
// CHECK-NEXT:     double t = _t0.value;
// CHECK-NEXT:     return _d_t * y + t * _d_y;
// CHECK-NEXT: }

//CHECK:   void sq_pullback(double x, double _d_y, clad::array_ref<double> _d_x) {
//CHECK-NEXT:       double _t0;
//CHECK-NEXT:       double _t1;
//CHECK-NEXT:       _t1 = x;
//CHECK-NEXT:       _t0 = x;
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r0 = _d_y * _t0;
//CHECK-NEXT:           * _d_x += _r0;
//CHECK-NEXT:           double _r1 = _t1 * _d_y;
//CHECK-NEXT:           * _d_x += _r1;
//CHECK-NEXT:       }
//CHECK-NEXT:   }

//CHECK:   void one_pullback(double x, double _d_y, clad::array_ref<double> _d_x) {
//CHECK-NEXT:       double _t0;
//CHECK-NEXT:       double _t1;
//CHECK-NEXT:       double _t2;
//CHECK-NEXT:       double _t3;
//CHECK-NEXT:       _t0 = x;
//CHECK-NEXT:       _t1 = std::sin(_t0);
//CHECK-NEXT:       _t2 = x;
//CHECK-NEXT:       _t3 = std::cos(_t2);
//CHECK-NEXT:       {
//CHECK-NEXT:           double _grad0 = 0.;
//CHECK-NEXT:           sq_pullback(_t1, _d_y, &_grad0);
//CHECK-NEXT:           double _r0 = _grad0;
//CHECK-NEXT:           double _r1 = _r0 * clad::custom_derivatives::sin_pushforward(_t0, 1.).pushforward;
//CHECK-NEXT:           * _d_x += _r1;
//CHECK-NEXT:           double _grad1 = 0.;
//CHECK-NEXT:           sq_pullback(_t3, _d_y, &_grad1);
//CHECK-NEXT:           double _r2 = _grad1;
//CHECK-NEXT:           double _r3 = _r2 * clad::custom_derivatives::cos_pushforward(_t2, 1.).pushforward;
//CHECK-NEXT:           * _d_x += _r3;
//CHECK-NEXT:       }
//CHECK-NEXT:   }

//CHECK:   void f_grad(double x, double y, clad::array_ref<double> _d_x, clad::array_ref<double> _d_y) {
//CHECK-NEXT:       double _t0;
//CHECK-NEXT:       double _d_t = 0;
//CHECK-NEXT:       double _t1;
//CHECK-NEXT:       double _t2;
//CHECK-NEXT:       _t0 = x;
//CHECK-NEXT:       double t = one(_t0);
//CHECK-NEXT:       _t2 = t;
//CHECK-NEXT:       _t1 = y;
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r1 = 1 * _t1;
//CHECK-NEXT:           _d_t += _r1;
//CHECK-NEXT:           double _r2 = _t2 * 1;
//CHECK-NEXT:           * _d_y += _r2;
//CHECK-NEXT:       }
//CHECK-NEXT:       {
//CHECK-NEXT:           double _grad0 = 0.;
//CHECK-NEXT:           one_pullback(_t0, _d_t, &_grad0);
//CHECK-NEXT:           double _r0 = _grad0;
//CHECK-NEXT:           * _d_x += _r0;
//CHECK-NEXT:       }
//CHECK-NEXT:   }

int main () { // expected-no-diagnostics
  auto df = clad::differentiate(f, 0);
  printf("%.2f\n", df.execute(1, 2)); // CHECK-EXEC: 0.00
  printf("%.2f\n", df.execute(10, 11)); // CHECK-EXEC: 0.00

  auto gradf = clad::gradient(f);
  double result[2] = {};
  gradf.execute(2, 3, &result[0], &result[1]);
  printf("{%.2f, %.2f}\n", result[0], result[1]); // CHECK-EXEC: {0.00, 1.00}
  return 0;
}
