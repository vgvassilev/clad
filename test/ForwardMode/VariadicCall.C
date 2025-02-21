// RUN: %cladclang -std=c++17 %s -I%S/../../include -oVariadicCall.out 2>&1 | %filecheck %s
// RUN: ./VariadicCall.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"

#include "../TestUtils.h"

template <typename... T>
double fn1(double x, T... y) {
    return (x * ... * y);
}

double fn2(double x, double y) {
    return fn1(x, y);
}

// CHECK: double fn2_darg0(double x, double y) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     double _d_y = 0;
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t0 = fn1_pushforward(x, y, _d_x, _d_y);
// CHECK-NEXT:     return _t0.pushforward;
// CHECK-NEXT: }

double fn3(double x, double y) {
    return fn1(x, y, y);
}

// CHECK: double fn3_darg0(double x, double y) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     double _d_y = 0;
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t0 = fn1_pushforward(x, y, y, _d_x, _d_y, _d_y);
// CHECK-NEXT:     return _t0.pushforward;
// CHECK-NEXT: }


int main() {
  INIT_DIFFERENTIATE(fn2, "x");
  TEST_DIFFERENTIATE(fn2, 1.0, 2.0); // CHECK-EXEC: {2.00}
  INIT_DIFFERENTIATE(fn3, "x");
  TEST_DIFFERENTIATE(fn3, 1.0, 2.0); // CHECK-EXEC: {4.00}
  return 0;
}

// CHECK: clad::ValueAndPushforward<double, double> fn1_pushforward(double x, double y, double _d_x, double _d_y) {
// CHECK-NEXT:    return {x * y, _d_x * y + x * _d_y};
// CHECK-NEXT:}

// CHECK: clad::ValueAndPushforward<double, double> fn1_pushforward(double x, double y, double y2, double _d_x, double _d_y, double _d_y2) {
// CHECK-NEXT:    double _t0 = x * y;
// CHECK-NEXT:    return {_t0 * y2, (_d_x * y + x * _d_y) * y2 + _t0 * _d_y2};
// CHECK-NEXT:}