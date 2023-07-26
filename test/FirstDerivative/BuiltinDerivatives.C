// RUN: %cladclang %s -I%S/../../include -Xclang -verify -oBuiltinDerivatives.out 2>&1 | FileCheck %s
// RUN: ./BuiltinDerivatives.out | FileCheck -check-prefix=CHECK-EXEC %s

//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"
#include "../TestUtils.h"
extern "C" int printf(const char* fmt, ...);

float f1(float x) {
  return sin(x);
}

// CHECK: float f1_darg0(float x) {
// CHECK-NEXT:     float _d_x = 1;
// CHECK-NEXT:     {{(clad::)?}}ValueAndPushforward<float, float> _t0 = clad::custom_derivatives{{(::std)?}}::sin_pushforward(x, _d_x);
// CHECK-NEXT:     return _t0.pushforward;
// CHECK-NEXT: }

float f2(float x) {
  return cos(x);
}

// CHECK: float f2_darg0(float x) {
// CHECK-NEXT:     float _d_x = 1;
// CHECK-NEXT:     {{(clad::)?}}ValueAndPushforward<float, float> _t0 = clad::custom_derivatives{{(::std)?}}::cos_pushforward(x, _d_x);
// CHECK-NEXT:     return _t0.pushforward;
// CHECK-NEXT: }

float f3(float x, float y) {
  return sin(x) + sin(y);
}

// CHECK: float f3_darg0(float x, float y) {
// CHECK-NEXT:     float _d_x = 1;
// CHECK-NEXT:     float _d_y = 0;
// CHECK-NEXT:     {{(clad::)?}}ValueAndPushforward<float, float> _t0 = clad::custom_derivatives{{(::std)?}}::sin_pushforward(x, _d_x);
// CHECK-NEXT:     {{(clad::)?}}ValueAndPushforward<float, float> _t1 = clad::custom_derivatives{{(::std)?}}::sin_pushforward(y, _d_y);
// CHECK-NEXT:     return _t0.pushforward + _t1.pushforward;
// CHECK-NEXT: }

// CHECK: float f3_darg1(float x, float y) {
// CHECK-NEXT:     float _d_x = 0;
// CHECK-NEXT:     float _d_y = 1;
// CHECK-NEXT:     {{(clad::)?}}ValueAndPushforward<float, float> _t0 = clad::custom_derivatives{{(::std)?}}::sin_pushforward(x, _d_x);
// CHECK-NEXT:     {{(clad::)?}}ValueAndPushforward<float, float> _t1 = clad::custom_derivatives{{(::std)?}}::sin_pushforward(y, _d_y);
// CHECK-NEXT:     return _t0.pushforward + _t1.pushforward;
// CHECK-NEXT: }

float f4(float x, float y) {
  return sin(x * x) + sin(y * y);
}

// CHECK: float f4_darg0(float x, float y) {
// CHECK-NEXT:     float _d_x = 1;
// CHECK-NEXT:     float _d_y = 0;
// CHECK-NEXT:     {{(clad::)?}}ValueAndPushforward<float, float> _t0 = clad::custom_derivatives{{(::std)?}}::sin_pushforward(x * x, _d_x * x + x * _d_x);
// CHECK-NEXT:     {{(clad::)?}}ValueAndPushforward<float, float> _t1 = clad::custom_derivatives{{(::std)?}}::sin_pushforward(y * y, _d_y * y + y * _d_y);
// CHECK-NEXT:     return _t0.pushforward + _t1.pushforward;
// CHECK-NEXT: }

// CHECK: float f4_darg1(float x, float y) {
// CHECK-NEXT:     float _d_x = 0;
// CHECK-NEXT:     float _d_y = 1;
// CHECK-NEXT:     {{(clad::)?}}ValueAndPushforward<float, float> _t0 = clad::custom_derivatives{{(::std)?}}::sin_pushforward(x * x, _d_x * x + x * _d_x);
// CHECK-NEXT:     {{(clad::)?}}ValueAndPushforward<float, float> _t1 = clad::custom_derivatives{{(::std)?}}::sin_pushforward(y * y, _d_y * y + y * _d_y);
// CHECK-NEXT:     return _t0.pushforward + _t1.pushforward;
// CHECK-NEXT: }

float f5(float x) {
  return exp(x);
}

// CHECK: float f5_darg0(float x) {
// CHECK-NEXT:     float _d_x = 1;
// CHECK-NEXT:     {{(clad::)?}}ValueAndPushforward<float, float> _t0 = clad::custom_derivatives{{(::std)?}}::exp_pushforward(x, _d_x);
// CHECK-NEXT:     return _t0.pushforward;
// CHECK-NEXT: }

float f6(float x) {
  return exp(x * x);
}

// CHECK: float f6_darg0(float x) {
// CHECK-NEXT:     float _d_x = 1;
// CHECK-NEXT:     {{(clad::)?}}ValueAndPushforward<float, float> _t0 = clad::custom_derivatives{{(::std)?}}::exp_pushforward(x * x, _d_x * x + x * _d_x);
// CHECK-NEXT:     return _t0.pushforward;
// CHECK-NEXT: }

float f7(float x) {
  return std::pow(x, 2.0);
}

// CHECK: float f7_darg0(float x) {
// CHECK-NEXT:     float _d_x = 1;
// CHECK-NEXT:     {{(clad::)?}}ValueAndPushforward<decltype(::std::pow(float(), double())), decltype(::std::pow(float(), double()))> _t0 = clad::custom_derivatives{{(::std)?}}::pow_pushforward(x, 2., _d_x, 0.);
// CHECK-NEXT:     return _t0.pushforward;
// CHECK-NEXT: }

void f7_grad(float x, clad::array_ref<float> _d_x);

// CHECK: void f7_grad(float x, clad::array_ref<float> _d_x) {
// CHECK-NEXT:     float _t0;
// CHECK-NEXT:     _t0 = x;
// CHECK-NEXT:     {
// CHECK-NEXT:         float _grad0 = 0.F;
// CHECK-NEXT:         double _grad1 = 0.;
// CHECK-NEXT:         {{(clad::)?}}custom_derivatives{{(::std)?}}::pow_pullback(_t0, 2., 1, &_grad0, &_grad1);
// CHECK-NEXT:         float _r0 = _grad0;
// CHECK-NEXT:         * _d_x += _r0;
// CHECK-NEXT:         double _r1 = _grad1;
// CHECK-NEXT:     }
// CHECK-NEXT: }

double f8(float x) {
  return std::pow(x, 2);
}

// CHECK: double f8_darg0(float x) {
// CHECK-NEXT:     float _d_x = 1;
// CHECK-NEXT:     {{(clad::)?}}ValueAndPushforward<decltype(::std::pow(float(), int())), decltype(::std::pow(float(), int()))> _t0 = clad::custom_derivatives{{(::std)?}}::pow_pushforward(x, 2, _d_x, 0);
// CHECK-NEXT:     return _t0.pushforward;
// CHECK-NEXT: }

void f8_grad(float x, clad::array_ref<float> _d_x);

// CHECK: void f8_grad(float x, clad::array_ref<float> _d_x) {
// CHECK-NEXT:     float _t0;
// CHECK-NEXT:     _t0 = x;
// CHECK-NEXT:     {
// CHECK-NEXT:         float _grad0 = 0.F;
// CHECK-NEXT:         int _grad1 = 0;
// CHECK-NEXT:         {{(clad::)?}}custom_derivatives{{(::std)?}}::pow_pullback(_t0, 2, 1, &_grad0, &_grad1);
// CHECK-NEXT:         float _r0 = _grad0;
// CHECK-NEXT:         * _d_x += _r0;
// CHECK-NEXT:         int _r1 = _grad1;
// CHECK-NEXT:     }
// CHECK-NEXT: }

float f9(float x, float y) {
  return std::pow(x, y);
}

// CHECK: float f9_darg0(float x, float y) {
// CHECK-NEXT:     float _d_x = 1;
// CHECK-NEXT:     float _d_y = 0;
// CHECK-NEXT:     {{(clad::)?}}ValueAndPushforward<decltype(::std::pow(float(), float())), decltype(::std::pow(float(), float()))> _t0 = clad::custom_derivatives{{(::std)?}}::pow_pushforward(x, y, _d_x, _d_y);
// CHECK-NEXT:     return _t0.pushforward;
// CHECK-NEXT: }

void f9_grad(float x, float y, clad::array_ref<float> _d_x, clad::array_ref<float> _d_y);

// CHECK: void f9_grad(float x, float y, clad::array_ref<float> _d_x, clad::array_ref<float> _d_y) {
// CHECK-NEXT:     float _t0;
// CHECK-NEXT:     float _t1;
// CHECK-NEXT:     _t0 = x;
// CHECK-NEXT:     _t1 = y;
// CHECK-NEXT:     {
// CHECK-NEXT:         float _grad0 = 0.F;
// CHECK-NEXT:         float _grad1 = 0.F;
// CHECK-NEXT:         {{(clad::)?}}custom_derivatives{{(::std)?}}::pow_pullback(_t0, _t1, 1, &_grad0, &_grad1);
// CHECK-NEXT:         float _r0 = _grad0;
// CHECK-NEXT:         * _d_x += _r0;
// CHECK-NEXT:         float _r1 = _grad1;
// CHECK-NEXT:         * _d_y += _r1;
// CHECK-NEXT:     }
// CHECK-NEXT: }

double f10(float x, int y) {
  return std::pow(x, y);
}

// CHECK: double f10_darg0(float x, int y) {
// CHECK-NEXT:     float _d_x = 1;
// CHECK-NEXT:     int _d_y = 0;
// CHECK-NEXT:     {{(clad::)?}}ValueAndPushforward<decltype(::std::pow(float(), int())), decltype(::std::pow(float(), int()))> _t0 = clad::custom_derivatives{{(::std)?}}::pow_pushforward(x, y, _d_x, _d_y);
// CHECK-NEXT:     return _t0.pushforward;
// CHECK-NEXT: }

void f10_grad(float x, int y, clad::array_ref<float> _d_x, clad::array_ref<int> _d_y);

// CHECK: void f10_grad(float x, int y, clad::array_ref<float> _d_x, clad::array_ref<int> _d_y) {
// CHECK-NEXT:     float _t0;
// CHECK-NEXT:     int _t1;
// CHECK-NEXT:     _t0 = x;
// CHECK-NEXT:     _t1 = y;
// CHECK-NEXT:     {
// CHECK-NEXT:         float _grad0 = 0.F;
// CHECK-NEXT:         int _grad1 = 0;
// CHECK-NEXT:         {{(clad::)?}}custom_derivatives{{(::std)?}}::pow_pullback(_t0, _t1, 1, &_grad0, &_grad1);
// CHECK-NEXT:         float _r0 = _grad0;
// CHECK-NEXT:         * _d_x += _r0;
// CHECK-NEXT:         int _r1 = _grad1;
// CHECK-NEXT:         * _d_y += _r1;
// CHECK-NEXT:     }
// CHECK-NEXT: }

double f11(double x, double y) {
  return std::pow((1.-x),2) + 100. * std::pow(y-std::pow(x,2),2);
}

// CHECK: void f11_grad(double x, double y, clad::array_ref<double> _d_x, clad::array_ref<double> _d_y) {
// CHECK-NEXT:     double _t0;
// CHECK-NEXT:     typename {{.*}} _t1;
// CHECK-NEXT:     double _t2;
// CHECK-NEXT:     double _t3;
// CHECK-NEXT:     _t0 = (1. - x);
// CHECK-NEXT:     _t2 = x;
// CHECK-NEXT:     _t3 = y - std::pow(_t2, 2);
// CHECK-NEXT:     _t1 = std::pow(_t3, 2);
// CHECK-NEXT:     {
// CHECK-NEXT:         double _grad0 = 0.;
// CHECK-NEXT:         int _grad1 = 0;
// CHECK-NEXT:         {{(clad::)?}}custom_derivatives{{(::std)?}}::pow_pullback(_t0, 2, 1, &_grad0, &_grad1);
// CHECK-NEXT:         double _r0 = _grad0;
// CHECK-NEXT:         * _d_x += -_r0;
// CHECK-NEXT:         int _r1 = _grad1;
// CHECK-NEXT:         double _r2 = 1 * _t1;
// CHECK-NEXT:         double _r3 = 100. * 1;
// CHECK-NEXT:         double _grad4 = 0.;
// CHECK-NEXT:         int _grad5 = 0;
// CHECK-NEXT:         {{(clad::)?}}custom_derivatives{{(::std)?}}::pow_pullback(_t3, 2, _r3, &_grad4, &_grad5);
// CHECK-NEXT:         double _r4 = _grad4;
// CHECK-NEXT:         * _d_y += _r4;
// CHECK-NEXT:         double _grad2 = 0.;
// CHECK-NEXT:         int _grad3 = 0;
// CHECK-NEXT:         {{(clad::)?}}custom_derivatives{{(::std)?}}::pow_pullback(_t2, 2, -_r4, &_grad2, &_grad3);
// CHECK-NEXT:         double _r5 = _grad2;
// CHECK-NEXT:         * _d_x += _r5;
// CHECK-NEXT:         int _r6 = _grad3;
// CHECK-NEXT:         int _r7 = _grad5;
// CHECK-NEXT:     }
// CHECK-NEXT: }

double f12(double a, double b) { return std::fma(a, b, b); }

//CHECK: double f12_darg1(double a, double b) {
//CHECK-NEXT:     double _d_a = 0;
//CHECK-NEXT:     double _d_b = 1;
//CHECK-NEXT:     {{(clad::)?}}ValueAndPushforward<decltype(::std::fma(double(), double(), double())), decltype(::std::fma(double(), double(), double()))> _t0 = clad::custom_derivatives::fma_pushforward(a, b, b, _d_a, _d_b, _d_b);
//CHECK-NEXT:     return _t0.pushforward;
//CHECK-NEXT: }

int main () { //expected-no-diagnostics
  float f_result[2];
  double d_result[2];
  int i_result[1];

  auto f1_darg0 = clad::differentiate(f1, 0);
  printf("Result is = %f\n", f1_darg0.execute(60)); // CHECK-EXEC: Result is = -0.952413

  auto f2_darg0 = clad::differentiate(f2, 0);
  printf("Result is = %f\n", f2_darg0.execute(60)); //CHECK-EXEC: Result is = 0.304811

  auto f3_darg0 = clad::differentiate(f3, 0);
  printf("Result is = %f\n", f3_darg0.execute(60, 30)); //CHECK-EXEC: Result is = -0.952413

  auto f3_darg1 = clad::differentiate(f3, 1);
  printf("Result is = %f\n", f3_darg1.execute(60, 30)); //CHECK-EXEC: Result is = 0.154251

  auto f4_darg0 = clad::differentiate(f4, 0);
  printf("Result is = %f\n", f4_darg0.execute(60, 30)); //CHECK-EXEC: Result is = 115.805412

  auto f4_darg1 = clad::differentiate(f4, 1);
  printf("Result is = %f\n", f4_darg1.execute(60, 30)); //CHECK-EXEC: Result is = 3.974802

  auto f5_darg0 = clad::differentiate(f5, 0);
  printf("Result is = %f\n", f5_darg0.execute(3)); //CHECK-EXEC: Result is = 20.085537

  auto f6_darg0 = clad::differentiate(f6, 0);
  printf("Result is = %f\n", f6_darg0.execute(3)); //CHECK-EXEC: Result is = 48618.503906

  auto f7_darg0 = clad::differentiate(f7, 0);
  printf("Result is = %f\n", f7_darg0.execute(3)); //CHECK-EXEC: Result is = 6.000000

  f_result[0] = 0;
  clad::gradient(f7);
  f7_grad(3, f_result);
  printf("Result is = %f\n", f_result[0]); //CHECK-EXEC: Result is = 6.000000

  auto f8_darg0 = clad::differentiate(f8, 0);
  printf("Result is = %f\n", f8_darg0.execute(3)); //CHECK-EXEC: Result is = 6.000000

  f_result[0] = 0;
  clad::gradient(f8);
  f8_grad(3, f_result);
  printf("Result is = %f\n", f_result[0]); //CHECK-EXEC: Result is = 6.000000

  auto f9_darg0 = clad::differentiate(f9, 0);
  printf("Result is = %f\n", f9_darg0.execute(3, 4)); //CHECK-EXEC: Result is = 108.000000

  f_result[0] = f_result[1] = 0;
  clad::gradient(f9);
  f9_grad(3, 4, &f_result[0], &f_result[1]);
  printf("Result is = {%f, %f}\n", f_result[0], f_result[1]); //CHECK-EXEC: Result is = {108.000000, 88.987595}

  auto f10_darg0 = clad::differentiate(f10, 0);
  printf("Result is = %f\n", f10_darg0.execute(3, 4)); //CHECK-EXEC: Result is = 108.000000

  f_result[0] = f_result[1] = 0;
  i_result[0] = 0;
  clad::gradient(f10);
  f10_grad(3, 4, &f_result[0], &i_result[0]);
  printf("Result is = {%f, %d}\n", f_result[0], i_result[0]); //CHECK-EXEC: Result is = {108.000000, 88}

  INIT_GRADIENT(f11);

  TEST_GRADIENT(f11, /*numOfDerivativeArgs=*/2, -1, 1, &d_result[0], &d_result[1]); // CHECK-EXEC: {-4.00, 0.00}

  auto f12_darg1 = clad::differentiate(f12, 1);
  printf("Result is = %f\n", f12_darg1.execute(2, 1)); //CHECK-EXEC: Result is = 3.000000

  return 0;
}
