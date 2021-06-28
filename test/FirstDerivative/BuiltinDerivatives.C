// RUN: %cladclang %s -I%S/../../include -Xclang -verify -oBuiltinDerivatives.out -lm 2>&1 | FileCheck %s
// RUN: ./BuiltinDerivatives.out | FileCheck -check-prefix=CHECK-EXEC %s

//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

extern "C" int printf(const char* fmt, ...);

float f1(float x) {
  return sin(x);
}

// CHECK: float f1_darg0(float x) {
// CHECK-NEXT: float _d_x = 1;
// CHECK-NEXT: return custom_derivatives::sin_darg0(x) * _d_x;
// CHECK-NEXT: }

float f2(float x) {
  return cos(x);
}

// CHECK: float f2_darg0(float x) {
// CHECK-NEXT: float _d_x = 1;
// CHECK-NEXT: custom_derivatives::cos_darg0(x) * _d_x;
// CHECK-NEXT: }

float f3(float x, float y) {
  return sin(x) + sin(y);
}

// CHECK: float f3_darg0(float x, float y) {
// CHECK-NEXT: float _d_x = 1;
// CHECK-NEXT: float _d_y = 0;
// CHECK-NEXT: return custom_derivatives::sin_darg0(x) * _d_x + custom_derivatives::sin_darg0(y) * _d_y;
// CHECK-NEXT: }

// CHECK: float f3_darg1(float x, float y) {
// CHECK-NEXT: float _d_x = 0;
// CHECK-NEXT: float _d_y = 1;
// CHECK-NEXT: return custom_derivatives::sin_darg0(x) * _d_x + custom_derivatives::sin_darg0(y) * _d_y;
// CHECK-NEXT: }

float f4(float x, float y) {
  return sin(x * x) + sin(y * y);
}

// CHECK: float f4_darg0(float x, float y) {
// CHECK-NEXT: float _d_x = 1;
// CHECK-NEXT: float _d_y = 0;
// CHECK-NEXT: return custom_derivatives::sin_darg0(x * x) * (_d_x * x + x * _d_x) + custom_derivatives::sin_darg0(y * y) * (_d_y * y + y * _d_y);
// CHECK-NEXT: }

// CHECK: float f4_darg1(float x, float y) {
// CHECK-NEXT: float _d_x = 0;
// CHECK-NEXT: float _d_y = 1;
// CHECK-NEXT: return custom_derivatives::sin_darg0(x * x) * (_d_x * x + x * _d_x) + custom_derivatives::sin_darg0(y * y) * (_d_y * y + y * _d_y);
// CHECK-NEXT: }

float f5(float x) {
  return exp(x);
}

// CHECK: float f5_darg0(float x) {
// CHECK-NEXT: float _d_x = 1;
// CHECK-NEXT: return custom_derivatives::exp_darg0(x) * _d_x;
// CHECK-NEXT: }

float f6(float x) {
  return exp(x * x);
}

// CHECK: float f6_darg0(float x) {
// CHECK-NEXT: float _d_x = 1;
// CHECK-NEXT: return custom_derivatives::exp_darg0(x * x) * (_d_x * x + x * _d_x);
// CHECK-NEXT: }

float f7(float x) {
  return pow(x, 2.0);
}

// CHECK: float f7_darg0(float x) {
// CHECK-NEXT:   float _d_x = 1;
// CHECK-NEXT:   return custom_derivatives::pow_darg0(x, 2.) * (_d_x + 0.);
// CHECK-NEXT: }

void f7_grad(float x, float* _d_x);

// CHECK: void f7_grad(float x, float *_d_x) {
// CHECK-NEXT:     float _t0;
// CHECK-NEXT:     _t0 = x;
// CHECK-NEXT:     {{.*}} f7_return = pow(_t0, 2.);
// CHECK-NEXT:     goto _label0;
// CHECK-NEXT:   _label0:
// CHECK-NEXT:     {
// CHECK-NEXT:         typename {{.*}} _grad0 = 0.;
// CHECK-NEXT:         typename {{.*}} _grad1 = 0.;
// CHECK-NEXT:         custom_derivatives::pow_grad(_t0, 2., &_grad0, &_grad1);
// CHECK-NEXT:         typename {{.*}} _r0 = 1 * _grad0;
// CHECK-NEXT:         *_d_x += _r0;
// CHECK-NEXT:         typename {{.*}} _r1 = 1 * _grad1;
// CHECK-NEXT:     }
// CHECK-NEXT: }

double f8(float x) {
  return pow(x, 2);
}

// CHECK: double f8_darg0(float x) {
// CHECK-NEXT:   float _d_x = 1;
// CHECK-NEXT:   return custom_derivatives::pow_darg0(x, 2) * (_d_x + 0);
// CHECK-NEXT: }

void f8_grad(float x, double* _d_x);

// CHECK: void f8_grad(float x, double *_d_x) {
// CHECK-NEXT:     float _t0;
// CHECK-NEXT:     _t0 = x;
// CHECK-NEXT:     typename {{.*}} f8_return = pow(_t0, 2);
// CHECK-NEXT:     goto _label0;
// CHECK-NEXT:   _label0:
// CHECK-NEXT:     {
// CHECK-NEXT:         typename {{.*}} _grad0 = 0.;
// CHECK-NEXT:         typename {{.*}} _grad1 = 0.;
// CHECK-NEXT:         custom_derivatives::pow_grad(_t0, 2, &_grad0, &_grad1);
// CHECK-NEXT:         typename {{.*}} _r0 = 1 * _grad0;
// CHECK-NEXT:         *_d_x += _r0;
// CHECK-NEXT:         typename {{.*}} _r1 = 1 * _grad1;
// CHECK-NEXT:     }
// CHECK-NEXT: }

float f9(float x, float y) {
  return pow(x, y);
}

// CHECK: float f9_darg0(float x, float y) {
// CHECK-NEXT:     float _d_x = 1;
// CHECK-NEXT:     float _d_y = 0;
// CHECK-NEXT:     return custom_derivatives::pow_darg0(x, y) * (_d_x + _d_y);
// CHECK-NEXT: }

void f9_grad(float x, float y, float* _d_x, float* _d_y);

// CHECK: void f9_grad(float x, float y, float *_d_x, float *_d_y) {
// CHECK-NEXT:     float _t0;
// CHECK-NEXT:     float _t1;
// CHECK-NEXT:     _t0 = x;
// CHECK-NEXT:     _t1 = y;
// CHECK-NEXT:     float f9_return = pow(_t0, _t1);
// CHECK-NEXT:     goto _label0;
// CHECK-NEXT:   _label0:
// CHECK-NEXT:     {
// CHECK-NEXT:         float _grad0 = 0.F;
// CHECK-NEXT:         float _grad1 = 0.F;
// CHECK-NEXT:         custom_derivatives::pow_grad(_t0, _t1, &_grad0, &_grad1);
// CHECK-NEXT:         float _r0 = 1 * _grad0;
// CHECK-NEXT:         *_d_x += _r0;
// CHECK-NEXT:         float _r1 = 1 * _grad1;
// CHECK-NEXT:         *_d_y += _r1;
// CHECK-NEXT:     }
// CHECK-NEXT: }

double f10(float x, int y) {
  return pow(x, y);
}

// CHECK: double f10_darg0(float x, int y) {
// CHECK-NEXT:   float _d_x = 1;
// CHECK-NEXT:   int _d_y = 0;
// CHECK-NEXT:   return custom_derivatives::pow_darg0(x, y) * (_d_x + _d_y);
// CHECK-NEXT: }

void f10_grad(float x, int y, double* _d_x, double* _d_y);

// CHECK: void f10_grad(float x, int y, double *_d_x, double *_d_y) {
// CHECK-NEXT:     float _t0;
// CHECK-NEXT:     int _t1;
// CHECK-NEXT:     _t0 = x;
// CHECK-NEXT:     _t1 = y;
// CHECK-NEXT:     typename {{.*}} f10_return = pow(_t0, _t1);
// CHECK-NEXT:     goto _label0;
// CHECK-NEXT:   _label0:
// CHECK-NEXT:     {
// CHECK-NEXT:         typename {{.*}} _grad0 = 0.;
// CHECK-NEXT:         typename {{.*}} _grad1 = 0.;
// CHECK-NEXT:         custom_derivatives::pow_grad(_t0, _t1, &_grad0, &_grad1);
// CHECK-NEXT:         typename {{.*}} _r0 = 1 * _grad0;
// CHECK-NEXT:         *_d_x += _r0;
// CHECK-NEXT:         typename {{.*}} _r1 = 1 * _grad1;
// CHECK-NEXT:         *_d_y += _r1;
// CHECK-NEXT:     }
// CHECK-NEXT: }

int main () { //expected-no-diagnostics
  float f_result[2];
  double d_result[2];

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

  d_result[0] = 0;
  clad::gradient(f8);
  f8_grad(3, d_result);
  printf("Result is = %f\n", d_result[0]); //CHECK-EXEC: Result is = 6.000000

  auto f9_darg0 = clad::differentiate(f9, 0);
  printf("Result is = %f\n", f9_darg0.execute(3, 4)); //CHECK-EXEC: Result is = 108.000000

  f_result[0] = f_result[1] = 0;
  clad::gradient(f9);
  f9_grad(3, 4, &f_result[0], &f_result[1]);
  printf("Result is = {%f, %f}\n", f_result[0], f_result[1]); //CHECK-EXEC: Result is = {108.000000, 88.987595}

  auto f10_darg0 = clad::differentiate(f10, 0);
  printf("Result is = %f\n", f10_darg0.execute(3, 4)); //CHECK-EXEC: Result is = 108.000000

  d_result[0] = d_result[1] = 0;
  clad::gradient(f10);
  f10_grad(3, 4, &d_result[0], &d_result[1]);
  printf("Result is = {%f, %f}\n", d_result[0], d_result[1]); //CHECK-EXEC: Result is = {108.000000, 88.987597}

  return 0;
}
