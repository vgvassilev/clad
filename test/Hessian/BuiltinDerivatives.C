// RUN: %cladclang %s -lm -I%S/../../include -oHessianBuiltinDerivatives.out 2>&1 | FileCheck %s
// RUN: ./HessianBuiltinDerivatives.out | FileCheck -check-prefix=CHECK-EXEC %s

//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"
#include <math.h>

float f1(float x) {
  return sin(x) + cos(x);
}

// CHECK: float f1_darg0(float x) {
// CHECK-NEXT:     float _d_x = 1;
// CHECK-NEXT:     return custom_derivatives::sin_darg0(x) * _d_x + custom_derivatives::cos_darg0(x) * _d_x;
// CHECK-NEXT: }

// CHECK:  void f1_darg0_grad(float x, clad::array_ref<float> _d_x) {
// CHECK-NEXT:     float _d__d_x = 0;
// CHECK-NEXT:     float _t0;
// CHECK-NEXT:     float _t1;
// CHECK-NEXT:     float _t2;
// CHECK-NEXT:     float _t3;
// CHECK-NEXT:     float _t4;
// CHECK-NEXT:     float _t5;
// CHECK-NEXT:     float _d_x0 = 1;
// CHECK-NEXT:     _t1 = x;
// CHECK-NEXT:     _t2 = custom_derivatives::sin_darg0(_t1);
// CHECK-NEXT:     _t0 = _d_x0;
// CHECK-NEXT:     _t4 = x;
// CHECK-NEXT:     _t5 = custom_derivatives::cos_darg0(_t4);
// CHECK-NEXT:     _t3 = _d_x0;
// CHECK-NEXT:     float f1_darg0_return = _t2 * _t0 + _t5 * _t3;
// CHECK-NEXT:     goto _label0;
// CHECK-NEXT:   _label0:
// CHECK-NEXT:     {
// CHECK-NEXT:         float _r0 = 1 * _t0;
// CHECK-NEXT:         float _r1 = _r0 * custom_derivatives::sin_darg0_darg0(_t1);
// CHECK-NEXT:         * _d_x += _r1;
// CHECK-NEXT:         float _r2 = _t2 * 1;
// CHECK-NEXT:         _d__d_x += _r2;
// CHECK-NEXT:         float _r3 = 1 * _t3;
// CHECK-NEXT:         float _r4 = _r3 * custom_derivatives::cos_darg0_darg0(_t4);
// CHECK-NEXT:         * _d_x += _r4;
// CHECK-NEXT:         float _r5 = _t5 * 1;
// CHECK-NEXT:         _d__d_x += _r5;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK:  void f1_hessian(float x, clad::array_ref<float> hessianMatrix) {
// CHECK-NEXT:     f1_darg0_grad(x, hessianMatrix.slice(0UL, 1UL));
// CHECK-NEXT: }

float f2(float x) {
  return exp(x);
}

// CHECK: float f2_darg0(float x) {
// CHECK-NEXT:     float _d_x = 1;
// CHECK-NEXT:     return custom_derivatives::exp_darg0(x) * _d_x;
// CHECK-NEXT: }

// CHECK:  void f2_darg0_grad(float x, clad::array_ref<float> _d_x) {
// CHECK-NEXT:     float _d__d_x = 0;
// CHECK-NEXT:     float _t0;
// CHECK-NEXT:     float _t1;
// CHECK-NEXT:     float _t2;
// CHECK-NEXT:     float _d_x0 = 1;
// CHECK-NEXT:     _t1 = x;
// CHECK-NEXT:     _t2 = custom_derivatives::exp_darg0(_t1);
// CHECK-NEXT:     _t0 = _d_x0;
// CHECK-NEXT:     float f2_darg0_return = _t2 * _t0;
// CHECK-NEXT:     goto _label0;
// CHECK-NEXT:   _label0:
// CHECK-NEXT:     {
// CHECK-NEXT:         float _r0 = 1 * _t0;
// CHECK-NEXT:         float _r1 = _r0 * custom_derivatives::exp_darg0_darg0(_t1);
// CHECK-NEXT:         * _d_x += _r1;
// CHECK-NEXT:         float _r2 = _t2 * 1;
// CHECK-NEXT:         _d__d_x += _r2;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK:  void f2_hessian(float x, clad::array_ref<float> hessianMatrix) {
// CHECK-NEXT:     f2_darg0_grad(x, hessianMatrix.slice(0UL, 1UL));
// CHECK-NEXT: }

float f3(float x) {
  return log(x);
}

// CHECK: float f3_darg0(float x) {
// CHECK-NEXT:     float _d_x = 1;
// CHECK-NEXT:     return custom_derivatives::log_darg0(x) * _d_x;
// CHECK-NEXT: }

// CHECK:  void f3_darg0_grad(float x, clad::array_ref<float> _d_x) {
// CHECK-NEXT:     float _d__d_x = 0;
// CHECK-NEXT:     float _t0;
// CHECK-NEXT:     float _t1;
// CHECK-NEXT:     float _t2;
// CHECK-NEXT:     float _d_x0 = 1;
// CHECK-NEXT:     _t1 = x;
// CHECK-NEXT:     _t2 = custom_derivatives::log_darg0(_t1);
// CHECK-NEXT:     _t0 = _d_x0;
// CHECK-NEXT:     float f3_darg0_return = _t2 * _t0;
// CHECK-NEXT:     goto _label0;
// CHECK-NEXT:   _label0:
// CHECK-NEXT:     {
// CHECK-NEXT:         float _r0 = 1 * _t0;
// CHECK-NEXT:         float _r1 = _r0 * custom_derivatives::log_darg0_darg0(_t1);
// CHECK-NEXT:         * _d_x += _r1;
// CHECK-NEXT:         float _r2 = _t2 * 1;
// CHECK-NEXT:         _d__d_x += _r2;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK:  void f3_hessian(float x, clad::array_ref<float> hessianMatrix) {
// CHECK-NEXT:     f3_darg0_grad(x, hessianMatrix.slice(0UL, 1UL));
// CHECK-NEXT: }

float f4(float x) {
  return pow(x, 4.0F);
}

// CHECK: float f4_darg0(float x) {
// CHECK-NEXT:     float _d_x = 1;
// CHECK-NEXT:     return custom_derivatives::pow_darg0(x, 4.F) * (_d_x + 0.F);
// CHECK-NEXT: }

// CHECK:  void f4_darg0_grad(float x, clad::array_ref<float> _d_x) {
// CHECK-NEXT:     float _d__d_x = 0;
// CHECK-NEXT:     float _t0;
// CHECK-NEXT:     float _t1;
// CHECK-NEXT:     decltype(pow(float(), float())) _t2;
// CHECK-NEXT:     float _d_x0 = 1;
// CHECK-NEXT:     _t1 = x;
// CHECK-NEXT:     _t2 = custom_derivatives::pow_darg0(_t1, 4.F);
// CHECK-NEXT:     _t0 = (_d_x0 + 0.F);
// CHECK-NEXT:     float f4_darg0_return = _t2 * _t0;
// CHECK-NEXT:     goto _label0;
// CHECK-NEXT:   _label0:
// CHECK-NEXT:     {
// CHECK-NEXT:         float _r0 = 1 * _t0;
// CHECK-NEXT:         decltype(pow(float(), float())) _grad0 = 0.F;
// CHECK-NEXT:         decltype(pow(float(), float())) _grad1 = 0.F;
// CHECK-NEXT:         custom_derivatives::pow_darg0_grad(_t1, 4.F, &_grad0, &_grad1);
// CHECK-NEXT:         decltype(pow(float(), float())) _r1 = _r0 * _grad0;
// CHECK-NEXT:         * _d_x += _r1;
// CHECK-NEXT:         decltype(pow(float(), float())) _r2 = _r0 * _grad1;
// CHECK-NEXT:         float _r3 = _t2 * 1;
// CHECK-NEXT:         _d__d_x += _r3;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK:  void f4_hessian(float x, clad::array_ref<float> hessianMatrix) {
// CHECK-NEXT:     f4_darg0_grad(x, hessianMatrix.slice(0UL, 1UL));
// CHECK-NEXT: }

float f5(float x) {
  return pow(2.0F, x);
}

// CHECK: float f5_darg0(float x) {
// CHECK-NEXT:   float _d_x = 1;
// CHECK-NEXT:   return custom_derivatives::pow_darg0(2.F, x) * (0.F + _d_x);
// CHECK-NEXT: }

// CHECK: void f5_darg0_grad(float x, clad::array_ref<float> _d_x) {
// CHECK-NEXT:   float _d__d_x = 0;
// CHECK-NEXT:   float _t0;
// CHECK-NEXT:   float _t1;
// CHECK-NEXT:   decltype(pow(float(), float())) _t2;
// CHECK-NEXT:   float _d_x0 = 1;
// CHECK-NEXT:   _t1 = x;
// CHECK-NEXT:   _t2 = custom_derivatives::pow_darg0(2.F, _t1);
// CHECK-NEXT:   _t0 = (0.F + _d_x0);
// CHECK-NEXT:   float f5_darg0_return = _t2 * _t0;
// CHECK-NEXT:   goto _label0;
// CHECK-NEXT:   _label0:
// CHECK-NEXT:   {
// CHECK-NEXT:     float _r0 = 1 * _t0;
// CHECK-NEXT:     decltype(pow(float(), float())) _grad0 = 0.F;
// CHECK-NEXT:     decltype(pow(float(), float())) _grad1 = 0.F;
// CHECK-NEXT:     custom_derivatives::pow_darg0_grad(2.F, _t1, &_grad0, &_grad1);
// CHECK-NEXT:     decltype(pow(float(), float())) _r1 = _r0 * _grad0;
// CHECK-NEXT:     decltype(pow(float(), float())) _r2 = _r0 * _grad1;
// CHECK-NEXT:     * _d_x += _r2;
// CHECK-NEXT:     float _r3 = _t2 * 1;
// CHECK-NEXT:     _d__d_x += _r3;
// CHECK-NEXT:   }
// CHECK-NEXT: }

// CHECK: void f5_hessian(float x, clad::array_ref<float> hessianMatrix) {
// CHECK-NEXT:   f5_darg0_grad(x, hessianMatrix.slice(0UL, 1UL));
// CHECK-NEXT: }

float f6(float x, float y) {
  return pow(x, y);
}

// CHECK: float f6_darg0(float x, float y) {
// CHECK-NEXT:     float _d_x = 1;
// CHECK-NEXT:     float _d_y = 0;
// CHECK-NEXT:     return custom_derivatives::pow_darg0(x, y) * (_d_x + _d_y);
// CHECK-NEXT: }

// CHECK:  void f6_darg0_grad(float x, float y, clad::array_ref<float> _d_x, clad::array_ref<float> _d_y) {
// CHECK-NEXT:     float _d__d_x = 0;
// CHECK-NEXT:     float _d__d_y = 0;
// CHECK-NEXT:     float _t0;
// CHECK-NEXT:     float _t1;
// CHECK-NEXT:     float _t2;
// CHECK-NEXT:     decltype(pow(float(), float())) _t3;
// CHECK-NEXT:     float _d_x0 = 1;
// CHECK-NEXT:     float _d_y0 = 0;
// CHECK-NEXT:     _t1 = x;
// CHECK-NEXT:     _t2 = y;
// CHECK-NEXT:     _t3 = custom_derivatives::pow_darg0(_t1, _t2);
// CHECK-NEXT:     _t0 = (_d_x0 + _d_y0);
// CHECK-NEXT:     float f6_darg0_return = _t3 * _t0;
// CHECK-NEXT:     goto _label0;
// CHECK-NEXT:   _label0:
// CHECK-NEXT:     {
// CHECK-NEXT:         float _r0 = 1 * _t0;
// CHECK-NEXT:         decltype(pow(float(), float())) _grad0 = 0.F;
// CHECK-NEXT:         decltype(pow(float(), float())) _grad1 = 0.F;
// CHECK-NEXT:         custom_derivatives::pow_darg0_grad(_t1, _t2, &_grad0, &_grad1);
// CHECK-NEXT:         decltype(pow(float(), float())) _r1 = _r0 * _grad0;
// CHECK-NEXT:         * _d_x += _r1;
// CHECK-NEXT:         decltype(pow(float(), float())) _r2 = _r0 * _grad1;
// CHECK-NEXT:         * _d_y += _r2;
// CHECK-NEXT:         float _r3 = _t3 * 1;
// CHECK-NEXT:         _d__d_x += _r3;
// CHECK-NEXT:         _d__d_y += _r3;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK:  float f6_darg1(float x, float y) {
// CHECK-NEXT:     float _d_x = 0;
// CHECK-NEXT:     float _d_y = 1;
// CHECK-NEXT:     return custom_derivatives::pow_darg0(x, y) * (_d_x + _d_y);
// CHECK-NEXT: }

// CHECK:  void f6_darg1_grad(float x, float y, clad::array_ref<float> _d_x, clad::array_ref<float> _d_y) {
// CHECK-NEXT:     float _d__d_x = 0;
// CHECK-NEXT:     float _d__d_y = 0;
// CHECK-NEXT:     float _t0;
// CHECK-NEXT:     float _t1;
// CHECK-NEXT:     float _t2;
// CHECK-NEXT:     decltype(pow(float(), float())) _t3;
// CHECK-NEXT:     float _d_x0 = 0;
// CHECK-NEXT:     float _d_y0 = 1;
// CHECK-NEXT:     _t1 = x;
// CHECK-NEXT:     _t2 = y;
// CHECK-NEXT:     _t3 = custom_derivatives::pow_darg0(_t1, _t2);
// CHECK-NEXT:     _t0 = (_d_x0 + _d_y0);
// CHECK-NEXT:     float f6_darg1_return = _t3 * _t0;
// CHECK-NEXT:     goto _label0;
// CHECK-NEXT:   _label0:
// CHECK-NEXT:     {
// CHECK-NEXT:         float _r0 = 1 * _t0;
// CHECK-NEXT:         decltype(pow(float(), float())) _grad0 = 0.F;
// CHECK-NEXT:         decltype(pow(float(), float())) _grad1 = 0.F;
// CHECK-NEXT:         custom_derivatives::pow_darg0_grad(_t1, _t2, &_grad0, &_grad1);
// CHECK-NEXT:         decltype(pow(float(), float())) _r1 = _r0 * _grad0;
// CHECK-NEXT:         * _d_x += _r1;
// CHECK-NEXT:         decltype(pow(float(), float())) _r2 = _r0 * _grad1;
// CHECK-NEXT:         * _d_y += _r2;
// CHECK-NEXT:         float _r3 = _t3 * 1;
// CHECK-NEXT:         _d__d_x += _r3;
// CHECK-NEXT:         _d__d_y += _r3;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK:  void f6_hessian(float x, float y, clad::array_ref<float> hessianMatrix) {
// CHECK-NEXT:     f6_darg0_grad(x, y, hessianMatrix.slice(0UL, 1UL), hessianMatrix.slice(1UL, 1UL));
// CHECK-NEXT:     f6_darg1_grad(x, y, hessianMatrix.slice(2UL, 1UL), hessianMatrix.slice(3UL, 1UL));
// CHECK-NEXT: }

#define TEST1(F, x) {                                \
  result[0] = 0;                                     \
  auto h = clad::hessian(F);                         \
  h.execute(x, result);                              \
  printf("Result is = {%.2f}\n", result[0]);         \
}

#define TEST2(F, x, y) {                             \
  result[0] = result[1] = result[2] = result[3] = 0; \
  auto h = clad::hessian(F);                         \
  clad::array_ref<float> ar(result, 4);              \
  h.execute(x, y, ar);                           \
  printf("Result is = {%.2f, %.2f, %.2f, %.2f}\n",   \
         result[0], result[1], result[2], result[3]);\
}

int main() {
  float result[4];

  TEST1(f1, 0); // CHECK-EXEC: Result is = {-1.00}
  TEST1(f2, 1); // CHECK-EXEC: Result is = {2.72}
  TEST1(f3, 1); // CHECK-EXEC: Result is = {-1.00}
  TEST1(f4, 3); // CHECK-EXEC: Result is = {108.00}
  TEST1(f5, 3); // CHECK-EXEC: Result is = {12.32}
  TEST2(f6, 3, 4); // CHECK-EXEC: Result is = {108.00, 145.65, 108.00, 145.65}
}