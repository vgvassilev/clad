// RUN: %cladclang %s -I%S/../../include -oHessianBuiltinDerivatives.out 2>&1 | FileCheck %s
// RUN: ./HessianBuiltinDerivatives.out | FileCheck -check-prefix=CHECK-EXEC %s

//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"
#include <math.h>

float f1(float x) {
  return sin(x) + cos(x);
}

// CHECK: float f1_darg0(float x) {
// CHECK-NEXT:     float _d_x = 1;
// CHECK-NEXT:     ValueAndPushforward<float, float> _t0 = clad::custom_derivatives{{(::std)?}}::sin_pushforward(x, _d_x);
// CHECK-NEXT:     ValueAndPushforward<float, float> _t1 = clad::custom_derivatives{{(::std)?}}::cos_pushforward(x, _d_x);
// CHECK-NEXT:     return _t0.pushforward + _t1.pushforward;
// CHECK-NEXT: }

// CHECK: void sin_pushforward_pullback(float x, float d_x, ValueAndPushforward<float, float> _d_y, clad::array_ref<float> _d_x, clad::array_ref<float> _d_d_x) {
// CHECK-NEXT:     float _t0;
// CHECK-NEXT:     float _t1;
// CHECK-NEXT:     float _t2;
// CHECK-NEXT:     float _t3;
// CHECK-NEXT:     _t0 = x;
// CHECK-NEXT:     _t2 = x;
// CHECK-NEXT:     _t3 = ::std::cos(_t2);
// CHECK-NEXT:     _t1 = d_x;
// CHECK-NEXT:     goto _label0;
// CHECK-NEXT:   _label0:
// CHECK-NEXT:     {
// CHECK-NEXT:         float _r0 = _d_y.value * clad::custom_derivatives{{(::std)?}}::sin_pushforward(_t0, 1.F).pushforward;
// CHECK-NEXT:         * _d_x += _r0;
// CHECK-NEXT:         float _r1 = _d_y.pushforward * _t1;
// CHECK-NEXT:         float _r2 = _r1 * clad::custom_derivatives{{(::std)?}}::cos_pushforward(_t2, 1.F).pushforward;
// CHECK-NEXT:         * _d_x += _r2;
// CHECK-NEXT:         float _r3 = _t3 * _d_y.pushforward;
// CHECK-NEXT:         * _d_d_x += _r3;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: void cos_pushforward_pullback(float x, float d_x, ValueAndPushforward<float, float> _d_y, clad::array_ref<float> _d_x, clad::array_ref<float> _d_d_x) {
// CHECK-NEXT:     float _t0;
// CHECK-NEXT:     float _t1;
// CHECK-NEXT:     float _t2;
// CHECK-NEXT:     float _t3;
// CHECK-NEXT:     float _t4;
// CHECK-NEXT:     _t0 = x;
// CHECK-NEXT:     _t3 = x;
// CHECK-NEXT:     _t2 = ::std::sin(_t3);
// CHECK-NEXT:     _t4 = -1 * _t2;
// CHECK-NEXT:     _t1 = d_x;
// CHECK-NEXT:     goto _label0;
// CHECK-NEXT:   _label0:
// CHECK-NEXT:     {
// CHECK-NEXT:         float _r0 = _d_y.value * clad::custom_derivatives{{(::std)?}}::cos_pushforward(_t0, 1.F).pushforward;
// CHECK-NEXT:         * _d_x += _r0;
// CHECK-NEXT:         float _r1 = _d_y.pushforward * _t1;
// CHECK-NEXT:         float _r2 = _r1 * _t2;
// CHECK-NEXT:         float _r3 = -1 * _r1;
// CHECK-NEXT:         float _r4 = _r3 * clad::custom_derivatives{{(::std)?}}::sin_pushforward(_t3, 1.F).pushforward;
// CHECK-NEXT:         * _d_x += _r4;
// CHECK-NEXT:         float _r5 = _t4 * _d_y.pushforward;
// CHECK-NEXT:         * _d_d_x += _r5;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: void f1_darg0_grad(float x, clad::array_ref<float> _d_x) {
// CHECK-NEXT:     float _d__d_x = 0;
// CHECK-NEXT:     float _t0;
// CHECK-NEXT:     float _t1;
// CHECK-NEXT:     ValueAndPushforward<float, float> _d__t0 = {};
// CHECK-NEXT:     float _t2;
// CHECK-NEXT:     float _t3;
// CHECK-NEXT:     ValueAndPushforward<float, float> _d__t1 = {};
// CHECK-NEXT:     float _d_x0 = 1;
// CHECK-NEXT:     _t0 = x;
// CHECK-NEXT:     _t1 = _d_x0;
// CHECK-NEXT:     ValueAndPushforward<float, float> _t00 = clad::custom_derivatives{{(::std)?}}::sin_pushforward(_t0, _t1);
// CHECK-NEXT:     _t2 = x;
// CHECK-NEXT:     _t3 = _d_x0;
// CHECK-NEXT:     ValueAndPushforward<float, float> _t10 = clad::custom_derivatives{{(::std)?}}::cos_pushforward(_t2, _t3);
// CHECK-NEXT:     goto _label0;
// CHECK-NEXT:   _label0:
// CHECK-NEXT:     {
// CHECK-NEXT:         _d__t0.pushforward += 1;
// CHECK-NEXT:         _d__t1.pushforward += 1;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         float _grad2 = 0.F;
// CHECK-NEXT:         float _grad3 = 0.F;
// CHECK-NEXT:         cos_pushforward_pullback(_t2, _t3, _d__t1, &_grad2, &_grad3);
// CHECK-NEXT:         float _r2 = _grad2;
// CHECK-NEXT:         * _d_x += _r2;
// CHECK-NEXT:         float _r3 = _grad3;
// CHECK-NEXT:         _d__d_x += _r3;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         float _grad0 = 0.F;
// CHECK-NEXT:         float _grad1 = 0.F;
// CHECK-NEXT:         sin_pushforward_pullback(_t0, _t1, _d__t0, &_grad0, &_grad1);
// CHECK-NEXT:         float _r0 = _grad0;
// CHECK-NEXT:         * _d_x += _r0;
// CHECK-NEXT:         float _r1 = _grad1;
// CHECK-NEXT:         _d__d_x += _r1;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: void f1_hessian(float x, clad::array_ref<float> hessianMatrix) {
// CHECK-NEXT:     f1_darg0_grad(x, hessianMatrix.slice(0UL, 1UL));
// CHECK-NEXT: }

float f2(float x) {
  return exp(x);
}

// CHECK: float f2_darg0(float x) {
// CHECK-NEXT:     float _d_x = 1;
// CHECK-NEXT:     ValueAndPushforward<float, float> _t0 = clad::custom_derivatives{{(::std)?}}::exp_pushforward(x, _d_x);
// CHECK-NEXT:     return _t0.pushforward;
// CHECK-NEXT: }

// CHECK: void exp_pushforward_pullback(float x, float d_x, ValueAndPushforward<float, float> _d_y, clad::array_ref<float> _d_x, clad::array_ref<float> _d_d_x) {
// CHECK-NEXT:     float _t0;
// CHECK-NEXT:     float _t1;
// CHECK-NEXT:     float _t2;
// CHECK-NEXT:     float _t3;
// CHECK-NEXT:     _t0 = x;
// CHECK-NEXT:     _t2 = x;
// CHECK-NEXT:     _t3 = ::std::exp(_t2);
// CHECK-NEXT:     _t1 = d_x;
// CHECK-NEXT:     goto _label0;
// CHECK-NEXT:   _label0:
// CHECK-NEXT:     {
// CHECK-NEXT:         float _r0 = _d_y.value * clad::custom_derivatives{{(::std)?}}::exp_pushforward(_t0, 1.F).pushforward;
// CHECK-NEXT:         * _d_x += _r0;
// CHECK-NEXT:         float _r1 = _d_y.pushforward * _t1;
// CHECK-NEXT:         float _r2 = _r1 * clad::custom_derivatives{{(::std)?}}::exp_pushforward(_t2, 1.F).pushforward;
// CHECK-NEXT:         * _d_x += _r2;
// CHECK-NEXT:         float _r3 = _t3 * _d_y.pushforward;
// CHECK-NEXT:         * _d_d_x += _r3;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: void f2_darg0_grad(float x, clad::array_ref<float> _d_x) {
// CHECK-NEXT:     float _d__d_x = 0;
// CHECK-NEXT:     float _t0;
// CHECK-NEXT:     float _t1;
// CHECK-NEXT:     ValueAndPushforward<float, float> _d__t0 = {};
// CHECK-NEXT:     float _d_x0 = 1;
// CHECK-NEXT:     _t0 = x;
// CHECK-NEXT:     _t1 = _d_x0;
// CHECK-NEXT:     ValueAndPushforward<float, float> _t00 = clad::custom_derivatives{{(::std)?}}::exp_pushforward(_t0, _t1);
// CHECK-NEXT:     goto _label0;
// CHECK-NEXT:   _label0:
// CHECK-NEXT:     _d__t0.pushforward += 1;
// CHECK-NEXT:     {
// CHECK-NEXT:         float _grad0 = 0.F;
// CHECK-NEXT:         float _grad1 = 0.F;
// CHECK-NEXT:         exp_pushforward_pullback(_t0, _t1, _d__t0, &_grad0, &_grad1);
// CHECK-NEXT:         float _r0 = _grad0;
// CHECK-NEXT:         * _d_x += _r0;
// CHECK-NEXT:         float _r1 = _grad1;
// CHECK-NEXT:         _d__d_x += _r1;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: void f2_hessian(float x, clad::array_ref<float> hessianMatrix) {
// CHECK-NEXT:     f2_darg0_grad(x, hessianMatrix.slice(0UL, 1UL));
// CHECK-NEXT: }


float f3(float x) {
  return log(x);
}

// CHECK: float f3_darg0(float x) {
// CHECK-NEXT:     float _d_x = 1;
// CHECK-NEXT:     ValueAndPushforward<float, float> _t0 = clad::custom_derivatives{{(::std)?}}::log_pushforward(x, _d_x);
// CHECK-NEXT:     return _t0.pushforward;
// CHECK-NEXT: }

// CHECK: void log_pushforward_pullback(float x, float d_x, ValueAndPushforward<float, float> _d_y, clad::array_ref<float> _d_x, clad::array_ref<float> _d_d_x) {
// CHECK-NEXT:     float _t0;
// CHECK-NEXT:     double _t1;
// CHECK-NEXT:     double _t2;
// CHECK-NEXT:     double _t3;
// CHECK-NEXT:     _t0 = x;
// CHECK-NEXT:     _t2 = x;
// CHECK-NEXT:     _t3 = (1. / _t2);
// CHECK-NEXT:     _t1 = d_x;
// CHECK-NEXT:     goto _label0;
// CHECK-NEXT:   _label0:
// CHECK-NEXT:     {
// CHECK-NEXT:         float _r0 = _d_y.value * clad::custom_derivatives{{(::std)?}}::log_pushforward(_t0, 1.F).pushforward;
// CHECK-NEXT:         * _d_x += _r0;
// CHECK-NEXT:         double _r1 = _d_y.pushforward * _t1;
// CHECK-NEXT:         double _r2 = _r1 / _t2;
// CHECK-NEXT:         double _r3 = _r1 * -1. / (_t2 * _t2);
// CHECK-NEXT:         * _d_x += _r3;
// CHECK-NEXT:         double _r4 = _t3 * _d_y.pushforward;
// CHECK-NEXT:         * _d_d_x += _r4;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: void f3_darg0_grad(float x, clad::array_ref<float> _d_x) {
// CHECK-NEXT:     float _d__d_x = 0;
// CHECK-NEXT:     float _t0;
// CHECK-NEXT:     float _t1;
// CHECK-NEXT:     ValueAndPushforward<float, float> _d__t0 = {};
// CHECK-NEXT:     float _d_x0 = 1;
// CHECK-NEXT:     _t0 = x;
// CHECK-NEXT:     _t1 = _d_x0;
// CHECK-NEXT:     ValueAndPushforward<float, float> _t00 = clad::custom_derivatives{{(::std)?}}::log_pushforward(_t0, _t1);
// CHECK-NEXT:     goto _label0;
// CHECK-NEXT:   _label0:
// CHECK-NEXT:     _d__t0.pushforward += 1;
// CHECK-NEXT:     {
// CHECK-NEXT:         float _grad0 = 0.F;
// CHECK-NEXT:         float _grad1 = 0.F;
// CHECK-NEXT:         log_pushforward_pullback(_t0, _t1, _d__t0, &_grad0, &_grad1);
// CHECK-NEXT:         float _r0 = _grad0;
// CHECK-NEXT:         * _d_x += _r0;
// CHECK-NEXT:         float _r1 = _grad1;
// CHECK-NEXT:         _d__d_x += _r1;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: void f3_hessian(float x, clad::array_ref<float> hessianMatrix) {
// CHECK-NEXT:     f3_darg0_grad(x, hessianMatrix.slice(0UL, 1UL));
// CHECK-NEXT: }


float f4(float x) {
  return pow(x, 4.0F);
}

// CHECK: float f4_darg0(float x) {
// CHECK-NEXT:     float _d_x = 1;
// CHECK-NEXT:     ValueAndPushforward<decltype(::std::pow(float(), float())), decltype(::std::pow(float(), float()))> _t0 = clad::custom_derivatives{{(::std)?}}::pow_pushforward(x, 4.F, _d_x, 0.F);
// CHECK-NEXT:     return _t0.pushforward;
// CHECK-NEXT: }

// CHECK: void pow_pushforward_pullback(float x, float exponent, float d_x, float d_exponent, ValueAndPushforward<decltype(::std::pow(float(), float())), decltype(::std::pow(float(), float()))> _d_y, clad::array_ref<float> _d_x, clad::array_ref<float> _d_exponent, clad::array_ref<float> _d_d_x, clad::array_ref<float> _d_d_exponent) {
// CHECK-NEXT:     float _t0;
// CHECK-NEXT:     float _t1;
// CHECK-NEXT:     float _d_val = 0;
// CHECK-NEXT:     float _t2;
// CHECK-NEXT:     float _t3;
// CHECK-NEXT:     float _t4;
// CHECK-NEXT:     float _t5;
// CHECK-NEXT:     float _t6;
// CHECK-NEXT:     float _t7;
// CHECK-NEXT:     float _d_derivative = 0;
// CHECK-NEXT:     float _cond0;
// CHECK-NEXT:     float _t8;
// CHECK-NEXT:     float _t9;
// CHECK-NEXT:     float _t10;
// CHECK-NEXT:     float _t11;
// CHECK-NEXT:     float _t12;
// CHECK-NEXT:     float _t13;
// CHECK-NEXT:     float _t14;
// CHECK-NEXT:     _t0 = x;
// CHECK-NEXT:     _t1 = exponent;
// CHECK-NEXT:     float val = ::std::pow(_t0, _t1);
// CHECK-NEXT:     _t4 = exponent;
// CHECK-NEXT:     _t5 = x;
// CHECK-NEXT:     _t6 = exponent - 1;
// CHECK-NEXT:     _t3 = ::std::pow(_t5, _t6);
// CHECK-NEXT:     _t7 = (_t4 * _t3);
// CHECK-NEXT:     _t2 = d_x;
// CHECK-NEXT:     float derivative = _t7 * _t2;
// CHECK-NEXT:     _cond0 = d_exponent;
// CHECK-NEXT:     if (_cond0) {
// CHECK-NEXT:         _t10 = x;
// CHECK-NEXT:         _t11 = exponent;
// CHECK-NEXT:         _t12 = ::std::pow(_t10, _t11);
// CHECK-NEXT:         _t13 = x;
// CHECK-NEXT:         _t9 = ::std::log(_t13);
// CHECK-NEXT:         _t14 = (_t12 * _t9);
// CHECK-NEXT:         _t8 = d_exponent;
// CHECK-NEXT:         derivative += _t14 * _t8;
// CHECK-NEXT:     }
// CHECK-NEXT:     goto _label0;
// CHECK-NEXT:   _label0:
// CHECK-NEXT:     {
// CHECK-NEXT:         _d_val += _d_y.value;
// CHECK-NEXT:         _d_derivative += _d_y.pushforward;
// CHECK-NEXT:     }
// CHECK-NEXT:     if (_cond0) {
// CHECK-NEXT:         float _r_d0 = _d_derivative;
// CHECK-NEXT:         _d_derivative += _r_d0;
// CHECK-NEXT:         float _r8 = _r_d0 * _t8;
// CHECK-NEXT:         float _r9 = _r8 * _t9;
// CHECK-NEXT:         float _grad4 = 0.F;
// CHECK-NEXT:         float _grad5 = 0.F;
// CHECK-NEXT:         clad::custom_derivatives{{(::std)?}}::pow_pullback(_t10, _t11, _r9, &_grad4, &_grad5);
// CHECK-NEXT:         float _r10 = _grad4;
// CHECK-NEXT:         * _d_x += _r10;
// CHECK-NEXT:         float _r11 = _grad5;
// CHECK-NEXT:         * _d_exponent += _r11;
// CHECK-NEXT:         float _r12 = _t12 * _r8;
// CHECK-NEXT:         float _r13 = _r12 * clad::custom_derivatives{{(::std)?}}::log_pushforward(_t13, 1.F).pushforward;
// CHECK-NEXT:         * _d_x += _r13;
// CHECK-NEXT:         float _r14 = _t14 * _r_d0;
// CHECK-NEXT:         * _d_d_exponent += _r14;
// CHECK-NEXT:         _d_derivative -= _r_d0;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         float _r2 = _d_derivative * _t2;
// CHECK-NEXT:         float _r3 = _r2 * _t3;
// CHECK-NEXT:         * _d_exponent += _r3;
// CHECK-NEXT:         float _r4 = _t4 * _r2;
// CHECK-NEXT:         float _grad2 = 0.F;
// CHECK-NEXT:         float _grad3 = 0.F;
// CHECK-NEXT:         clad::custom_derivatives{{(::std)?}}::pow_pullback(_t5, _t6, _r4, &_grad2, &_grad3);
// CHECK-NEXT:         float _r5 = _grad2;
// CHECK-NEXT:         * _d_x += _r5;
// CHECK-NEXT:         float _r6 = _grad3;
// CHECK-NEXT:         * _d_exponent += _r6;
// CHECK-NEXT:         float _r7 = _t7 * _d_derivative;
// CHECK-NEXT:         * _d_d_x += _r7;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         float _grad0 = 0.F;
// CHECK-NEXT:         float _grad1 = 0.F;
// CHECK-NEXT:         clad::custom_derivatives{{(::std)?}}::pow_pullback(_t0, _t1, _d_val, &_grad0, &_grad1);
// CHECK-NEXT:         float _r0 = _grad0;
// CHECK-NEXT:         * _d_x += _r0;
// CHECK-NEXT:         float _r1 = _grad1;
// CHECK-NEXT:         * _d_exponent += _r1;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: void f4_darg0_grad(float x, clad::array_ref<float> _d_x) {
// CHECK-NEXT:     float _d__d_x = 0;
// CHECK-NEXT:     float _t0;
// CHECK-NEXT:     float _t1;
// CHECK-NEXT:     ValueAndPushforward<decltype(::std::pow(float(), float())), decltype(::std::pow(float(), float()))> _d__t0 = {};
// CHECK-NEXT:     float _d_x0 = 1;
// CHECK-NEXT:     _t0 = x;
// CHECK-NEXT:     _t1 = _d_x0;
// CHECK-NEXT:     ValueAndPushforward<decltype(::std::pow(float(), float())), decltype(::std::pow(float(), float()))> _t00 = clad::custom_derivatives{{(::std)?}}::pow_pushforward(_t0, 4.F, _t1, 0.F);
// CHECK-NEXT:     goto _label0;
// CHECK-NEXT:   _label0:
// CHECK-NEXT:     _d__t0.pushforward += 1;
// CHECK-NEXT:     {
// CHECK-NEXT:         float _grad0 = 0.F;
// CHECK-NEXT:         float _grad1 = 0.F;
// CHECK-NEXT:         float _grad2 = 0.F;
// CHECK-NEXT:         float _grad3 = 0.F;
// CHECK-NEXT:         pow_pushforward_pullback(_t0, 4.F, _t1, 0.F, _d__t0, &_grad0, &_grad1, &_grad2, &_grad3);
// CHECK-NEXT:         float _r0 = _grad0;
// CHECK-NEXT:         * _d_x += _r0;
// CHECK-NEXT:         float _r1 = _grad1;
// CHECK-NEXT:         float _r2 = _grad2;
// CHECK-NEXT:         _d__d_x += _r2;
// CHECK-NEXT:         float _r3 = _grad3;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: void f4_hessian(float x, clad::array_ref<float> hessianMatrix) {
// CHECK-NEXT:     f4_darg0_grad(x, hessianMatrix.slice(0UL, 1UL));
// CHECK-NEXT: }


float f5(float x) {
  return pow(2.0F, x);
}

// CHECK: float f5_darg0(float x) {
// CHECK-NEXT:     float _d_x = 1;
// CHECK-NEXT:     ValueAndPushforward<decltype(::std::pow(float(), float())), decltype(::std::pow(float(), float()))> _t0 = clad::custom_derivatives{{(::std)?}}::pow_pushforward(2.F, x, 0.F, _d_x);
// CHECK-NEXT:     return _t0.pushforward;
// CHECK-NEXT: }

// CHECK: void f5_darg0_grad(float x, clad::array_ref<float> _d_x) {
// CHECK-NEXT:     float _d__d_x = 0;
// CHECK-NEXT:     float _t0;
// CHECK-NEXT:     float _t1;
// CHECK-NEXT:     ValueAndPushforward<decltype(::std::pow(float(), float())), decltype(::std::pow(float(), float()))> _d__t0 = {};
// CHECK-NEXT:     float _d_x0 = 1;
// CHECK-NEXT:     _t0 = x;
// CHECK-NEXT:     _t1 = _d_x0;
// CHECK-NEXT:     ValueAndPushforward<decltype(::std::pow(float(), float())), decltype(::std::pow(float(), float()))> _t00 = clad::custom_derivatives{{(::std)?}}::pow_pushforward(2.F, _t0, 0.F, _t1);
// CHECK-NEXT:     goto _label0;
// CHECK-NEXT:   _label0:
// CHECK-NEXT:     _d__t0.pushforward += 1;
// CHECK-NEXT:     {
// CHECK-NEXT:         float _grad0 = 0.F;
// CHECK-NEXT:         float _grad1 = 0.F;
// CHECK-NEXT:         float _grad2 = 0.F;
// CHECK-NEXT:         float _grad3 = 0.F;
// CHECK-NEXT:         pow_pushforward_pullback(2.F, _t0, 0.F, _t1, _d__t0, &_grad0, &_grad1, &_grad2, &_grad3);
// CHECK-NEXT:         float _r0 = _grad0;
// CHECK-NEXT:         float _r1 = _grad1;
// CHECK-NEXT:         * _d_x += _r1;
// CHECK-NEXT:         float _r2 = _grad2;
// CHECK-NEXT:         float _r3 = _grad3;
// CHECK-NEXT:         _d__d_x += _r3;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: void f5_hessian(float x, clad::array_ref<float> hessianMatrix) {
// CHECK-NEXT:     f5_darg0_grad(x, hessianMatrix.slice(0UL, 1UL));
// CHECK-NEXT: }


float f6(float x, float y) {
  return pow(x, y);
}

// CHECK: float f6_darg0(float x, float y) {
// CHECK-NEXT:     float _d_x = 1;
// CHECK-NEXT:     float _d_y = 0;
// CHECK-NEXT:     ValueAndPushforward<decltype(::std::pow(float(), float())), decltype(::std::pow(float(), float()))> _t0 = clad::custom_derivatives{{(::std)?}}::pow_pushforward(x, y, _d_x, _d_y);
// CHECK-NEXT:     return _t0.pushforward;
// CHECK-NEXT: }

// CHECK: void f6_darg0_grad(float x, float y, clad::array_ref<float> _d_x, clad::array_ref<float> _d_y) {
// CHECK-NEXT:     float _d__d_x = 0;
// CHECK-NEXT:     float _d__d_y = 0;
// CHECK-NEXT:     float _t0;
// CHECK-NEXT:     float _t1;
// CHECK-NEXT:     float _t2;
// CHECK-NEXT:     float _t3;
// CHECK-NEXT:     ValueAndPushforward<decltype(::std::pow(float(), float())), decltype(::std::pow(float(), float()))> _d__t0 = {};
// CHECK-NEXT:     float _d_x0 = 1;
// CHECK-NEXT:     float _d_y0 = 0;
// CHECK-NEXT:     _t0 = x;
// CHECK-NEXT:     _t1 = y;
// CHECK-NEXT:     _t2 = _d_x0;
// CHECK-NEXT:     _t3 = _d_y0;
// CHECK-NEXT:     ValueAndPushforward<decltype(::std::pow(float(), float())), decltype(::std::pow(float(), float()))> _t00 = clad::custom_derivatives{{(::std)?}}::pow_pushforward(_t0, _t1, _t2, _t3);
// CHECK-NEXT:     goto _label0;
// CHECK-NEXT:   _label0:
// CHECK-NEXT:     _d__t0.pushforward += 1;
// CHECK-NEXT:     {
// CHECK-NEXT:         float _grad0 = 0.F;
// CHECK-NEXT:         float _grad1 = 0.F;
// CHECK-NEXT:         float _grad2 = 0.F;
// CHECK-NEXT:         float _grad3 = 0.F;
// CHECK-NEXT:         pow_pushforward_pullback(_t0, _t1, _t2, _t3, _d__t0, &_grad0, &_grad1, &_grad2, &_grad3);
// CHECK-NEXT:         float _r0 = _grad0;
// CHECK-NEXT:         * _d_x += _r0;
// CHECK-NEXT:         float _r1 = _grad1;
// CHECK-NEXT:         * _d_y += _r1;
// CHECK-NEXT:         float _r2 = _grad2;
// CHECK-NEXT:         _d__d_x += _r2;
// CHECK-NEXT:         float _r3 = _grad3;
// CHECK-NEXT:         _d__d_y += _r3;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: float f6_darg1(float x, float y) {
// CHECK-NEXT:     float _d_x = 0;
// CHECK-NEXT:     float _d_y = 1;
// CHECK-NEXT:     ValueAndPushforward<decltype(::std::pow(float(), float())), decltype(::std::pow(float(), float()))> _t0 = clad::custom_derivatives{{(::std)?}}::pow_pushforward(x, y, _d_x, _d_y);
// CHECK-NEXT:     return _t0.pushforward;
// CHECK-NEXT: }

// CHECK: void f6_darg1_grad(float x, float y, clad::array_ref<float> _d_x, clad::array_ref<float> _d_y) {
// CHECK-NEXT:     float _d__d_x = 0;
// CHECK-NEXT:     float _d__d_y = 0;
// CHECK-NEXT:     float _t0;
// CHECK-NEXT:     float _t1;
// CHECK-NEXT:     float _t2;
// CHECK-NEXT:     float _t3;
// CHECK-NEXT:     ValueAndPushforward<decltype(::std::pow(float(), float())), decltype(::std::pow(float(), float()))> _d__t0 = {};
// CHECK-NEXT:     float _d_x0 = 0;
// CHECK-NEXT:     float _d_y0 = 1;
// CHECK-NEXT:     _t0 = x;
// CHECK-NEXT:     _t1 = y;
// CHECK-NEXT:     _t2 = _d_x0;
// CHECK-NEXT:     _t3 = _d_y0;
// CHECK-NEXT:     ValueAndPushforward<decltype(::std::pow(float(), float())), decltype(::std::pow(float(), float()))> _t00 = clad::custom_derivatives{{(::std)?}}::pow_pushforward(_t0, _t1, _t2, _t3);
// CHECK-NEXT:     goto _label0;
// CHECK-NEXT:   _label0:
// CHECK-NEXT:     _d__t0.pushforward += 1;
// CHECK-NEXT:     {
// CHECK-NEXT:         float _grad0 = 0.F;
// CHECK-NEXT:         float _grad1 = 0.F;
// CHECK-NEXT:         float _grad2 = 0.F;
// CHECK-NEXT:         float _grad3 = 0.F;
// CHECK-NEXT:         pow_pushforward_pullback(_t0, _t1, _t2, _t3, _d__t0, &_grad0, &_grad1, &_grad2, &_grad3);
// CHECK-NEXT:         float _r0 = _grad0;
// CHECK-NEXT:         * _d_x += _r0;
// CHECK-NEXT:         float _r1 = _grad1;
// CHECK-NEXT:         * _d_y += _r1;
// CHECK-NEXT:         float _r2 = _grad2;
// CHECK-NEXT:         _d__d_x += _r2;
// CHECK-NEXT:         float _r3 = _grad3;
// CHECK-NEXT:         _d__d_y += _r3;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: void f6_hessian(float x, float y, clad::array_ref<float> hessianMatrix) {
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
  TEST1(f5, 3); // CHECK-EXEC: Result is = {3.84}
  TEST2(f6, 3, 4); // CHECK-EXEC: Result is = {108.00, 145.65, 145.65, 97.76}
}
