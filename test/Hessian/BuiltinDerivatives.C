// RUN: %cladclang %s -I%S/../../include -oHessianBuiltinDerivatives.out 2>&1 | %filecheck %s
// RUN: ./HessianBuiltinDerivatives.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -enable-tbr %s -I%S/../../include -oHessianBuiltinDerivatives.out
// RUN: ./HessianBuiltinDerivatives.out | %filecheck_exec %s


//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"
#include <math.h>

float f1(float x) {
  return sin(x) + cos(x);
}

// CHECK: float f1_darg0(float x);

// CHECK: void f1_darg0_grad(float x, float *_d_x);

// CHECK: void f1_hessian(float x, float *hessianMatrix) {
// CHECK-NEXT:     f1_darg0_grad(x, hessianMatrix + {{0U|0UL}});
// CHECK-NEXT: }

float f2(float x) {
  return exp(x);
}

// CHECK: float f2_darg0(float x);

// CHECK: void f2_darg0_grad(float x, float *_d_x);

// CHECK: void f2_hessian(float x, float *hessianMatrix) {
// CHECK-NEXT:     f2_darg0_grad(x, hessianMatrix + {{0U|0UL}});
// CHECK-NEXT: }


float f3(float x) {
  return log(x);
}

// CHECK: float f3_darg0(float x);

// CHECK: void f3_darg0_grad(float x, float *_d_x);

// CHECK: void f3_hessian(float x, float *hessianMatrix) {
// CHECK-NEXT:     f3_darg0_grad(x, hessianMatrix + {{0U|0UL}});
// CHECK-NEXT: }


float f4(float x) {
  return pow(x, 4.0F);
}

// CHECK: float f4_darg0(float x);

// CHECK: void f4_darg0_grad(float x, float *_d_x);

// CHECK: void f4_hessian(float x, float *hessianMatrix) {
// CHECK-NEXT:     f4_darg0_grad(x, hessianMatrix + {{0U|0UL}});
// CHECK-NEXT: }


float f5(float x) {
  return pow(2.0F, x);
}

// CHECK: float f5_darg0(float x);

// CHECK: void f5_darg0_grad(float x, float *_d_x);

// CHECK: void f5_hessian(float x, float *hessianMatrix) {
// CHECK-NEXT:     f5_darg0_grad(x, hessianMatrix + {{0U|0UL}});
// CHECK-NEXT: }


float f6(float x, float y) {
  return pow(x, y);
}

// CHECK: float f6_darg0(float x, float y);

// CHECK: void f6_darg0_grad(float x, float y, float *_d_x, float *_d_y);

// CHECK: float f6_darg1(float x, float y);

// CHECK: void f6_darg1_grad(float x, float y, float *_d_x, float *_d_y);

// CHECK: void f6_hessian(float x, float y, float *hessianMatrix) {
// CHECK-NEXT:     f6_darg0_grad(x, y, hessianMatrix + {{0U|0UL}}, hessianMatrix + {{1U|1UL}});
// CHECK-NEXT:     f6_darg1_grad(x, y, hessianMatrix + {{2U|2UL}}, hessianMatrix + {{3U|3UL}});
// CHECK-NEXT: }

namespace clad {
  namespace custom_derivatives {
    float f7_darg0(float x, float y) {
      return cos(x);
    }

    float f7_darg1(float x, float y) {
      return exp(y);
    }

    void f7_darg1_grad(float x, float y, float *d_x, float *d_y) {
      *d_y += exp(y);
    }

    void f8_hessian(float x, float y, float *hessianMatrix) {
      hessianMatrix[0] = 1.0;
      hessianMatrix[1] = 1.0;
      hessianMatrix[2] = 1.0;
      hessianMatrix[3] = 1.0;
    }
  }
}

float f7(float x, float y) {
  return sin(x) + exp(y);
}

// CHECK: float f7_darg0(float x, float y) {
// CHECK-NEXT:     return cos(x);
// CHECK-NEXT: }

// CHECK: void f7_darg0_grad(float x, float y, float *_d_x, float *_d_y);

// CHECK: float f7_darg1(float x, float y) {
// CHECK-NEXT:     return exp(y);
// CHECK-NEXT: }

// CHECK: void f7_darg1_grad(float x, float y, float *d_x, float *d_y) {
// CHECK-NEXT:     *d_y += exp(y);
// CHECK-NEXT: }

// CHECK: void f7_hessian(float x, float y, float *hessianMatrix) {
// CHECK-NEXT:     f7_darg0_grad(x, y, hessianMatrix + {{0U|0UL}}, hessianMatrix + {{1U|1UL}});
// CHECK-NEXT:     f7_darg1_grad(x, y, hessianMatrix + {{2U|2UL}}, hessianMatrix + {{3U|3UL}});
// CHECK-NEXT: }

float f8(float x, float y) {
  return (x*x + y*y)/2 + x*y;
}

// CHECK: void f8_hessian(float x, float y, float *hessianMatrix) {
// CHECK-NEXT:     hessianMatrix[0] = 1.;
// CHECK-NEXT:     hessianMatrix[1] = 1.;
// CHECK-NEXT:     hessianMatrix[2] = 1.;
// CHECK-NEXT:     hessianMatrix[3] = 1.;
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
  h.execute(x, y, result);                           \
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
  TEST2(f7, 3, 4); // CHECK-EXEC: Result is = {-0.14, 0.00, 0.00, 54.60}
  TEST2(f8, 3, 4); // CHECK-EXEC: Result is = {1.00, 1.00, 1.00, 1.00}

// CHECK: float f1_darg0(float x) {
// CHECK-NEXT:     float _d_x = 1;
// CHECK-NEXT:     ValueAndPushforward<float, float> _t0 = clad::custom_derivatives{{(::std)?}}::sin_pushforward(x, _d_x);
// CHECK-NEXT:     ValueAndPushforward<float, float> _t1 = clad::custom_derivatives{{(::std)?}}::cos_pushforward(x, _d_x);
// CHECK-NEXT:     return _t0.pushforward + _t1.pushforward;
// CHECK-NEXT: }

// CHECK: void sin_pushforward_pullback(float x, float d_x, ValueAndPushforward<float, float> _d_y, float *_d_x, float *_d_d_x);

// CHECK: void cos_pushforward_pullback(float x, float d_x, ValueAndPushforward<float, float> _d_y, float *_d_x, float *_d_d_x);

// CHECK: void f1_darg0_grad(float x, float *_d_x) {
// CHECK-NEXT:     float _d__d_x = 0;
// CHECK-NEXT:     float _d_x0 = 1;
// CHECK-NEXT:     ValueAndPushforward<float, float> _d__t0 = {};
// CHECK-NEXT:     ValueAndPushforward<float, float> _t00 = clad::custom_derivatives{{(::std)?}}::sin_pushforward(x, _d_x0);
// CHECK-NEXT:     ValueAndPushforward<float, float> _d__t1 = {};
// CHECK-NEXT:     ValueAndPushforward<float, float> _t10 = clad::custom_derivatives{{(::std)?}}::cos_pushforward(x, _d_x0);
// CHECK-NEXT:     {
// CHECK-NEXT:         _d__t0.pushforward += 1;
// CHECK-NEXT:         _d__t1.pushforward += 1;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         float _r2 = 0;
// CHECK-NEXT:         float _r3 = 0;
// CHECK-NEXT:         cos_pushforward_pullback(x, _d_x0, _d__t1, &_r2, &_r3);
// CHECK-NEXT:         *_d_x += _r2;
// CHECK-NEXT:         _d__d_x += _r3;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         float _r0 = 0;
// CHECK-NEXT:         float _r1 = 0;
// CHECK-NEXT:         sin_pushforward_pullback(x, _d_x0, _d__t0, &_r0, &_r1);
// CHECK-NEXT:         *_d_x += _r0;
// CHECK-NEXT:         _d__d_x += _r1;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: float f2_darg0(float x) {
// CHECK-NEXT:     float _d_x = 1;
// CHECK-NEXT:     ValueAndPushforward<float, float> _t0 = clad::custom_derivatives{{(::std)?}}::exp_pushforward(x, _d_x);
// CHECK-NEXT:     return _t0.pushforward;
// CHECK-NEXT: }

// CHECK: void exp_pushforward_pullback(float x, float d_x, ValueAndPushforward<float, float> _d_y, float *_d_x, float *_d_d_x);

// CHECK: void f2_darg0_grad(float x, float *_d_x) {
// CHECK-NEXT:     float _d__d_x = 0;
// CHECK-NEXT:     float _d_x0 = 1;
// CHECK-NEXT:     ValueAndPushforward<float, float> _d__t0 = {};
// CHECK-NEXT:     ValueAndPushforward<float, float> _t00 = clad::custom_derivatives{{(::std)?}}::exp_pushforward(x, _d_x0);
// CHECK-NEXT:     _d__t0.pushforward += 1;
// CHECK-NEXT:     {
// CHECK-NEXT:         float _r0 = 0;
// CHECK-NEXT:         float _r1 = 0;
// CHECK-NEXT:         exp_pushforward_pullback(x, _d_x0, _d__t0, &_r0, &_r1);
// CHECK-NEXT:         *_d_x += _r0;
// CHECK-NEXT:         _d__d_x += _r1;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: float f3_darg0(float x) {
// CHECK-NEXT:     float _d_x = 1;
// CHECK-NEXT:     ValueAndPushforward<float, float> _t0 = clad::custom_derivatives{{(::std)?}}::log_pushforward(x, _d_x);
// CHECK-NEXT:     return _t0.pushforward;
// CHECK-NEXT: }

// CHECK: void log_pushforward_pullback(float x, float d_x, ValueAndPushforward<float, float> _d_y, float *_d_x, float *_d_d_x);

// CHECK: void f3_darg0_grad(float x, float *_d_x) {
// CHECK-NEXT:     float _d__d_x = 0;
// CHECK-NEXT:     float _d_x0 = 1;
// CHECK-NEXT:     ValueAndPushforward<float, float> _d__t0 = {};
// CHECK-NEXT:     ValueAndPushforward<float, float> _t00 = clad::custom_derivatives{{(::std)?}}::log_pushforward(x, _d_x0);
// CHECK-NEXT:     _d__t0.pushforward += 1;
// CHECK-NEXT:     {
// CHECK-NEXT:         float _r0 = 0;
// CHECK-NEXT:         float _r1 = 0;
// CHECK-NEXT:         log_pushforward_pullback(x, _d_x0, _d__t0, &_r0, &_r1);
// CHECK-NEXT:         *_d_x += _r0;
// CHECK-NEXT:         _d__d_x += _r1;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: float f4_darg0(float x) {
// CHECK-NEXT:     float _d_x = 1;
// CHECK-NEXT:     ValueAndPushforward<decltype(::std::pow(float(), float())), decltype(::std::pow(float(), float()))> _t0 = clad::custom_derivatives{{(::std)?}}::pow_pushforward(x, 4.F, _d_x, 0.F);
// CHECK-NEXT:     return _t0.pushforward;
// CHECK-NEXT: }

// CHECK: void pow_pushforward_pullback(float x, float exponent, float d_x, float d_exponent, ValueAndPushforward<decltype(::std::pow(float(), float())), decltype(::std::pow(float(), float()))> _d_y, float *_d_x, float *_d_exponent, float *_d_d_x, float *_d_d_exponent);

// CHECK: void f4_darg0_grad(float x, float *_d_x) {
// CHECK-NEXT:     float _d__d_x = 0;
// CHECK-NEXT:     float _d_x0 = 1;
// CHECK-NEXT:     ValueAndPushforward<decltype(::std::pow(float(), float())), decltype(::std::pow(float(), float()))> _d__t0 = {};
// CHECK-NEXT:     ValueAndPushforward<decltype(::std::pow(float(), float())), decltype(::std::pow(float(), float()))> _t00 = clad::custom_derivatives{{(::std)?}}::pow_pushforward(x, 4.F, _d_x0, 0.F);
// CHECK-NEXT:     _d__t0.pushforward += 1;
// CHECK-NEXT:     {
// CHECK-NEXT:         float _r0 = 0;
// CHECK-NEXT:         float _r1 = 0;
// CHECK-NEXT:         float _r2 = 0;
// CHECK-NEXT:         float _r3 = 0;
// CHECK-NEXT:         pow_pushforward_pullback(x, 4.F, _d_x0, 0.F, _d__t0, &_r0, &_r1, &_r2, &_r3);
// CHECK-NEXT:         *_d_x += _r0;
// CHECK-NEXT:         _d__d_x += _r2;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: float f5_darg0(float x) {
// CHECK-NEXT:     float _d_x = 1;
// CHECK-NEXT:     ValueAndPushforward<decltype(::std::pow(float(), float())), decltype(::std::pow(float(), float()))> _t0 = clad::custom_derivatives{{(::std)?}}::pow_pushforward(2.F, x, 0.F, _d_x);
// CHECK-NEXT:     return _t0.pushforward;
// CHECK-NEXT: }

// CHECK: void f5_darg0_grad(float x, float *_d_x) {
// CHECK-NEXT:     float _d__d_x = 0;
// CHECK-NEXT:     float _d_x0 = 1;
// CHECK-NEXT:     ValueAndPushforward<decltype(::std::pow(float(), float())), decltype(::std::pow(float(), float()))> _d__t0 = {};
// CHECK-NEXT:     ValueAndPushforward<decltype(::std::pow(float(), float())), decltype(::std::pow(float(), float()))> _t00 = clad::custom_derivatives{{(::std)?}}::pow_pushforward(2.F, x, 0.F, _d_x0);
// CHECK-NEXT:     _d__t0.pushforward += 1;
// CHECK-NEXT:     {
// CHECK-NEXT:         float _r0 = 0;
// CHECK-NEXT:         float _r1 = 0;
// CHECK-NEXT:         float _r2 = 0;
// CHECK-NEXT:         float _r3 = 0;
// CHECK-NEXT:         pow_pushforward_pullback(2.F, x, 0.F, _d_x0, _d__t0, &_r0, &_r1, &_r2, &_r3);
// CHECK-NEXT:         *_d_x += _r1;
// CHECK-NEXT:         _d__d_x += _r3;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: float f6_darg0(float x, float y) {
// CHECK-NEXT:     float _d_x = 1;
// CHECK-NEXT:     float _d_y = 0;
// CHECK-NEXT:     ValueAndPushforward<decltype(::std::pow(float(), float())), decltype(::std::pow(float(), float()))> _t0 = clad::custom_derivatives{{(::std)?}}::pow_pushforward(x, y, _d_x, _d_y);
// CHECK-NEXT:     return _t0.pushforward;
// CHECK-NEXT: }

// CHECK: void f6_darg0_grad(float x, float y, float *_d_x, float *_d_y) {
// CHECK-NEXT:     float _d__d_x = 0;
// CHECK-NEXT:     float _d_x0 = 1;
// CHECK-NEXT:     float _d__d_y = 0;
// CHECK-NEXT:     float _d_y0 = 0;
// CHECK-NEXT:     ValueAndPushforward<decltype(::std::pow(float(), float())), decltype(::std::pow(float(), float()))> _d__t0 = {};
// CHECK-NEXT:     ValueAndPushforward<decltype(::std::pow(float(), float())), decltype(::std::pow(float(), float()))> _t00 = clad::custom_derivatives{{(::std)?}}::pow_pushforward(x, y, _d_x0, _d_y0);
// CHECK-NEXT:     _d__t0.pushforward += 1;
// CHECK-NEXT:     {
// CHECK-NEXT:         float _r0 = 0;
// CHECK-NEXT:         float _r1 = 0;
// CHECK-NEXT:         float _r2 = 0;
// CHECK-NEXT:         float _r3 = 0;
// CHECK-NEXT:         pow_pushforward_pullback(x, y, _d_x0, _d_y0, _d__t0, &_r0, &_r1, &_r2, &_r3);
// CHECK-NEXT:         *_d_x += _r0;
// CHECK-NEXT:         *_d_y += _r1;
// CHECK-NEXT:         _d__d_x += _r2;
// CHECK-NEXT:         _d__d_y += _r3;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: float f6_darg1(float x, float y) {
// CHECK-NEXT:     float _d_x = 0;
// CHECK-NEXT:     float _d_y = 1;
// CHECK-NEXT:     ValueAndPushforward<decltype(::std::pow(float(), float())), decltype(::std::pow(float(), float()))> _t0 = clad::custom_derivatives{{(::std)?}}::pow_pushforward(x, y, _d_x, _d_y);
// CHECK-NEXT:     return _t0.pushforward;
// CHECK-NEXT: }

// CHECK: void f6_darg1_grad(float x, float y, float *_d_x, float *_d_y) {
// CHECK-NEXT:     float _d__d_x = 0;
// CHECK-NEXT:     float _d_x0 = 0;
// CHECK-NEXT:     float _d__d_y = 0;
// CHECK-NEXT:     float _d_y0 = 1;
// CHECK-NEXT:     ValueAndPushforward<decltype(::std::pow(float(), float())), decltype(::std::pow(float(), float()))> _d__t0 = {};
// CHECK-NEXT:     ValueAndPushforward<decltype(::std::pow(float(), float())), decltype(::std::pow(float(), float()))> _t00 = clad::custom_derivatives{{(::std)?}}::pow_pushforward(x, y, _d_x0, _d_y0);
// CHECK-NEXT:     _d__t0.pushforward += 1;
// CHECK-NEXT:     {
// CHECK-NEXT:         float _r0 = 0;
// CHECK-NEXT:         float _r1 = 0;
// CHECK-NEXT:         float _r2 = 0;
// CHECK-NEXT:         float _r3 = 0;
// CHECK-NEXT:         pow_pushforward_pullback(x, y, _d_x0, _d_y0, _d__t0, &_r0, &_r1, &_r2, &_r3);
// CHECK-NEXT:         *_d_x += _r0;
// CHECK-NEXT:         *_d_y += _r1;
// CHECK-NEXT:         _d__d_x += _r2;
// CHECK-NEXT:         _d__d_y += _r3;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: void f7_darg0_grad(float x, float y, float *_d_x, float *_d_y) {
// CHECK-NEXT:     {
// CHECK-NEXT:         float _r0 = 0;
// CHECK-NEXT:         _r0 += 1 * clad::custom_derivatives{{(::std)?}}::cos_pushforward(x, 1.F).pushforward; 
// CHECK-NEXT:         *_d_x += _r0;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: void sin_pushforward_pullback(float x, float d_x, ValueAndPushforward<float, float> _d_y, float *_d_x, float *_d_d_x) {
// CHECK-NEXT:     float _t0 = ::std::cos(x);
// CHECK-NEXT:     {
// CHECK-NEXT:         float _r0 = 0;
// CHECK-NEXT:         _r0 += _d_y.value * clad::custom_derivatives{{(::std)?}}::sin_pushforward(x, 1.F).pushforward;
// CHECK-NEXT:         *_d_x += _r0;
// CHECK-NEXT:         float _r1 = 0;
// CHECK-NEXT:         _r1 += _d_y.pushforward * d_x * clad::custom_derivatives{{(::std)?}}::cos_pushforward(x, 1.F).pushforward;
// CHECK-NEXT:         *_d_x += _r1;
// CHECK-NEXT:         *_d_d_x += _t0 * _d_y.pushforward;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: void cos_pushforward_pullback(float x, float d_x, ValueAndPushforward<float, float> _d_y, float *_d_x, float *_d_d_x) {
// CHECK-NEXT:     float _t0 = ::std::sin(x);
// CHECK-NEXT:     {
// CHECK-NEXT:         float _r0 = 0;
// CHECK-NEXT:         _r0 += _d_y.value * clad::custom_derivatives{{(::std)?}}::cos_pushforward(x, 1.F).pushforward;
// CHECK-NEXT:         *_d_x += _r0;
// CHECK-NEXT:         float _r1 = 0;
// CHECK-NEXT:         _r1 += -1 * _d_y.pushforward * d_x * clad::custom_derivatives{{(::std)?}}::sin_pushforward(x, 1.F).pushforward;
// CHECK-NEXT:         *_d_x += _r1;
// CHECK-NEXT:         *_d_d_x += -1 * _t0 * _d_y.pushforward;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: void exp_pushforward_pullback(float x, float d_x, ValueAndPushforward<float, float> _d_y, float *_d_x, float *_d_d_x) {
// CHECK-NEXT:     float _t0 = ::std::exp(x);
// CHECK-NEXT:     {
// CHECK-NEXT:         float _r0 = 0;
// CHECK-NEXT:         _r0 += _d_y.value * clad::custom_derivatives{{(::std)?}}::exp_pushforward(x, 1.F).pushforward;
// CHECK-NEXT:         *_d_x += _r0;
// CHECK-NEXT:         float _r1 = 0;
// CHECK-NEXT:         _r1 += _d_y.pushforward * d_x * clad::custom_derivatives{{(::std)?}}::exp_pushforward(x, 1.F).pushforward;
// CHECK-NEXT:         *_d_x += _r1;
// CHECK-NEXT:         *_d_d_x += _t0 * _d_y.pushforward;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: void log_pushforward_pullback(float x, float d_x, ValueAndPushforward<float, float> _d_y, float *_d_x, float *_d_d_x) {
// CHECK-NEXT:     {
// CHECK-NEXT:         float _r0 = 0;
// CHECK-NEXT:         _r0 += _d_y.value * clad::custom_derivatives{{(::std)?}}::log_pushforward(x, 1.F).pushforward;
// CHECK-NEXT:         *_d_x += _r0;
// CHECK-NEXT:         double _r1 = _d_y.pushforward * d_x * -(1. / (x * x));
// CHECK-NEXT:         *_d_x += _r1;
// CHECK-NEXT:         *_d_d_x += (1. / x) * _d_y.pushforward;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: void pow_pushforward_pullback(float x, float exponent, float d_x, float d_exponent, ValueAndPushforward<decltype(::std::pow(float(), float())), decltype(::std::pow(float(), float()))> _d_y, float *_d_x, float *_d_exponent, float *_d_d_x, float *_d_d_exponent) {
// CHECK-NEXT:     bool _cond0;
// CHECK-NEXT:     float _t1;
// CHECK-NEXT:     float _t2;
// CHECK-NEXT:     float _t3;
// CHECK-NEXT:     float _d_val = 0;
// CHECK-NEXT:     float val = ::std::pow(x, exponent);
// CHECK-NEXT:     float _t0 = ::std::pow(x, exponent - 1);
// CHECK-NEXT:     float _d_derivative = 0;
// CHECK-NEXT:     float derivative = (exponent * _t0) * d_x;
// CHECK-NEXT:     {
// CHECK-NEXT:     _cond0 = d_exponent;
// CHECK-NEXT:     if (_cond0) {
// CHECK-NEXT:         _t1 = derivative;
// CHECK-NEXT:         _t3 = ::std::pow(x, exponent);
// CHECK-NEXT:         _t2 = ::std::log(x);
// CHECK-NEXT:         derivative += (_t3 * _t2) * d_exponent;
// CHECK-NEXT:     }
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         _d_val += _d_y.value;
// CHECK-NEXT:         _d_derivative += _d_y.pushforward;
// CHECK-NEXT:     }
// CHECK-NEXT:     if (_cond0) {
// CHECK-NEXT:         derivative = _t1;
// CHECK-NEXT:         float _r_d0 = _d_derivative;
// CHECK-NEXT:         float _r4 = 0;
// CHECK-NEXT:         float _r5 = 0;
// CHECK-NEXT:         clad::custom_derivatives{{(::std)?}}::pow_pullback(x, exponent, _r_d0 * d_exponent * _t2, &_r4, &_r5);
// CHECK-NEXT:         *_d_x += _r4;
// CHECK-NEXT:         *_d_exponent += _r5;
// CHECK-NEXT:         float _r6 = 0;
// CHECK-NEXT:         _r6 += _t3 * _r_d0 * d_exponent * clad::custom_derivatives{{(::std)?}}::log_pushforward(x, 1.F).pushforward;
// CHECK-NEXT:         *_d_x += _r6;
// CHECK-NEXT:         *_d_d_exponent += (_t3 * _t2) * _r_d0;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         *_d_exponent += _d_derivative * d_x * _t0;
// CHECK-NEXT:         float _r2 = 0;
// CHECK-NEXT:         float _r3 = 0;
// CHECK-NEXT:         clad::custom_derivatives{{(::std)?}}::pow_pullback(x, exponent - 1, exponent * _d_derivative * d_x, &_r2, &_r3);
// CHECK-NEXT:         *_d_x += _r2;
// CHECK-NEXT:         *_d_exponent += _r3;
// CHECK-NEXT:         *_d_d_x += (exponent * _t0) * _d_derivative;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         float _r0 = 0;
// CHECK-NEXT:         float _r1 = 0;
// CHECK-NEXT:         clad::custom_derivatives{{(::std)?}}::pow_pullback(x, exponent, _d_val, &_r0, &_r1);
// CHECK-NEXT:         *_d_x += _r0;
// CHECK-NEXT:         *_d_exponent += _r1;
// CHECK-NEXT:     }
// CHECK-NEXT: }
}
