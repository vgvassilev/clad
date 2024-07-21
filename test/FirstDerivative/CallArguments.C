// RUN: %cladclang %s -I%S/../../include -Xclang -verify -oCallArguments.out 2>&1 | %filecheck %s
// RUN: ./CallArguments.out | %filecheck_exec %s

//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

float f(float y) {
  return y * y - 10;
}

namespace clad {
namespace custom_derivatives {
clad::ValueAndPushforward<float, float> f_pushforward(float y, float d_y) { return {f(y), 2 * y * d_y}; }
} // namespace custom_derivatives
} // namespace clad


float g(float x) {
  return f(x*x*x);
}

// CHECK: float g_darg0(float x) {
// CHECK-NEXT: float _d_x = 1;
// CHECK-NEXT: float _t0 = x * x;
// CHECK-NEXT: clad::ValueAndPushforward<float, float> _t1 = clad::custom_derivatives::f_pushforward(_t0 * x, (_d_x * x + x * _d_x) * x + _t0 * _d_x);
// CHECK-NEXT: return _t1.pushforward;
// CHECK-NEXT: }

float sqrt_func(float x, float y) {
  return sqrt(x * x + y * y) - y;
}

// CHECK: float sqrt_func_darg0(float x, float y) {
// CHECK-NEXT: float _d_x = 1;
// CHECK-NEXT: float _d_y = 0;
// CHECK-NEXT: ValueAndPushforward<float, float> _t0 = clad::custom_derivatives{{(::std)?}}::sqrt_pushforward(x * x + y * y, _d_x * x + x * _d_x + _d_y * y + y * _d_y);
// CHECK-NEXT: return _t0.pushforward - _d_y;
// CHECK-NEXT: }

float f_const_args_func_1(const float x, const float y) {
  return x * x + y * y - 1.F;
}

// CHECK: float f_const_args_func_1_darg0(const float x, const float y) {
// CHECK-NEXT: const float _d_x = 1;
// CHECK-NEXT: const float _d_y = 0;
// CHECK-NEXT: return _d_x * x + x * _d_x + _d_y * y + y * _d_y - 0.F;
// CHECK-NEXT: }

float f_const_args_func_2(float x, const float y) {
  return x * x + y * y - 1.F;
}

// CHECK: float f_const_args_func_2_darg0(float x, const float y) {
// CHECK-NEXT: float _d_x = 1;
// CHECK-NEXT: const float _d_y = 0;
// CHECK-NEXT: return _d_x * x + x * _d_x + _d_y * y + y * _d_y - 0.F;
// CHECK-NEXT: }

float f_const_args_func_3(const float x, float y) {
  return x * x + y * y - 1.F;
}

// CHECK: float f_const_args_func_3_darg0(const float x, float y) {
// CHECK-NEXT: const float _d_x = 1;
// CHECK-NEXT: float _d_y = 0;
// CHECK-NEXT: return _d_x * x + x * _d_x + _d_y * y + y * _d_y - 0.F;
// CHECK-NEXT: }

struct Vec { float x=0, y=0, z=0; };
float f_const_args_func_4(float x, float y, const Vec v) {
  return x * x + y * y - v.x;
}

// CHECK: float f_const_args_func_4_darg0(float x, float y, const Vec v) {
// CHECK-NEXT: float _d_x = 1;
// CHECK-NEXT: float _d_y = 0;
// CHECK-NEXT: const Vec _d_v;
// CHECK-NEXT: return _d_x * x + x * _d_x + _d_y * y + y * _d_y - _d_v.x;
// CHECK-NEXT: }

float f_const_args_func_5(float x, float y, const Vec &v) {
  return x * x + y * y - v.x;
}

// CHECK: float f_const_args_func_5_darg0(float x, float y, const Vec &v) {
// CHECK-NEXT: float _d_x = 1;
// CHECK-NEXT: float _d_y = 0;
// CHECK-NEXT: const Vec _d_v;
// CHECK-NEXT: return _d_x * x + x * _d_x + _d_y * y + y * _d_y - _d_v.x;
// CHECK-NEXT: }

float f_const_args_func_6(const float x, const float y, const Vec &v) {
  return x * x + y * y - v.x;
}

// CHECK: float f_const_args_func_6_darg0(const float x, const float y, const Vec &v) {
// CHECK-NEXT: const float _d_x = 1;
// CHECK-NEXT: const float _d_y = 0;
// CHECK-NEXT: const Vec _d_v;
// CHECK-NEXT: return _d_x * x + x * _d_x + _d_y * y + y * _d_y - _d_v.x;
// CHECK-NEXT: }

float f_const_helper(const float x) {
  return x * x;
}

// CHECK: clad::ValueAndPushforward<float, float> f_const_helper_pushforward(const float x, const float _d_x);

float f_const_args_func_7(const float x, const float y) {
  return f_const_helper(x) + f_const_helper(y) - y;
}

// CHECK: float f_const_args_func_7_darg0(const float x, const float y) {
// CHECK-NEXT: const float _d_x = 1;
// CHECK-NEXT: const float _d_y = 0;
// CHECK-NEXT: clad::ValueAndPushforward<float, float> _t0 = f_const_helper_pushforward(x, _d_x);
// CHECK-NEXT: clad::ValueAndPushforward<float, float> _t1 = f_const_helper_pushforward(y, _d_y);
// CHECK-NEXT: return _t0.pushforward + _t1.pushforward - _d_y;
// CHECK-NEXT: }

float f_const_args_func_8(const float x, float y) {
  return f_const_helper(x) + f_const_helper(y) - y;
}

// CHECK: float f_const_args_func_8_darg0(const float x, float y) {
// CHECK-NEXT: const float _d_x = 1;
// CHECK-NEXT: float _d_y = 0;
// CHECK-NEXT: clad::ValueAndPushforward<float, float> _t0 = f_const_helper_pushforward(x, _d_x);
// CHECK-NEXT: clad::ValueAndPushforward<float, float> _t1 = f_const_helper_pushforward(y, _d_y);
// CHECK-NEXT: return _t0.pushforward + _t1.pushforward - _d_y;
// CHECK-NEXT: }

float f_literal_helper(float x, char ch, float* p, float* q) {
  if (ch == 'a')
    return x * x;
  return -x * x;
}

float f_literal_args_func(float x, float y, float *z) {
  printf("hello world ");
  return x * f_literal_helper(0.5, 'a', z, nullptr);
}

// CHECK: float f_literal_args_func_darg0(float x, float y, float *z) {
// CHECK-NEXT: float _d_x = 1;
// CHECK-NEXT: float _d_y = 0;
// CHECK-NEXT: printf("hello world ");
// CHECK-NEXT: float _t0 = f_literal_helper(0.5, 'a', z, nullptr);
// CHECK-NEXT: return _d_x * _t0 + x * 0;
// CHECK-NEXT: }

inline unsigned int getBin(double low, double high, double val, unsigned int numBins) {
   double binWidth = (high - low) / numBins;
   return val >= high ? numBins - 1 : std::abs((val - low) / binWidth);
}

float f_call_inline_fxn(float *params, float const *obs, float const *xlArr) {
   const float t116 = *(xlArr + getBin(0., 1., params[0], 1));
   return t116 * params[0];
}

// CHECK: inline clad::ValueAndPushforward<unsigned int, unsigned int> getBin_pushforward(double low, double high, double val, unsigned int numBins, double _d_low, double _d_high, double _d_val, unsigned int _d_numBins);

// CHECK: float f_call_inline_fxn_darg0_0(float *params, const float *obs, const float *xlArr) {
// CHECK-NEXT:     clad::ValueAndPushforward<unsigned int, unsigned int> _t0 = getBin_pushforward(0., 1., params[0], 1, 0., 0., 1.F, 0);
// CHECK-NEXT:     const float _d_t116 = 0;
// CHECK-NEXT:     const float t116 = *(xlArr + _t0.value);
// CHECK-NEXT:     return _d_t116 * params[0] + t116 * 1.F;
// CHECK-NEXT: }

extern "C" int printf(const char* fmt, ...);
int main () { // expected-no-diagnostics
  auto f = clad::differentiate(g, 0);
  printf("g_darg0=%f\n", f.execute(1));
  //CHECK-EXEC: g_darg0=6.000000

  clad::differentiate(sqrt_func, 0);

  auto f1 = clad::differentiate(f_const_args_func_1, 0);
  printf("f1_darg0=%f\n", f1.execute(1.F,2.F));
  //CHECK-EXEC: f1_darg0=2.000000
  auto f2 = clad::differentiate(f_const_args_func_2, 0);
  printf("f2_darg0=%f\n", f2.execute(1.F,2.F));
  //CHECK-EXEC: f2_darg0=2.000000
  auto f3 = clad::differentiate(f_const_args_func_3, 0);
  printf("f3_darg0=%f\n", f3.execute(1.F,2.F));
  //CHECK-EXEC: f3_darg0=2.000000
  auto f4 = clad::differentiate(f_const_args_func_4, 0);
  printf("f4_darg0=%f\n", f4.execute(1.F,2.F,Vec()));
  //CHECK-EXEC: f4_darg0=2.000000
  auto f5 = clad::differentiate(f_const_args_func_5, 0);
  printf("f5_darg0=%f\n", f5.execute(1.F,2.F,Vec()));
  //CHECK-EXEC: f5_darg0=2.000000
  auto f6 = clad::differentiate(f_const_args_func_6, 0);
  printf("f6_darg0=%f\n", f6.execute(1.F,2.F,Vec()));
  //CHECK-EXEC: f6_darg0=2.000000
  auto f7 = clad::differentiate(f_const_args_func_7, 0);
  printf("f7_darg0=%f\n", f7.execute(1.F,2.F));
  //CHECK-EXEC: f7_darg0=2.000000
  auto f8 = clad::differentiate(f_const_args_func_8, 0);
  const float f8x = 1.F;
  printf("f8_darg0=%f\n", f8.execute(f8x,2.F));
  //CHECK-EXEC: f8_darg0=2.000000
  auto f9 = clad::differentiate(f_literal_args_func, 0);
  float z = 3.0;
  printf("f9_darg0=%.2f\n", f9.execute(1.F, 2.F, &z));
  //CHECK-EXEC: hello world f9_darg0=0.25
  auto f10 = clad::differentiate(f_call_inline_fxn, "params[0]");
  float params = 1.0, obs = 5.0, xlArr = 7.0;
  printf("f10_darg0_0=%.2f\n", f10.execute(&params, &obs, &xlArr));
  //CHECK-EXEC: f10_darg0_0=7.00

// CHECK: clad::ValueAndPushforward<float, float> f_const_helper_pushforward(const float x, const float _d_x) {
// CHECK-NEXT:     return {x * x, _d_x * x + x * _d_x};
// CHECK-NEXT: }

// CHECK: inline clad::ValueAndPushforward<unsigned int, unsigned int> getBin_pushforward(double low, double high, double val, unsigned int numBins, double _d_low, double _d_high, double _d_val, unsigned int _d_numBins) {
// CHECK-NEXT:     double _t0 = (high - low);
// CHECK-NEXT:     double _d_binWidth = ((_d_high - _d_low) * numBins - _t0 * _d_numBins) / (numBins * numBins);
// CHECK-NEXT:     double binWidth = _t0 / numBins;
// CHECK-NEXT:     double _t1 = (val - low);
// CHECK-NEXT:     {{(clad::)?}}ValueAndPushforward<double, double> _t2 = clad::custom_derivatives{{(::std)?}}::abs_pushforward(_t1 / binWidth, ((_d_val - _d_low) * binWidth - _t1 * _d_binWidth) / (binWidth * binWidth));
// CHECK-NEXT:     bool _t3 = val >= high;
// CHECK-NEXT:     return {(unsigned int)(_t3 ? numBins - 1 : _t2.value), (unsigned int)(_t3 ? _d_numBins - 0 : _t2.pushforward)};
// CHECK-NEXT: }

  return 0;
}
