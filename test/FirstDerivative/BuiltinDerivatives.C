// RUN: %cladclang %s -I%S/../../include -Xclang -verify -oBuiltinDerivatives.out 2>&1 | %filecheck %s
// RUN: ./BuiltinDerivatives.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -enable-tbr %s -I%S/../../include -Xclang -verify -oBuiltinDerivatives.out
// RUN: ./BuiltinDerivatives.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"
#include "../TestUtils.h"
extern "C" int printf(const char* fmt, ...);


namespace N {
  namespace impl {
    double sq(double x) { return x * x;}
  }
  using impl::sq; // using shadow
}

namespace clad {
  namespace custom_derivatives {
    namespace N {
      clad::ValueAndPushforward<double, double> sq_pushforward(double x, double d_x) {
        return { x * x, 2 * x * d_x };
      }
    }
  }
}

float f0 (float x) {
  return N::sq(x); // must find the sq_pushforward.
}

// CHECK: float f0_darg0(float x) {
// CHECK-NEXT: float _d_x = 1;
// CHECK-NEXT: {{.*}} _t0 = clad::custom_derivatives::N::sq_pushforward(x, _d_x);
// CHECK-NEXT: return _t0.pushforward;
// CHECK-NEXT: }

namespace clad{
  namespace custom_derivatives{
    float f1_darg0(float x) {
      return cos(x);
    }
  }
}

float f1(float x) {
  return sin(x);
}

// CHECK: float f1_darg0(float x) {
// CHECK-NEXT:    return cos(x);
// CHECK-NEXT: }

float f2(float x) {
  return cos(x);
}

// CHECK: float f2_darg0(float x) {
// CHECK-NEXT:     float _d_x = 1;
// CHECK-NEXT:     {{(clad::)?}}ValueAndPushforward<float, float> _t0 = clad::custom_derivatives::cos_pushforward(x, _d_x);
// CHECK-NEXT:     return _t0.pushforward;
// CHECK-NEXT: }

float f3(float x, float y) {
  return sin(x) + sin(y);
}

// CHECK: float f3_darg0(float x, float y) {
// CHECK-NEXT:     float _d_x = 1;
// CHECK-NEXT:     float _d_y = 0;
// CHECK-NEXT:     {{(clad::)?}}ValueAndPushforward<float, float> _t0 = clad::custom_derivatives::sin_pushforward(x, _d_x);
// CHECK-NEXT:     {{(clad::)?}}ValueAndPushforward<float, float> _t1 = clad::custom_derivatives::sin_pushforward(y, _d_y);
// CHECK-NEXT:     return _t0.pushforward + _t1.pushforward;
// CHECK-NEXT: }

// CHECK: float f3_darg1(float x, float y) {
// CHECK-NEXT:     float _d_x = 0;
// CHECK-NEXT:     float _d_y = 1;
// CHECK-NEXT:     {{(clad::)?}}ValueAndPushforward<float, float> _t0 = clad::custom_derivatives::sin_pushforward(x, _d_x);
// CHECK-NEXT:     {{(clad::)?}}ValueAndPushforward<float, float> _t1 = clad::custom_derivatives::sin_pushforward(y, _d_y);
// CHECK-NEXT:     return _t0.pushforward + _t1.pushforward;
// CHECK-NEXT: }

float f4(float x, float y) {
  return sin(x * x) + sin(y * y);
}

// CHECK: float f4_darg0(float x, float y) {
// CHECK-NEXT:     float _d_x = 1;
// CHECK-NEXT:     float _d_y = 0;
// CHECK-NEXT:     {{(clad::)?}}ValueAndPushforward<float, float> _t0 = clad::custom_derivatives::sin_pushforward(x * x, _d_x * x + x * _d_x);
// CHECK-NEXT:     {{(clad::)?}}ValueAndPushforward<float, float> _t1 = clad::custom_derivatives::sin_pushforward(y * y, _d_y * y + y * _d_y);
// CHECK-NEXT:     return _t0.pushforward + _t1.pushforward;
// CHECK-NEXT: }

// CHECK: float f4_darg1(float x, float y) {
// CHECK-NEXT:     float _d_x = 0;
// CHECK-NEXT:     float _d_y = 1;
// CHECK-NEXT:     {{(clad::)?}}ValueAndPushforward<float, float> _t0 = clad::custom_derivatives::sin_pushforward(x * x, _d_x * x + x * _d_x);
// CHECK-NEXT:     {{(clad::)?}}ValueAndPushforward<float, float> _t1 = clad::custom_derivatives::sin_pushforward(y * y, _d_y * y + y * _d_y);
// CHECK-NEXT:     return _t0.pushforward + _t1.pushforward;
// CHECK-NEXT: }

float f5(float x) {
  return exp(x);
}

// CHECK: float f5_darg0(float x) {
// CHECK-NEXT:     float _d_x = 1;
// CHECK-NEXT:     {{(clad::)?}}ValueAndPushforward<float, float> _t0 = clad::custom_derivatives::exp_pushforward(x, _d_x);
// CHECK-NEXT:     return _t0.pushforward;
// CHECK-NEXT: }

float f6(float x) {
  return exp(x * x);
}

// CHECK: float f6_darg0(float x) {
// CHECK-NEXT:     float _d_x = 1;
// CHECK-NEXT:     {{(clad::)?}}ValueAndPushforward<float, float> _t0 = clad::custom_derivatives::exp_pushforward(x * x, _d_x * x + x * _d_x);
// CHECK-NEXT:     return _t0.pushforward;
// CHECK-NEXT: }

float f7(float x) {
  return std::pow(x, 2.0);
}

// CHECK: float f7_darg0(float x) {
// CHECK-NEXT:     float _d_x = 1;
// CHECK-NEXT:     {{(clad::)?}}ValueAndPushforward<double, double> _t0 = clad::custom_derivatives::std::pow_pushforward(x, 2., _d_x, 0.);
// CHECK-NEXT:     return _t0.pushforward;
// CHECK-NEXT: }

void f7_grad(float x, float *_d_x);

// CHECK: void f7_grad(float x, float *_d_x) {
// CHECK-NEXT:     {
// CHECK-NEXT:         float _r0 = 0.F;
// CHECK-NEXT:         double _r1 = 0.;
// CHECK-NEXT:         {{(clad::)?}}custom_derivatives::std::pow_pullback(x, 2., 1, &_r0, &_r1);
// CHECK-NEXT:         *_d_x += _r0;
// CHECK-NEXT:     }
// CHECK-NEXT: }

double f8(float x) {
  return std::pow(x, 2);
}

// CHECK: double f8_darg0(float x) {
// CHECK-NEXT:     float _d_x = 1;
// CHECK-NEXT:     {{(clad::)?}}ValueAndPushforward<double, double> _t0 = clad::custom_derivatives::std::pow_pushforward(x, 2, _d_x, 0);
// CHECK-NEXT:     return _t0.pushforward;
// CHECK-NEXT: }

void f8_grad(float x, float *_d_x);

// CHECK: void f8_grad(float x, float *_d_x) {
// CHECK-NEXT:     {
// CHECK-NEXT:         float _r0 = 0.F;
// CHECK-NEXT:         int _r1 = 0;
// CHECK-NEXT:         {{(clad::)?}}custom_derivatives::std::pow_pullback(x, 2, 1, &_r0, &_r1);
// CHECK-NEXT:         *_d_x += _r0;
// CHECK-NEXT:     }
// CHECK-NEXT: }

float f9(float x, float y) {
  return std::pow(x, y);
}

// CHECK: float f9_darg0(float x, float y) {
// CHECK-NEXT:     float _d_x = 1;
// CHECK-NEXT:     float _d_y = 0;
// CHECK-NEXT:     {{(clad::)?}}ValueAndPushforward<float, float> _t0 = clad::custom_derivatives::std::pow_pushforward(x, y, _d_x, _d_y);
// CHECK-NEXT:     return _t0.pushforward;
// CHECK-NEXT: }

void f9_grad(float x, float y, float *_d_x, float *_d_y);

// CHECK: void f9_grad(float x, float y, float *_d_x, float *_d_y) {
// CHECK-NEXT:     {
// CHECK-NEXT:         float _r0 = 0.F;
// CHECK-NEXT:         float _r1 = 0.F;
// CHECK-NEXT:         {{(clad::)?}}custom_derivatives::std::pow_pullback(x, y, 1, &_r0, &_r1);
// CHECK-NEXT:         *_d_x += _r0;
// CHECK-NEXT:         *_d_y += _r1;
// CHECK-NEXT:     }
// CHECK-NEXT: }

double f10(float x, int y) {
  return std::pow(x, y);
}

// CHECK: double f10_darg0(float x, int y) {
// CHECK-NEXT:     float _d_x = 1;
// CHECK-NEXT:     int _d_y = 0;
// CHECK-NEXT:     {{(clad::)?}}ValueAndPushforward<double, double> _t0 = clad::custom_derivatives::std::pow_pushforward(x, y, _d_x, _d_y);
// CHECK-NEXT:     return _t0.pushforward;
// CHECK-NEXT: }

void f10_grad(float x, int y, float *_d_x, int *_d_y);

// CHECK: void f10_grad(float x, int y, float *_d_x, int *_d_y) {
// CHECK-NEXT:     {
// CHECK-NEXT:         float _r0 = 0.F;
// CHECK-NEXT:         int _r1 = 0;
// CHECK-NEXT:         {{(clad::)?}}custom_derivatives::std::pow_pullback(x, y, 1, &_r0, &_r1);
// CHECK-NEXT:         *_d_x += _r0;
// CHECK-NEXT:         *_d_y += _r1;
// CHECK-NEXT:     }
// CHECK-NEXT: }

double f11(double x, double y) {
  return std::pow((1.-x),2) + 100. * std::pow(y-std::pow(x,2),2);
}

// CHECK: void f11_grad(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:     typename {{.*}} _t0 = std::pow(y - std::pow(x, 2), 2);
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r0 = 0.;
// CHECK-NEXT:         int _r1 = 0;
// CHECK-NEXT:         {{(clad::)?}}custom_derivatives::std::pow_pullback((1. - x), 2, 1, &_r0, &_r1);
// CHECK-NEXT:         *_d_x += -_r0;
// CHECK-NEXT:         double _r2 = 0.;
// CHECK-NEXT:         int _r5 = 0;
// CHECK-NEXT:         {{(clad::)?}}custom_derivatives::std::pow_pullback(y - std::pow(x, 2), 2, 100. * 1, &_r2, &_r5);
// CHECK-NEXT:         *_d_y += _r2;
// CHECK-NEXT:         double _r3 = 0.;
// CHECK-NEXT:         int _r4 = 0;
// CHECK-NEXT:         {{(clad::)?}}custom_derivatives::std::pow_pullback(x, 2, -_r2, &_r3, &_r4);
// CHECK-NEXT:         *_d_x += _r3;
// CHECK-NEXT:     }
// CHECK-NEXT: }

double f12(double a, double b) { return std::fma(a, b, b); }

//CHECK: double f12_darg1(double a, double b) {
//CHECK-NEXT:     double _d_a = 0;
//CHECK-NEXT:     double _d_b = 1;
//CHECK-NEXT:     {{(clad::)?}}ValueAndPushforward<decltype(::std::fma(double(), double(), double())), decltype(::std::fma(double(), double(), double()))> _t0 = clad::custom_derivatives::std::fma_pushforward(a, b, b, _d_a, _d_b, _d_b);
//CHECK-NEXT:     return _t0.pushforward;
//CHECK-NEXT: }

namespace clad{
  namespace custom_derivatives{
    clad::ValueAndPushforward<double, double> custom_f13_pushforward(double x, double d_x) {
      return {exp(x), exp(x)*d_x};
    }
    clad::ValueAndPushforward<clad::ValueAndPushforward<double, double>, clad::ValueAndPushforward<double, double> > custom_f13_pushforward_pushforward(double x, double d_x, double _d_x, double _d_d_x) {
      return {{exp(x), exp(x)*d_x}, {exp(x)*_d_x, exp(x)*_d_x + exp(x)*_d_d_x}};
    }
  }
}
double custom_f13(double x) {
  return exp(x);
}
double f13(double x) {
  return custom_f13(x);
}

//CHECK: double f13_darg0(double x) {
//CHECK-NEXT:     double _d_x = 1;
//CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t0 = clad::custom_derivatives::custom_f13_pushforward(x, _d_x);
//CHECK-NEXT:     return _t0.pushforward;
//CHECK-NEXT: }
//CHECK-NEXT: double f13_d2arg0(double x) {
//CHECK-NEXT:     double _d_x = 1;
//CHECK-NEXT:     double _d__d_x = 0;
//CHECK-NEXT:     double _d_x0 = 1;
//CHECK-NEXT:     clad::ValueAndPushforward<clad::ValueAndPushforward<double, double>, clad::ValueAndPushforward<double, double> > _t0 = clad::custom_derivatives::custom_f13_pushforward_pushforward(x, _d_x0, _d_x, _d__d_x);
//CHECK-NEXT:     clad::ValueAndPushforward<double, double> _d__t0 = _t0.pushforward;
//CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t00 = _t0.value;
//CHECK-NEXT:     return _d__t0.pushforward;
//CHECK-NEXT: }

double f14(double x) {
  return __builtin_pow(x, 3);
}

double f15(double y, double x) {
  return std::atan2(y, x);
}

//CHECK:  {{float|double}} f15_darg0({{float|double}} y, {{float|double}} x) {
//CHECK-NEXT:      {{float|double}} _d_y = 1;
//CHECK-NEXT:      {{float|double}} _d_x = 0;
//CHECK-NEXT:      {{.*}}ValueAndPushforward<{{float|double}}, {{float|double}}> _t0 = {{.*}}atan2_pushforward(y, x, _d_y, _d_x);
//CHECK-NEXT:      return _t0.pushforward;
//CHECK-NEXT: }

//CHECK: {{float|double}} f15_darg1({{float|double}} y, {{float|double}} x) {
//CHECK-NEXT:     {{float|double}} _d_y = 0;
//CHECK-NEXT:     {{float|double}} _d_x = 1;
//CHECK-NEXT:     {{.*}}ValueAndPushforward<{{float|double}}, {{float|double}}> _t0 = {{.*}}atan2_pushforward(y, x, _d_y, _d_x);
//CHECK-NEXT:     return _t0.pushforward;
//CHECK-NEXT: }

void f15_grad(double y, double x, double *_d_y, double *_d_x);
//CHECK: void f15_grad(double y, double x, double *_d_y, double *_d_x) {
//CHECK:     {
//CHECK-NEXT:         double _r0 = 0{{.*}};
//CHECK-NEXT:         double _r1 = 0{{.*}};
//CHECK-NEXT:         {{.*}}atan2_pullback(y, x, 1, &_r0, &_r1);
//CHECK-NEXT:         *_d_y += _r0;
//CHECK-NEXT:         *_d_x += _r1;
//CHECK-NEXT:     }
//CHECK-NEXT: }

float f16(float x) {
  return std::acos(x);
}
// CHECK: {{float|double}} f16_darg0({{float|double}} x) {
//CHECK-NEXT:    {{float|double}} _d_x = 1;
//CHECK-NEXT:    {{.*}}ValueAndPushforward<{{float|double}}, {{float|double}}> _t0 = {{.*}}acos_pushforward(x, _d_x);
//CHECK-NEXT:    return _t0.pushforward;
//CHECK-NEXT: }

double f17(double x) {
  double y = std::fabs(x);
  return 2*y;
}

extern "C" {
  double f18(double x) { return x * x; }
}

double f_tan(double x) { return std::tan(x); }
// CHECK: double f_tan_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     {{.*}}ValueAndPushforward<double, double> _t0 = {{.*}}tan_pushforward(x, _d_x);
// CHECK-NEXT:     return _t0.pushforward;
// CHECK-NEXT: }

double f_cosh(double x) { return std::cosh(x); }
// CHECK: double f_cosh_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     {{.*}}ValueAndPushforward<double, double> _t0 = {{.*}}cosh_pushforward(x, _d_x);
// CHECK-NEXT:     return _t0.pushforward;
// CHECK-NEXT: }

double f_sinh(double x) { return std::sinh(x); }
// CHECK: double f_sinh_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     {{.*}}ValueAndPushforward<double, double> _t0 = {{.*}}sinh_pushforward(x, _d_x);
// CHECK-NEXT:     return _t0.pushforward;
// CHECK-NEXT: }

double f_asin(double x) { return std::asin(x); }
// CHECK: double f_asin_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     {{.*}}ValueAndPushforward<double, double> _t0 = {{.*}}asin_pushforward(x, _d_x);
// CHECK-NEXT:     return _t0.pushforward;
// CHECK-NEXT: }

double f_atan(double x) { return std::atan(x); }
// CHECK: double f_atan_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     {{.*}}ValueAndPushforward<double, double> _t0 = {{.*}}atan_pushforward(x, _d_x);
// CHECK-NEXT:     return _t0.pushforward;
// CHECK-NEXT: }

double f_log10(double x) { return std::log10(x); }
// CHECK: double f_log10_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     {{.*}}ValueAndPushforward<double, double> _t0 = {{.*}}log10_pushforward(x, _d_x);
// CHECK-NEXT:     return _t0.pushforward;
// CHECK-NEXT: }

double f_log2(double x) { return std::log2(x); }
// CHECK: double f_log2_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     {{.*}}ValueAndPushforward<double, double> _t0 = {{.*}}log2_pushforward(x, _d_x);
// CHECK-NEXT:     return _t0.pushforward;
// CHECK-NEXT: }

double f_cbrt(double x) { return std::cbrt(x); }
// CHECK: double f_cbrt_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     {{.*}}ValueAndPushforward<double, double> _t0 = {{.*}}cbrt_pushforward(x, _d_x);
// CHECK-NEXT:     return _t0.pushforward;
// CHECK-NEXT: }

double f_hypot(double x, double y) { return std::hypot(x, y); }
// CHECK: double f_hypot_darg0(double x, double y) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     double _d_y = 0;
// CHECK-NEXT:     {{.*}}ValueAndPushforward<double, double> _t0 = {{.*}}hypot_pushforward(x, y, _d_x, _d_y);
// CHECK-NEXT:     return _t0.pushforward;
// CHECK-NEXT: }

double f_fmodf(double x, double y) { return ::fmodf(x, y); }
// CHECK: double f_fmodf_darg0(double x, double y) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     double _d_y = 0;
// CHECK-NEXT:     {{.*}}ValueAndPushforward<double, double> _t0 = {{.*}}fmodf_pushforward(x, y, _d_x, _d_y);
// CHECK-NEXT:     return _t0.pushforward;
// CHECK-NEXT: }

double f_remainder(double x, double y) { return std::remainder(x, y); }
// CHECK: double f_remainder_darg0(double x, double y) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     double _d_y = 0;
// CHECK-NEXT:     {{.*}}ValueAndPushforward<double, double> _t0 = {{.*}}remainder_pushforward(x, y, _d_x, _d_y);
// CHECK-NEXT:     return _t0.pushforward;
// CHECK-NEXT: }

double f_fmaxf(double x, double y) { return std::fmaxf(x, y); }
// CHECK: double f_fmaxf_darg0(double x, double y) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     double _d_y = 0;
// CHECK-NEXT:     {{.*}}ValueAndPushforward<double, double> _t0 = {{.*}}fmaxf_pushforward(x, y, _d_x, _d_y);
// CHECK-NEXT:     return _t0.pushforward;
// CHECK-NEXT: }

double f_fmin(double x, double y) { return std::fmin(x, y); }
// CHECK: double f_fmin_darg0(double x, double y) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     double _d_y = 0;
// CHECK-NEXT:     {{.*}}ValueAndPushforward<double, double> _t0 = {{.*}}fmin_pushforward(x, y, _d_x, _d_y);
// CHECK-NEXT:     return _t0.pushforward;
// CHECK-NEXT: }

double f_fdim(double x, double y) { return std::fdim(x, y); }
// CHECK: double f_fdim_darg0(double x, double y) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     double _d_y = 0;
// CHECK-NEXT:     {{.*}}ValueAndPushforward<double, double> _t0 = {{.*}}fdim_pushforward(x, y, _d_x, _d_y);
// CHECK-NEXT:     return _t0.pushforward;
// CHECK-NEXT: }

double f_exp2f(double x) { return std::exp2f(x); }
// CHECK: double f_exp2f_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     {{.*}}ValueAndPushforward<double, double> _t0 = {{.*}}exp2f_pushforward(x, _d_x);
// CHECK-NEXT:     return _t0.pushforward;
// CHECK-NEXT: }

double f_expm1l(double x) { return std::expm1l(x); }
// CHECK: double f_expm1l_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     {{.*}}ValueAndPushforward<double, double> _t0 = {{.*}}expm1l_pushforward(x, _d_x);
// CHECK-NEXT:     return _t0.pushforward;
// CHECK-NEXT: }

double f_log1pl(double x) { return std::log1pl(x); }
// CHECK: double f_log1pl_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     {{.*}}ValueAndPushforward<double, double> _t0 = {{.*}}log1pl_pushforward(x, _d_x);
// CHECK-NEXT:     return _t0.pushforward;
// CHECK-NEXT: }

double f_tanhl(double x) { return tanhl(x); }
// CHECK: double f_tanhl_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     {{.*}}ValueAndPushforward<double, double> _t0 = {{.*}}tanhl_pushforward(x, _d_x);
// CHECK-NEXT:     return _t0.pushforward;
// CHECK-NEXT: }

double f_asinhf(double x) { return std::asinhf(x); }
// CHECK: double f_asinhf_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     {{.*}}ValueAndPushforward<double, double> _t0 = {{.*}}asinhf_pushforward(x, _d_x);
// CHECK-NEXT:     return _t0.pushforward;
// CHECK-NEXT: }

double f_acoshl(double x) { return std::acoshl(x); }
// CHECK: double f_acoshl_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     {{.*}}ValueAndPushforward<double, double> _t0 = {{.*}}acoshl_pushforward(x, _d_x);
// CHECK-NEXT:     return _t0.pushforward;
// CHECK-NEXT: }

double f_atanhl(double x) { return std::atanhl(x); }
// CHECK: double f_atanhl_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     {{.*}}ValueAndPushforward<double, double> _t0 = {{.*}}atanhl_pushforward(x, _d_x);
// CHECK-NEXT:     return _t0.pushforward;
// CHECK-NEXT: }

double f_erff(double x) { return std::erff(x); }
// CHECK: double f_erff_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     {{.*}}ValueAndPushforward<double, double> _t0 = {{.*}}erff_pushforward(x, _d_x);
// CHECK-NEXT:     return _t0.pushforward;
// CHECK-NEXT: }

double f_erfc(double x) { return std::erfc(x); }
// CHECK: double f_erfc_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     {{.*}}ValueAndPushforward<double, double> _t0 = {{.*}}erfc_pushforward(x, _d_x);
// CHECK-NEXT:     return _t0.pushforward;
// CHECK-NEXT: }

double f_pow_zero(double x, double y) { return std::pow(x, y); }
// CHECK: double f_pow_zero_darg0(double x, double y) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     double _d_y = 0;
// CHECK-NEXT:     {{.*}}ValueAndPushforward<double, double> _t0 = {{.*}}pow_pushforward(x, y, _d_x, _d_y);
// CHECK-NEXT:     return _t0.pushforward;
// CHECK-NEXT: }

int main () { //expected-no-diagnostics
  float f_result[2];
  double d_result[2];
  int i_result[1];

  auto f0_darg0 = clad::differentiate(f0, 0);
  printf("Result is = %f\n", f0_darg0.execute(1)); // CHECK-EXEC: Result is = 2

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

  auto f13_ddx = clad::differentiate<2>(f13);
  printf("Result is = %.2f\n", f13_ddx.execute(1)); //CHECK-EXEC: Result is = 2.72

  auto f14_darg0 = clad::differentiate(f14, 0);
  printf("Result is = %f\n", f14_darg0.execute(1)); //CHECK-EXEC: Result is = 3.000000

  auto f14_ddarg0 = clad::differentiate<2>(f14, 0);
  printf("Result is = %f\n", f14_ddarg0.execute(1)); //CHECK-EXEC: Result is = 6.000000

  auto f15_darg0 = clad::differentiate(f15, 0);
  printf("Result is = %f\n", f15_darg0.execute(4, 3)); //CHECK-EXEC: Result is = 0.120000

  auto f15_darg1 = clad::differentiate(f15, 1);
  printf("Result is = %f\n", f15_darg1.execute(4, 3)); //CHECK-EXEC: Result is = -0.160000

  d_result[0] = d_result[1] = 0;
  clad::gradient(f15);
  f15_grad(4, 3, &d_result[0], &d_result[1]);
  printf("Result is = {%f, %f}\n", d_result[0], d_result[1]); //CHECK-EXEC: Result is = {0.120000, -0.160000}

  auto f16_darg0 = clad::differentiate(f16, 0);
  printf("Result is = %f\n", f16_darg0.execute(0.9)); //CHECK-EXEC: Result is = -2.294157

  INIT_GRADIENT(f17);

  TEST_GRADIENT(f17, /*numOfDerivativeArgs=*/1, -3, &d_result[0]); // CHECK-EXEC: {-2.00}

  auto f18_darg0 = clad::differentiate(f18, 0);
  printf("Result is = %f\n", f18_darg0.execute(1)); // CHECK-EXEC: Result is = 2

  auto d_tan = clad::differentiate(f_tan, 0);
  printf("Result is = %.6f\n", d_tan.execute(0.5)); // CHECK-EXEC: Result is = 1.298446

  auto d_cosh = clad::differentiate(f_cosh, 0);
  printf("Result is = %.6f\n", d_cosh.execute(0.5)); // CHECK-EXEC: Result is = 0.521095

  auto d_sinh = clad::differentiate(f_sinh, 0);
  printf("Result is = %.6f\n", d_sinh.execute(0.5)); // CHECK-EXEC: Result is = 1.127626

  auto d_asin = clad::differentiate(f_asin, 0);
  printf("Result is = %.6f\n", d_asin.execute(0.5)); // CHECK-EXEC: Result is = 1.154701

  auto d_atan = clad::differentiate(f_atan, 0);
  printf("Result is = %.6f\n", d_atan.execute(0.5)); // CHECK-EXEC: Result is = 0.800000

  auto d_log10 = clad::differentiate(f_log10, 0);
  printf("Result is = %.6f\n", d_log10.execute(0.5)); // CHECK-EXEC: Result is = 0.868589

  auto d_log2 = clad::differentiate(f_log2, 0);
  printf("Result is = %.6f\n", d_log2.execute(0.5)); // CHECK-EXEC: Result is = 2.885390

  auto d_cbrt = clad::differentiate(f_cbrt, 0);
  printf("Result is = %.6f\n", d_cbrt.execute(0.5)); // CHECK-EXEC: Result is = 0.529134

  auto d_hypot = clad::differentiate(f_hypot, 0);
  printf("Result is = %.6f\n", d_hypot.execute(0.5, 0.5)); // CHECK-EXEC: Result is = 0.707107

  auto d_fmodf = clad::differentiate(f_fmodf, 0);
  printf("Result is = %.6f\n", d_fmodf.execute(0.5, 0.5)); // CHECK-EXEC: Result is = 1.000000

  auto d_remainder = clad::differentiate(f_remainder, 0);
  printf("Result is = %.6f\n", d_remainder.execute(0.5, 0.5)); // CHECK-EXEC: Result is = 1.000000

  auto d_fmaxf = clad::differentiate(f_fmaxf, 0);
  printf("Result is = %.6f\n", d_fmaxf.execute(1, 0.5)); // CHECK-EXEC: Result is = 1.000000

  auto d_fmin = clad::differentiate(f_fmin, 0);
  printf("Result is = %.6f\n", d_fmin.execute(0.5, 0.5)); // CHECK-EXEC: Result is = 0.000000

  auto d_fdim = clad::differentiate(f_fdim, 0);
  printf("Result is = %.6f\n", d_fdim.execute(1, 0.5)); // CHECK-EXEC: Result is = 1.000000

  auto d_exp2f = clad::differentiate(f_exp2f, 0);
  printf("Result is = %.6f\n", d_exp2f.execute(0.5)); // CHECK-EXEC: Result is = 0.980258

  auto d_expm1l = clad::differentiate(f_expm1l, 0);
  printf("Result is = %.6f\n", d_expm1l.execute(0.5)); // CHECK-EXEC: Result is = 1.648721

  auto d_log1pl = clad::differentiate(f_log1pl, 0);
  printf("Result is = %.6f\n", d_log1pl.execute(0.5)); // CHECK-EXEC: Result is = 0.666667

  auto d_tanhl = clad::differentiate(f_tanhl, 0);
  printf("Result is = %.6f\n", d_tanhl.execute(0.5)); // CHECK-EXEC: Result is = 0.786448

  auto d_asinhf = clad::differentiate(f_asinhf, 0);
  printf("Result is = %.6f\n", d_asinhf.execute(0.5)); // CHECK-EXEC: Result is = 0.894427

  auto d_acoshl = clad::differentiate(f_acoshl, 0);
  printf("Result is = %.6f\n", d_acoshl.execute(1.5)); // CHECK-EXEC: Result is = 0.894427

  auto d_atanhl = clad::differentiate(f_atanhl, 0);
  printf("Result is = %.6f\n", d_atanhl.execute(0.5)); // CHECK-EXEC: Result is = 1.333333

  auto d_erff = clad::differentiate(f_erff, 0);
  printf("Result is = %.6f\n", d_erff.execute(0.5)); // CHECK-EXEC: Result is = 0.878783

  auto d_erfc = clad::differentiate(f_erfc, 0);
  printf("Result is = %.6f\n", d_erfc.execute(0.5)); // CHECK-EXEC: Result is = -0.878783

  auto d_pow_zero =  clad::differentiate(f_pow_zero, 0);
  printf("Result is = %.6f\n", d_pow_zero.execute(0, 0)); // CHECK-EXEC: Result is = 0.000000

  return 0;
}
