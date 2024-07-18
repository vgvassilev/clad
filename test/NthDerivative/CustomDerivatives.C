// RUN: %cladclang %s -I%S/../../include -oCustomDerivative.out 2>&1 | %filecheck %s
// RUN: ./CustomDerivative.out | %filecheck_exec %s
#include "clad/Differentiator/Differentiator.h"
#include "clad/Differentiator/BuiltinDerivatives.h"
#include "../TestUtils.h"

extern "C" int printf(const char* fmt, ...);

float test_sin(float x) {
  return std::sin(x);
}

// CHECK: float test_sin_d2arg0(float x) {
// CHECK-NEXT:    float _d_x = 1;
// CHECK-NEXT:    float _d__d_x = 0;
// CHECK-NEXT:    float _d_x0 = 1;
// CHECK-NEXT:    clad::ValueAndPushforward<ValueAndPushforward<float, float>, ValueAndPushforward<float, float> > _t0 = sin_pushforward_pushforward(x, _d_x0, _d_x, _d__d_x);
// CHECK-NEXT:    ValueAndPushforward<float, float> _d__t0 = _t0.pushforward;
// CHECK-NEXT:    ValueAndPushforward<float, float> _t00 = _t0.value;
// CHECK-NEXT:    return _d__t0.pushforward;
// CHECK-NEXT:}

float test_cos(float x) {
    return std::cos(x);
}

// CHECK: float test_cos_d2arg0(float x) {
// CHECK-NEXT:    float _d_x = 1;
// CHECK-NEXT:    float _d__d_x = 0;
// CHECK-NEXT:    float _d_x0 = 1;
// CHECK-NEXT:    clad::ValueAndPushforward<ValueAndPushforward<float, float>, ValueAndPushforward<float, float> > _t0 = cos_pushforward_pushforward(x, _d_x0, _d_x, _d__d_x);
// CHECK-NEXT:    ValueAndPushforward<float, float> _d__t0 = _t0.pushforward;
// CHECK-NEXT:    ValueAndPushforward<float, float> _t00 = _t0.value;
// CHECK-NEXT:    return _d__t0.pushforward;
// CHECK-NEXT:}

float test_trig(float x, float y, int a, int b) {
    float k = pow(std::sin(x*y), a) * pow(std::cos(x*y), b);
    return k;
}

// CHECK: float test_trig_d2arg0(float x, float y, int a, int b) {
// CHECK-NEXT:    float _d_x = 1;
// CHECK-NEXT:    float _d_y = 0;
// CHECK-NEXT:    int _d_a = 0;
// CHECK-NEXT:    int _d_b = 0;
// CHECK-NEXT:    float _d__d_x = 0;
// CHECK-NEXT:    float _d_x0 = 1;
// CHECK-NEXT:    float _d__d_y = 0;
// CHECK-NEXT:    float _d_y0 = 0;
// CHECK-NEXT:    int _d__d_a = 0;
// CHECK-NEXT:    int _d_a0 = 0;
// CHECK-NEXT:    int _d__d_b = 0;
// CHECK-NEXT:    int _d_b0 = 0;
// CHECK-NEXT:    clad::ValueAndPushforward<ValueAndPushforward<float, float>, ValueAndPushforward<float, float> > _t0 = clad::custom_derivatives::std::sin_pushforward_pushforward(x * y, _d_x0 * y + x * _d_y0, _d_x * y + x * _d_y, _d__d_x * y + _d_x0 * _d_y + _d_x * _d_y0 + x * _d__d_y);
// CHECK-NEXT:    ValueAndPushforward<float, float> _d__t0 = _t0.pushforward;
// CHECK-NEXT:    ValueAndPushforward<float, float> _t00 = _t0.value;
// CHECK-NEXT:    clad::ValueAndPushforward<ValueAndPushforward<decltype(::std::pow(float(), int())), decltype(::std::pow(float(), int()))>, ValueAndPushforward<decltype(::std::pow(float(), int())), decltype(::std::pow(float(), int()))> > _t1 = pow_pushforward_pushforward(_t00.value, a, _t00.pushforward, _d_a0, _d__t0.value, _d_a, _d__t0.pushforward, _d__d_a);
// CHECK-NEXT:    ValueAndPushforward<decltype(::std::pow(float(), int())), decltype(::std::pow(float(), int()))> _d__t1 = _t1.pushforward;
// CHECK-NEXT:    ValueAndPushforward<decltype(::std::pow(float(), int())), decltype(::std::pow(float(), int()))> _t10 = _t1.value;
// CHECK-NEXT:    clad::ValueAndPushforward<ValueAndPushforward<float, float>, ValueAndPushforward<float, float> > _t2 = clad::custom_derivatives::std::cos_pushforward_pushforward(x * y, _d_x0 * y + x * _d_y0, _d_x * y + x * _d_y, _d__d_x * y + _d_x0 * _d_y + _d_x * _d_y0 + x * _d__d_y);
// CHECK-NEXT:    ValueAndPushforward<float, float> _d__t2 = _t2.pushforward;
// CHECK-NEXT:    ValueAndPushforward<float, float> _t20 = _t2.value;
// CHECK-NEXT:    clad::ValueAndPushforward<ValueAndPushforward<decltype(::std::pow(float(), int())), decltype(::std::pow(float(), int()))>, ValueAndPushforward<decltype(::std::pow(float(), int())), decltype(::std::pow(float(), int()))> > _t3 = clad::custom_derivatives::std::pow_pushforward_pushforward(_t20.value, b, _t20.pushforward, _d_b0, _d__t2.value, _d_b, _d__t2.pushforward, _d__d_b);
// CHECK-NEXT:    ValueAndPushforward<decltype(::std::pow(float(), int())), decltype(::std::pow(float(), int()))> _d__t3 = _t3.pushforward;
// CHECK-NEXT:    ValueAndPushforward<decltype(::std::pow(float(), int())), decltype(::std::pow(float(), int()))> _t30 = _t3.value;
// CHECK-NEXT:    double &_d__t4 = _d__t1.value;
// CHECK-NEXT:    double &_t40 = _t10.value;
// CHECK-NEXT:    double &_d__t5 = _d__t3.value;
// CHECK-NEXT:    double &_t50 = _t30.value;
// CHECK-NEXT:    double &_t4 = _t10.pushforward;
// CHECK-NEXT:    double &_t5 = _t30.pushforward;
// CHECK-NEXT:    float _d__d_k = _d__t1.pushforward * _t50 + _t4 * _d__t5 + _d__t4 * _t5 + _t40 * _d__t3.pushforward;
// CHECK-NEXT:    float _d_k = _t4 * _t50 + _t40 * _t5;
// CHECK-NEXT:    float _d_k0 = _d__t4 * _t50 + _t40 * _d__t5;
// CHECK-NEXT:    float k = _t40 * _t50;
// CHECK-NEXT:    return _d__d_k;
// CHECK-NEXT:}
// CHECK:   float test_trig_d2arg1(float x, float y, int a, int b) {
// CHECK-NEXT:    float _d_x = 0;
// CHECK-NEXT:    float _d_y = 1;
// CHECK-NEXT:    int _d_a = 0;
// CHECK-NEXT:    int _d_b = 0;
// CHECK-NEXT:    float _d__d_x = 0;
// CHECK-NEXT:    float _d_x0 = 0;
// CHECK-NEXT:    float _d__d_y = 0;
// CHECK-NEXT:    float _d_y0 = 1;
// CHECK-NEXT:    int _d__d_a = 0;
// CHECK-NEXT:    int _d_a0 = 0;
// CHECK-NEXT:    int _d__d_b = 0;
// CHECK-NEXT:    int _d_b0 = 0;
// CHECK-NEXT:    clad::ValueAndPushforward<ValueAndPushforward<float, float>, ValueAndPushforward<float, float> > _t0 = clad::custom_derivatives::std::sin_pushforward_pushforward(x * y, _d_x0 * y + x * _d_y0, _d_x * y + x * _d_y, _d__d_x * y + _d_x0 * _d_y + _d_x * _d_y0 + x * _d__d_y);
// CHECK-NEXT:    ValueAndPushforward<float, float> _d__t0 = _t0.pushforward;
// CHECK-NEXT:    ValueAndPushforward<float, float> _t00 = _t0.value;
// CHECK-NEXT:    clad::ValueAndPushforward<ValueAndPushforward<decltype(::std::pow(float(), int())), decltype(::std::pow(float(), int()))>, ValueAndPushforward<decltype(::std::pow(float(), int())), decltype(::std::pow(float(), int()))> > _t1 = clad::custom_derivatives::std::pow_pushforward_pushforward(_t00.value, a, _t00.pushforward, _d_a0, _d__t0.value, _d_a, _d__t0.pushforward, _d__d_a);
// CHECK-NEXT:    ValueAndPushforward<decltype(::std::pow(float(), int())), decltype(::std::pow(float(), int()))> _d__t1 = _t1.pushforward;
// CHECK-NEXT:    ValueAndPushforward<decltype(::std::pow(float(), int())), decltype(::std::pow(float(), int()))> _t10 = _t1.value;
// CHECK-NEXT:    clad::ValueAndPushforward<ValueAndPushforward<float, float>, ValueAndPushforward<float, float> > _t2 = clad::custom_derivatives::std::cos_pushforward_pushforward(x * y, _d_x0 * y + x * _d_y0, _d_x * y + x * _d_y, _d__d_x * y + _d_x0 * _d_y + _d_x * _d_y0 + x * _d__d_y);
// CHECK-NEXT:    ValueAndPushforward<float, float> _d__t2 = _t2.pushforward;
// CHECK-NEXT:    ValueAndPushforward<float, float> _t20 = _t2.value;
// CHECK-NEXT:    clad::ValueAndPushforward<ValueAndPushforward<decltype(::std::pow(float(), int())), decltype(::std::pow(float(), int()))>, ValueAndPushforward<decltype(::std::pow(float(), int())), decltype(::std::pow(float(), int()))> > _t3 = clad::custom_derivatives::std::pow_pushforward_pushforward(_t20.value, b, _t20.pushforward, _d_b0, _d__t2.value, _d_b, _d__t2.pushforward, _d__d_b);
// CHECK-NEXT:    ValueAndPushforward<decltype(::std::pow(float(), int())), decltype(::std::pow(float(), int()))> _d__t3 = _t3.pushforward;
// CHECK-NEXT:    ValueAndPushforward<decltype(::std::pow(float(), int())), decltype(::std::pow(float(), int()))> _t30 = _t3.value;
// CHECK-NEXT:    double &_d__t4 = _d__t1.value;
// CHECK-NEXT:    double &_t40 = _t10.value;
// CHECK-NEXT:    double &_d__t5 = _d__t3.value;
// CHECK-NEXT:    double &_t50 = _t30.value;
// CHECK-NEXT:    double &_t4 = _t10.pushforward;
// CHECK-NEXT:    double &_t5 = _t30.pushforward;
// CHECK-NEXT:    float _d__d_k = _d__t1.pushforward * _t50 + _t4 * _d__t5 + _d__t4 * _t5 + _t40 * _d__t3.pushforward;
// CHECK-NEXT:    float _d_k = _t4 * _t50 + _t40 * _t5;
// CHECK-NEXT:    float _d_k0 = _d__t4 * _t50 + _t40 * _d__t5;
// CHECK-NEXT:    float k = _t40 * _t50;
// CHECK-NEXT:    return _d__d_k;
// CHECK-NEXT:}

float test_exp(float x) {
    return exp(x * x);
}

// CHECK:   float test_exp_darg0(float x) {
// CHECK-NEXT:    float _d_x = 1;
// CHECK-NEXT:    ValueAndPushforward<float, float> _t0 = clad::custom_derivatives::exp_pushforward(x * x, _d_x * x + x * _d_x);
// CHECK-NEXT:    return _t0.pushforward;
// CHECK-NEXT:}

// CHECK:   clad::ValueAndPushforward<ValueAndPushforward<float, float>, ValueAndPushforward<float, float> > exp_pushforward_pushforward(float x, float d_x, float _d_x, float _d_d_x);

// CHECK:   float test_exp_d2arg0(float x) {
// CHECK-NEXT:    float _d_x = 1;
// CHECK-NEXT:    float _d__d_x = 0;
// CHECK-NEXT:    float _d_x0 = 1;
// CHECK-NEXT:    clad::ValueAndPushforward<ValueAndPushforward<float, float>, ValueAndPushforward<float, float> > _t0 = exp_pushforward_pushforward(x * x, _d_x0 * x + x * _d_x0, _d_x * x + x * _d_x, _d__d_x * x + _d_x0 * _d_x + _d_x * _d_x0 + x * _d__d_x);
// CHECK-NEXT:    {{(clad::)?}}ValueAndPushforward<float, float> _d__t0 = _t0.pushforward;
// CHECK-NEXT:    {{(clad::)?}}ValueAndPushforward<float, float> _t00 = _t0.value;
// CHECK-NEXT:    return _d__t0.pushforward;
// CHECK-NEXT:}

float test_sin_d2arg0(float x);
float test_cos_d2arg0(float x);
float test_trig_d2arg0(float x, float y, int a, int b);
float test_trig_d2arg1(float x, float y, int a, int b);
float test_exp_darg0(float x);
float test_exp_d2arg0(float x);

int main() {
    clad::differentiate<2>(test_sin);
    printf("Result is = %f\n", test_sin_d2arg0(0.5)); // CHECK-EXEC: Result is = -0.479426

    clad::differentiate<2>(test_cos);
    printf("Result is = %f\n", test_cos_d2arg0(1.0)); // CHECK-EXEC: Result is = -0.540302

    clad::differentiate<2>(test_trig, "x");
    printf("Result is = %f\n", test_trig_d2arg0(0.5, 1.0, 1, 2)); // CHECK-EXEC: Result is = -2.364220

    clad::differentiate<2>(test_trig, "y");
    printf("Result is = %f\n", test_trig_d2arg1(1.0, 0.5, 2, 1)); // CHECK-EXEC: Result is = -0.060237

    clad::differentiate<2>(test_exp);
    printf("Result is = %f\n", test_exp_d2arg0(2)); // CHECK-EXEC: Result is = 982.766663

// CHECK:   clad::ValueAndPushforward<ValueAndPushforward<float, float>, ValueAndPushforward<float, float> > exp_pushforward_pushforward(float x, float d_x, float _d_x, float _d_d_x) {
// CHECK-NEXT:    {{(clad::)?}}ValueAndPushforward<float, float> _t0 = clad::custom_derivatives{{(::std)?}}::exp_pushforward(x, _d_x);
// CHECK-NEXT:    {{(clad::)?}}ValueAndPushforward<float, float> _t1 = clad::custom_derivatives{{(::std)?}}::exp_pushforward(x, _d_x);
// CHECK-NEXT:    float &_t2 = _t1.value;
// CHECK-NEXT:    return {{[{][(]ValueAndPushforward<float, float>[)][{]}}_t0.value, _t2 * d_x}, (ValueAndPushforward<float, float>){_t0.pushforward, _t1.pushforward * d_x + _t2 * _d_d_x{{[}][}]}};
// CHECK-NEXT:}

}