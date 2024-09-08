// RUN: %cladnumdiffclang %s -I%S/../../include -fsyntax-only -Xclang -verify 2>&1 | %filecheck %s

//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

#include <algorithm>
#include <cmath>

int printf(const char* fmt, ...);
int no_body(int x);
int custom_fn(int x);
float custom_fn(float x);
int custom_fn();

int overloaded(int x) {
  printf("A was called.\n");
  return x*x;
}

float overloaded(float x) {
  return x;
}

int overloaded() {
  return 3;
}

float test_1(float x) {
  return overloaded(x) + custom_fn(x);
}

namespace clad {
namespace custom_derivatives {
clad::ValueAndPushforward<float, float> overloaded_pushforward(float x,
                                                               float d_x) {
  return {overloaded(x), x * d_x};
}

clad::ValueAndPushforward<int, int> overloaded_pushforward(int x, int d_x) {
  return {overloaded(x), x * d_x};
}

clad::ValueAndPushforward<float, float> no_body_pushforward(float x,
                                                            float d_x) {
  return {0, 1 * d_x};
}

clad::ValueAndPushforward<int, int> custom_fn_pushforward(int x, int d_x) {
  return {custom_fn(x), (x + x) * d_x};
}

clad::ValueAndPushforward<float, float> custom_fn_pushforward(float x,
                                                              float d_x) {
  return {custom_fn(x), x * x * d_x};
}

clad::ValueAndPushforward<int, int> custom_fn_pushforward() { return {0, 5}; }
} // namespace custom_derivatives
} // namespace clad

// CHECK: float test_1_darg0(float x) {
// CHECK-NEXT: float _d_x = 1;
// CHECK-NEXT: {{(clad::)?}}ValueAndPushforward<float, float> _t0 = clad::custom_derivatives::overloaded_pushforward(x, _d_x);
// CHECK-NEXT: {{(clad::)?}}ValueAndPushforward<float, float> _t1 = clad::custom_derivatives::custom_fn_pushforward(x, _d_x);
// CHECK-NEXT: return _t0.pushforward + _t1.pushforward;
// CHECK-NEXT: }

float test_2(float x) {
  return overloaded(x) + custom_fn(x);
}

// CHECK: float test_2_darg0(float x) {
// CHECK-NEXT: float _d_x = 1;
// CHECK-NEXT: {{(clad::)?}}ValueAndPushforward<float, float> _t0 = clad::custom_derivatives::overloaded_pushforward(x, _d_x);
// CHECK-NEXT: {{(clad::)?}}ValueAndPushforward<float, float> _t1 = clad::custom_derivatives::custom_fn_pushforward(x, _d_x);
// CHECK-NEXT: return _t0.pushforward + _t1.pushforward;
// CHECK-NEXT: }

float test_3() {
  return custom_fn();
}

// CHECK-NOT: float test_3_darg0() {

float test_4(int x) {
  return overloaded();
}

// CHECK: float test_4_darg0(int x) {
// CHECK-NEXT: int _d_x = 1;
// CHECK-NEXT: return 0;
// CHECK-NEXT: }

float test_5(int x) {
  return no_body(x);
}

// CHECK: float test_5_darg0(int x) {
// CHECK-NEXT: int _d_x = 1;
// CHECK-NEXT: {{(clad::)?}}ValueAndPushforward<float, float> _t0 = clad::custom_derivatives::no_body_pushforward(x, _d_x);
// CHECK-NEXT: return _t0.pushforward;
// CHECK-NEXT: }

float test_6(float x, float y) {
  return std::sin(x) + std::cos(y);
}

// CHECK: float test_6_darg0(float x, float y) {
// CHECK-NEXT: float _d_x = 1;
// CHECK-NEXT: float _d_y = 0;
// CHECK-NEXT: {{(clad::)?}}ValueAndPushforward<float, float> _t0 = clad::custom_derivatives{{(::std)?}}::sin_pushforward(x, _d_x);
// CHECK-NEXT: {{(clad::)?}}ValueAndPushforward<float, float> _t1 = clad::custom_derivatives{{(::std)?}}::cos_pushforward(y, _d_y);
// CHECK-NEXT: return _t0.pushforward + _t1.pushforward;
// CHECK-NEXT: }

void increment(int &i) {
  ++i;
}

double test_7(double i, double j) {
  double res = 0;
  for (int i=0; i < 5; increment(i))
    res += i*j;
  return res;
}

// CHECK: void increment_pushforward(int &i, int &_d_i);

// CHECK: double test_7_darg0(double i, double j) {
// CHECK-NEXT: double _d_i = 1;
// CHECK-NEXT: double _d_j = 0;
// CHECK-NEXT: double _d_res = 0;
// CHECK-NEXT: double res = 0;
// CHECK-NEXT: {
// CHECK-NEXT:    int _d_i0 = 0;
// CHECK-NEXT:    for (int i0 = 0; i0 < 5; increment_pushforward(i0, _d_i0)) {
// CHECK-NEXT:      _d_res += _d_i0 * j + i0 * _d_j;
// CHECK-NEXT:      res += i0 * j;
// CHECK-NEXT:    }
// CHECK-NEXT: }
// CHECK-NEXT: return _d_res;
// CHECK-NEXT: }

enum E {A, B, C};
double func_with_enum(double x, E e) {
  return x*x;
}

double test_8(double x) {
  E e;
  return func_with_enum(x, e);
}

// CHECK: clad::ValueAndPushforward<double, double> func_with_enum_pushforward(double x, E e, double _d_x);

// CHECK: double test_8_darg0(double x) {
// CHECK-NEXT: double _d_x = 1;
// CHECK-NEXT: E _d_e;
// CHECK-NEXT: E e;
// CHECK-NEXT: {{(clad::)?}}ValueAndPushforward<double, double> _t0 = func_with_enum_pushforward(x, e, _d_x);
// CHECK-NEXT: return _t0.pushforward;
// CHECK-NEXT: }

class A {
  public:
  static double static_method(double x);
};

double A::static_method(double x) {
  return x;
}

double test_9(double x) {
  return A::static_method(x);
}

// CHECK: static clad::ValueAndPushforward<double, double> static_method_pushforward(double x, double _d_x);

// CHECK: double test_9_darg0(double x) {
// CHECK-NEXT: double _d_x = 1;
// CHECK-NEXT: clad::ValueAndPushforward<double, double> _t0 = static_method_pushforward(x, _d_x);
// CHECK-NEXT: return _t0.pushforward;
// CHECK-NEXT: }

void some_important_void_func(double y) {
    assert(y >= 1);
}

double test_10(double x) {
  some_important_void_func(1);
  return x;
}

// CHECK:      double test_10_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     some_important_void_func(1);
// CHECK-NEXT:     return _d_x;
// CHECK-NEXT: }

int main () {
  clad::differentiate(test_1, 0);
  clad::differentiate(test_2, 0);
  clad::differentiate(test_3, 0); //expected-error {{Invalid argument index '0' of '0' argument(s)}}
  clad::differentiate(test_4, 0);
  clad::differentiate(test_5, 0);
  clad::differentiate(test_6, "x");
  clad::differentiate(test_7, "i");
  clad::differentiate(test_8, "x");
  clad::differentiate<clad::opts::enable_tbr>(test_8); // expected-error {{TBR analysis is not meant for forward mode AD.}}
  clad::differentiate<clad::opts::enable_tbr, clad::opts::disable_tbr>(test_8); // expected-error {{Both enable and disable TBR options are specified.}}
  clad::differentiate<clad::opts::diagonal_only>(test_8); // expected-error {{Diagonal only option is only valid for Hessian mode.}}
  clad::differentiate(test_9);
  clad::differentiate(test_10);
  return 0;

// CHECK: void increment_pushforward(int &i, int &_d_i) {
// CHECK-NEXT: ++i;
// CHECK-NEXT: }

// CHECK: clad::ValueAndPushforward<double, double> func_with_enum_pushforward(double x, E e, double _d_x) {
// CHECK-NEXT: return {x * x, _d_x * x + x * _d_x};
// CHECK-NEXT: }

// CHECK: static clad::ValueAndPushforward<double, double> static_method_pushforward(double x, double _d_x) {
// CHECK-NEXT: return {x, _d_x};
// CHECK-NEXT: }
}
