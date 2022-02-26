// RUN: %cladnumdiffclang %s -I%S/../../include -fsyntax-only -Xclang -verify 2>&1 | FileCheck %s

//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

#include <math.h>

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
// CHECK-NEXT: clad::ValueAndPushforward<float, float> _t0 = clad::custom_derivatives::overloaded_pushforward(x, _d_x);
// CHECK-NEXT: clad::ValueAndPushforward<float, float> _t1 = clad::custom_derivatives::custom_fn_pushforward(x, _d_x);
// CHECK-NEXT: return _t0.pushforward + _t1.pushforward;
// CHECK-NEXT: }

float test_2(float x) {
  return overloaded(x) + custom_fn(x);
}

// CHECK: float test_2_darg0(float x) {
// CHECK-NEXT: float _d_x = 1;
// CHECK-NEXT: clad::ValueAndPushforward<float, float> _t0 = clad::custom_derivatives::overloaded_pushforward(x, _d_x);
// CHECK-NEXT: clad::ValueAndPushforward<float, float> _t1 = clad::custom_derivatives::custom_fn_pushforward(x, _d_x);
// CHECK-NEXT: return _t0.pushforward + _t1.pushforward;
// CHECK-NEXT: }

float test_3() {
  return custom_fn();
}

// CHECK-NOT: float test_3_darg0() {

float test_4(int x) {
  return overloaded();
}

// CHECK: clad::ValueAndPushforward<int, int> overloaded_pushforward() {
// CHECK-NEXT:     return {3, 0};
// CHECK-NEXT: }

// CHECK: float test_4_darg0(int x) {
// CHECK-NEXT: int _d_x = 1;
// CHECK-NEXT: clad::ValueAndPushforward<int, int> _t0 = overloaded_pushforward();
// CHECK-NEXT: return _t0.pushforward;
// CHECK-NEXT: }

float test_5(int x) {
  return no_body(x);
}

// CHECK: float test_5_darg0(int x) {
// CHECK-NEXT: int _d_x = 1;
// CHECK-NEXT: clad::ValueAndPushforward<float, float> _t0 = clad::custom_derivatives::no_body_pushforward(x, _d_x);
// CHECK-NEXT: return _t0.pushforward;
// CHECK-NEXT: }

int main () {
  clad::differentiate(test_1, 0);
  clad::differentiate(test_2, 0);
  clad::differentiate(test_3, 0); //expected-error {{Invalid argument index '0' of '0' argument(s)}}
  clad::differentiate(test_4, 0);
  clad::differentiate(test_5, 0);
  return 0;
}
