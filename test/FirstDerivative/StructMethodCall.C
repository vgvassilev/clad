// RUN: %cladclang %s -I%S/../../include -oStructMethodCall.out 2>&1 | FileCheck %s
// RUN: ./StructMethodCall.out | FileCheck -check-prefix=CHECK-EXEC %s
//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

extern "C" int printf(const char* fmt, ...);

struct A {
public:
  float f(float x) {
    return x;
  }

  // CHECK: float f_darg0(float x) {
  // CHECK-NEXT:     float _d_x = 1;
  // CHECK-NEXT:     A _d_this_obj;
  // CHECK-NEXT:     A *_d_this = &_d_this_obj;
  // CHECK-NEXT:     return _d_x;
  // CHECK-NEXT: }

  float g_1(float x, float y) {
    return x*x + y;
  }

  // CHECK: float g_1_darg0(float x, float y) {
  // CHECK-NEXT:     float _d_x = 1;
  // CHECK-NEXT:     float _d_y = 0;
  // CHECK-NEXT:     A _d_this_obj;
  // CHECK-NEXT:     A *_d_this = &_d_this_obj;
  // CHECK-NEXT:     return _d_x * x + x * _d_x + _d_y;
  // CHECK-NEXT: }

  // CHECK: float g_1_darg1(float x, float y) {
  // CHECK-NEXT:     float _d_x = 0;
  // CHECK-NEXT:     float _d_y = 1;
  // CHECK-NEXT:     A _d_this_obj;
  // CHECK-NEXT:     A *_d_this = &_d_this_obj;
  // CHECK-NEXT:     return _d_x * x + x * _d_x + _d_y;
  // CHECK-NEXT: }

  float g_2(float x, float y) {
    return x + y*y;
  }

  // CHECK: float g_2_darg0(float x, float y) {
  // CHECK-NEXT:     float _d_x = 1;
  // CHECK-NEXT:     float _d_y = 0;
  // CHECK-NEXT:     A _d_this_obj;
  // CHECK-NEXT:     A *_d_this = &_d_this_obj;
  // CHECK-NEXT:     return _d_x + _d_y * y + y * _d_y;
  // CHECK-NEXT: }

  // CHECK: float g_2_darg1(float x, float y) {
  // CHECK-NEXT:     float _d_x = 0;
  // CHECK-NEXT:     float _d_y = 1;
  // CHECK-NEXT:     A _d_this_obj;
  // CHECK-NEXT:     A *_d_this = &_d_this_obj;
  // CHECK-NEXT:     return _d_x + _d_y * y + y * _d_y;
  // CHECK-NEXT: }


  float m(float x, float y) {
    return f(x) + g_1(x, y);
  }

  // CHECK: clad::ValueAndPushforward<float, float> f_pushforward(float x, A *_d_this, float _d_x);

  // CHECK: clad::ValueAndPushforward<float, float> g_1_pushforward(float x, float y, A *_d_this, float _d_x, float _d_y);

  // CHECK: float m_darg0(float x, float y) {
  // CHECK-NEXT:     float _d_x = 1;
  // CHECK-NEXT:     float _d_y = 0;
  // CHECK-NEXT:     A _d_this_obj;
  // CHECK-NEXT:     A *_d_this = &_d_this_obj;
  // CHECK-NEXT:     clad::ValueAndPushforward<float, float> _t0 = this->f_pushforward(x, _d_this, _d_x);
  // CHECK-NEXT:     clad::ValueAndPushforward<float, float> _t1 = this->g_1_pushforward(x, y, _d_this, _d_x, _d_y);
  // CHECK-NEXT:     return _t0.pushforward + _t1.pushforward;
  // CHECK-NEXT: }

  // CHECK: float m_darg1(float x, float y) {
  // CHECK-NEXT:     float _d_x = 0;
  // CHECK-NEXT:     float _d_y = 1;
  // CHECK-NEXT:     A _d_this_obj;
  // CHECK-NEXT:     A *_d_this = &_d_this_obj;
  // CHECK-NEXT:     clad::ValueAndPushforward<float, float> _t0 = this->f_pushforward(x, _d_this, _d_x);
  // CHECK-NEXT:     clad::ValueAndPushforward<float, float> _t1 = this->g_1_pushforward(x, y, _d_this, _d_x, _d_y);
  // CHECK-NEXT:     return _t0.pushforward + _t1.pushforward;
  // CHECK-NEXT: }
};

int main () {
  A a;
  auto f_dx = clad::differentiate(&A::f, 0);
  printf("Result is = %.2f\n", f_dx.execute(a, 1)); // CHECK-EXEC: Result is = 1.00

  auto g_1_dx = clad::differentiate(&A::g_1, 0);
  auto g_1_dy = clad::differentiate(&A::g_1, 1);
  printf("Result is = {%.2f, %.2f}\n", g_1_dx.execute(a, 1, 2), g_1_dy.execute(a, 1, 2)); // CHECK-EXEC: Result is = {2.00, 1.00}

  auto g_2_dx = clad::differentiate(&A::g_2, 0);
  auto g_2_dy = clad::differentiate(&A::g_2, 1);
  printf("Result is = {%.2f, %.2f}\n", g_2_dx.execute(a, 1, 2), g_2_dy.execute(a, 1, 2)); // CHECK-EXEC: Result is = {1.00, 4.00}

  auto m_dx = clad::differentiate(&A::m, 0);
  auto m_dy = clad::differentiate(&A::m, 1);
  printf("Result is = {%.2f, %.2f}\n", m_dx.execute(a, 1, 2), m_dy.execute(a, 1, 2)); // CHECK-EXEC: Result is = {3.00, 1.00}
  return 0;

  // CHECK: clad::ValueAndPushforward<float, float> f_pushforward(float x, A *_d_this, float _d_x) {
  // CHECK-NEXT:     return {x, _d_x};
  // CHECK-NEXT: }

  // CHECK: clad::ValueAndPushforward<float, float> g_1_pushforward(float x, float y, A *_d_this, float _d_x, float _d_y) {
  // CHECK-NEXT:     return {x * x + y, _d_x * x + x * _d_x + _d_y};
  // CHECK-NEXT: }
}
