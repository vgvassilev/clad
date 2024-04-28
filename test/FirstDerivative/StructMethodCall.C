// RUN: %cladclang %s -I%S/../../include -oStructMethodCall.out 2>&1 | %filecheck %s
// RUN: ./StructMethodCall.out | %filecheck_exec %s
//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

extern "C" int printf(const char* fmt, ...);

struct A {
public:
  int f(int x) {
    return x;
  }

  // CHECK: int f_darg0(int x) {
  // CHECK-NEXT:     int _d_x = 1;
  // CHECK-NEXT:     A _d_this_obj;
  // CHECK-NEXT:     A *_d_this = &_d_this_obj;
  // CHECK-NEXT:     return _d_x;
  // CHECK-NEXT: }

  int g_1(int x, int y) {
    return x*x + y;
  }

  // CHECK: int g_1_darg0(int x, int y) {
  // CHECK-NEXT:     int _d_x = 1;
  // CHECK-NEXT:     int _d_y = 0;
  // CHECK-NEXT:     A _d_this_obj;
  // CHECK-NEXT:     A *_d_this = &_d_this_obj;
  // CHECK-NEXT:     return _d_x * x + x * _d_x + _d_y;
  // CHECK-NEXT: }

  // CHECK: int g_1_darg1(int x, int y) {
  // CHECK-NEXT:     int _d_x = 0;
  // CHECK-NEXT:     int _d_y = 1;
  // CHECK-NEXT:     A _d_this_obj;
  // CHECK-NEXT:     A *_d_this = &_d_this_obj;
  // CHECK-NEXT:     return _d_x * x + x * _d_x + _d_y;
  // CHECK-NEXT: }

  int g_2(int x, int y) {
    return x + y*y;
  }

  // CHECK: int g_2_darg0(int x, int y) {
  // CHECK-NEXT:     int _d_x = 1;
  // CHECK-NEXT:     int _d_y = 0;
  // CHECK-NEXT:     A _d_this_obj;
  // CHECK-NEXT:     A *_d_this = &_d_this_obj;
  // CHECK-NEXT:     return _d_x + _d_y * y + y * _d_y;
  // CHECK-NEXT: }

  // CHECK: int g_2_darg1(int x, int y) {
  // CHECK-NEXT:     int _d_x = 0;
  // CHECK-NEXT:     int _d_y = 1;
  // CHECK-NEXT:     A _d_this_obj;
  // CHECK-NEXT:     A *_d_this = &_d_this_obj;
  // CHECK-NEXT:     return _d_x + _d_y * y + y * _d_y;
  // CHECK-NEXT: }


  int m(int x, int y) {
    return f(x) + g_1(x, y);
  }

  // CHECK: clad::ValueAndPushforward<int, int> f_pushforward(int x, A *_d_this, int _d_x);

  // CHECK: clad::ValueAndPushforward<int, int> g_1_pushforward(int x, int y, A *_d_this, int _d_x, int _d_y);

  // CHECK: int m_darg0(int x, int y) {
  // CHECK-NEXT:     int _d_x = 1;
  // CHECK-NEXT:     int _d_y = 0;
  // CHECK-NEXT:     A _d_this_obj;
  // CHECK-NEXT:     A *_d_this = &_d_this_obj;
  // CHECK-NEXT:     clad::ValueAndPushforward<int, int> _t0 = this->f_pushforward(x, _d_this, _d_x);
  // CHECK-NEXT:     clad::ValueAndPushforward<int, int> _t1 = this->g_1_pushforward(x, y, _d_this, _d_x, _d_y);
  // CHECK-NEXT:     return _t0.pushforward + _t1.pushforward;
  // CHECK-NEXT: }

  // CHECK: int m_darg1(int x, int y) {
  // CHECK-NEXT:     int _d_x = 0;
  // CHECK-NEXT:     int _d_y = 1;
  // CHECK-NEXT:     A _d_this_obj;
  // CHECK-NEXT:     A *_d_this = &_d_this_obj;
  // CHECK-NEXT:     clad::ValueAndPushforward<int, int> _t0 = this->f_pushforward(x, _d_this, _d_x);
  // CHECK-NEXT:     clad::ValueAndPushforward<int, int> _t1 = this->g_1_pushforward(x, y, _d_this, _d_x, _d_y);
  // CHECK-NEXT:     return _t0.pushforward + _t1.pushforward;
  // CHECK-NEXT: }
};

int main () {
  A a;
  auto f_dx = clad::differentiate(&A::f, 0);
  printf("Result is = %d\n", f_dx.execute(a, 1)); // CHECK-EXEC: Result is = 1

  auto g_1_dx = clad::differentiate(&A::g_1, 0);
  auto g_1_dy = clad::differentiate(&A::g_1, 1);
  printf("Result is = {%d, %d}\n", g_1_dx.execute(a, 1, 2), g_1_dy.execute(a, 1, 2)); // CHECK-EXEC: Result is = {2, 1}

  auto g_2_dx = clad::differentiate(&A::g_2, 0);
  auto g_2_dy = clad::differentiate(&A::g_2, 1);
  printf("Result is = {%d, %d}\n", g_2_dx.execute(a, 1, 2), g_2_dy.execute(a, 1, 2)); // CHECK-EXEC: Result is = {1, 4}

  auto m_dx = clad::differentiate(&A::m, 0);
  auto m_dy = clad::differentiate(&A::m, 1);
  printf("Result is = {%d, %d}\n", m_dx.execute(a, 1, 2), m_dy.execute(a, 1, 2)); // CHECK-EXEC: Result is = {3, 1}
  return 0;

  // CHECK: clad::ValueAndPushforward<int, int> f_pushforward(int x, A *_d_this, int _d_x) {
  // CHECK-NEXT:     return {x, _d_x};
  // CHECK-NEXT: }

  // CHECK: clad::ValueAndPushforward<int, int> g_1_pushforward(int x, int y, A *_d_this, int _d_x, int _d_y) {
  // CHECK-NEXT:     return {x * x + y, _d_x * x + x * _d_x + _d_y};
  // CHECK-NEXT: }
}
