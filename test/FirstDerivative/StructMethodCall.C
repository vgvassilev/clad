// RUN: %cladclang %s -I%S/../../include 
//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

extern "C" int printf(const char* fmt, ...);

struct A {
public:
  int f(int x) {
    return x;
  }

  // CHECK: int f_darg0(int x) {
  // CHECK-NEXT: int _d_x = 1;
  // CHECK-NEXT:     A _d_this_obj;
  // CHECK-NEXT:     A *_d_this = &_d_this_obj;
  // CHECK-NEXT: return _d_x;
  // CHECK-NEXT: }

  int g_1(int x, int y) {
    return x*x + y;
  }

  // CHECK: int g_1_darg0(int x, int y) {
  // CHECK-NEXT: int _d_x = 1;
  // CHECK-NEXT: int _d_y = 0;
  // CHECK-NEXT:     A _d_this_obj;
  // CHECK-NEXT:     A *_d_this = &_d_this_obj;
  // CHECK-NEXT: return _d_x * x + x * _d_x + _d_y;
  // CHECK-NEXT: }

  // CHECK: int g_1_darg1(int x, int y) {
  // CHECK-NEXT: int _d_x = 0;
  // CHECK-NEXT: int _d_y = 1;
  // CHECK-NEXT:     A _d_this_obj;
  // CHECK-NEXT:     A *_d_this = &_d_this_obj;
  // CHECK-NEXT: return _d_x * x + x * _d_x + _d_y;
  // CHECK-NEXT: }

  int g_2(int x, int y) {
    return x + y*y;
  }

  // CHECK: int g_2_darg0(int x, int y) {
  // CHECK-NEXT: int _d_x = 1;
  // CHECK-NEXT: int _d_y = 0;
  // CHECK-NEXT:     A _d_this_obj;
  // CHECK-NEXT:     A *_d_this = &_d_this_obj;
  // CHECK-NEXT: return _d_x + _d_y * y + y * _d_y;
  // CHECK-NEXT: }

  // CHECK: int g_2_darg1(int x, int y) {
  // CHECK-NEXT: int _d_x = 0;
  // CHECK-NEXT: int _d_y = 1;
  // CHECK-NEXT:     A _d_this_obj;
  // CHECK-NEXT:     A *_d_this = &_d_this_obj;
  // CHECK-NEXT: return _d_x + _d_y * y + y * _d_y;
  // CHECK-NEXT: }


  int m(int x, int y) {
    return f(x) + g_1(x, y);
  }
};

int main () {
  A a;
  clad::differentiate(&A::f, 0);
  clad::differentiate(&A::g_1, 0);
  clad::differentiate(&A::g_1, 1);
  clad::differentiate(&A::g_2, 0);
  clad::differentiate(&A::g_2, 1);
  //clad::differentiate(&A::m, 0);
  //clad::differentiate(&A::m, 1);
  return 0;
}
