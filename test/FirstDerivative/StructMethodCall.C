// RUN: %cladclang %s -I%S/../../include 2>&1 | FileCheck %s
//CHECK-NOT: {{.*error:.*}}
//XFAIL:*
#include "clad/Differentiator/Differentiator.h"

extern "C" int printf(const char* fmt, ...);

struct A {
public:
  int f(int x) {
    return x;
  }

  // CHECK: int f_derived_x(int x) {
  // CHECK-NEXT: return 1;
  // CHECK-NEXT: }

  int g_1(int x, int y) {
    return x*x + y;
  }

  // CHECK: int g_1_derived_x(int x, int y) {
  // CHECK-NEXT: return (1 * x + x * 1) + (0);
  // CHECK-NEXT: }

  // CHECK: int g_1_derived_y(int x, int y) {
  // CHECK-NEXT: return (0 * x + x * 0) + (1);
  // CHECK-NEXT: }

  int g_2(int x, int y) {
    return x + y*y;
  }

  // CHECK: int g_2_derived_x(int x, int y) {
  // CHECK-NEXT: return 1 + ((0 * y + y * 0));
  // CHECK-NEXT: }

  // CHECK: int g_2_derived_y(int x, int y) {
  // CHECK-NEXT: return 0 + ((1 * y + y * 1));
  // CHECK-NEXT: }


  int m(int x, int y) {
    return f(x) + g_1(x, y);
  }
};

int main () {
  A a;
  clad::differentiate(&A::f, 1);
  clad::differentiate(&A::g_1, 1);
  clad::differentiate(&A::g_1, 2);
  clad::differentiate(&A::g_2, 1);
  clad::differentiate(&A::g_2, 2);
  //clad::differentiate(&A::m, 1);
  //clad::differentiate(&A::m, 2);
  return 0;
}
