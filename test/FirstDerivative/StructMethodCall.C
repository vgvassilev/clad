// RUN: %cladclang %s -I%S/../../include -fsyntax-only 2>&1

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

  // CHECK: float g_1_derived_x(int x, int y) {
  // CHECK-NEXT: return 2 * x + 0;
  // CHECK-NEXT: }

  // CHECK: float g_1_derived_y(int x, int y) {
  // CHECK-NEXT: return 0 + 1;
  // CHECK-NEXT: }

  int g_2(int x, int y) {
    return x + y*y;
  }

  // CHECK: float g_2_derived_x(int x, int y) {
  // CHECK-NEXT: return 1 + 0;
  // CHECK-NEXT: }

  // CHECK: float g_2_derived_y(int x, int y) {
  // CHECK-NEXT: return 0 + 2 * y;
  // CHECK-NEXT: }


  int m(int x, int y) {
    return f(x) + g_1(x, y);
  }

  // CHECK: float m_derived_x(int x, int y) {
  // CHECK-NEXT: return f_derived_x(int x) + g_1_derived_x(x, y);
  // CHECK-NEXT: }

  // CHECK: float m_derived_y(int x, int y) {
  // CHECK-NEXT: return 0 + g_1_derived_y(x, y);
  // CHECK-NEXT: }

};


int main () {
  A* a = new A();
  diff(&A::f, 1);
  diff(&A::g_1, 1);
  diff(&A::g_1, 2);
  diff(&A::g_1, 1);
  diff(&A::g_2, 2);
  diff(&A::m, 1);
  diff(&A::m, 2);
  return 0;
}
