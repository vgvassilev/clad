// RUN: %cladclang %s -I%S/../../include 2>&1 | FileCheck %s
//CHECK-NOT: {{.*error|warning|note:.*}}
//XFAIL:*
#include "clad/Differentiator/Differentiator.h"

extern "C" int printf(const char* fmt, ...);

class A {
public:
  int f(int x) {
    return x;
  }

  // CHECK: int f_dx(int x) {
  // CHECK-NEXT: return 1;
  // CHECK-NEXT: }

  int g_1(int x, int y) {
    return x*x + y;
  }

  // CHECK: int g_1_dx(int x, int y) {
  // CHECK-NEXT: return (1 * x + x * 1) + (0);
  // CHECK-NEXT: }

  // CHECK: int g_1_dy(int x, int y) {
  // CHECK-NEXT: return (0 * x + x * 0) + (1);
  // CHECK-NEXT: }

  int g_2(int x, int y) {
    return x + y*y;
  }

  // CHECK: int g_2_dx(int x, int y) {
  // CHECK-NEXT: return 1 + ((0 * y + y * 0));
  // CHECK-NEXT: }

  // CHECK: int g_2_dy(int x, int y) {
  // CHECK-NEXT: return 0 + ((1 * y + y * 1));
  // CHECK-NEXT: }


  int m(int x, int y) {
    return f(x) + g_1(x, y);
  }

  virtual float vm(float x, float y) {
    return x + y;
  }

  // CHECK: virtual float vm_dx(float x, float y) {
  // CHECK-NEXT: return 1.F;
  // CHECK-NEXT: }

};

class B : public A {
public:
  float vm(float x, float y) override {
    return x*x + y*y;
  }

  // CHECK: float vm_dx(float x, float y) override {
  // CHECK-NEXT: return 2.F * x;
  // CHECK-NEXT: }

}

int main () {
  A a;
  clad::differentiate(&A::f, 0);
  clad::differentiate(&A::g_1, 0);
  clad::differentiate(&A::g_1, 1);
  clad::differentiate(&A::g_1, 0);
  clad::differentiate(&A::g_2, 1);
  //clad::differentiate(&A::m, 0);
  //clad::differentiate(&A::m, 1);
  clad::differentiate(&A::vm, 0);
  printf("Result is = %f\n", a.vm_dx(2,3)); // CHECK-EXEC: Result is = 2.F
  clad::differentiate(&B::vm, 0);
  printf("Result is = %f\n", a.vm_dx(2,3)); // CHECK-EXEC: Result is = 4.F
  return 0;
}
