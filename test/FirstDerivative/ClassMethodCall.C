// RUN: %cladclang %s -lm -I%S/../../include -lstdc++ -oClassMethods.out 2>&1 | FileCheck %s
// RUN: ./ClassMethods.out | FileCheck -check-prefix=CHECK-EXEC %s
//XFAIL:*
//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

extern "C" int printf(const char* fmt, ...);

class A {
public:
  virtual ~A() {}
  A() {}

  int f(int x) {
    return x;
  }

  // CHECK: int f_darg0(int x) {
  // CHECK-NEXT: return 1;
  // CHECK-NEXT: }

  int g_1(int x, int y) {
    return x*x + y;
  }

  // CHECK: int g_1_darg0(int x, int y) {
  // CHECK-NEXT: return (1 * x + x * 1) + (0);
  // CHECK-NEXT: }

  // CHECK: int g_1_darg1(int x, int y) {
  // CHECK-NEXT: return (0 * x + x * 0) + (1);
  // CHECK-NEXT: }

  int g_2(int x, int y) {
    return x + y*y;
  }

  // CHECK: int g_2_darg0(int x, int y) {
  // CHECK-NEXT: return 1 + ((0 * y + y * 0));
  // CHECK-NEXT: }

  // CHECK: int g_2_darg1(int x, int y) {
  // CHECK-NEXT: return 0 + ((1 * y + y * 1));
  // CHECK-NEXT: }


  int m(int x, int y) {
    return f(x) + g_1(x, y);
  }

  virtual float vm(float x, float y) {
    return x + y;
  }

  // CHECK: float vm_darg0(float x, float y) {
  // CHECK-NEXT: return 1.F + (0.F);
  // CHECK-NEXT: }

};

class B : public A {
public:
  B() {}
  virtual ~B() {}
  float vm(float x, float y) override {
    return x*x + y*y;
  }

  // CHECK: float vm_darg0(float x, float y) {
  // CHECK-NEXT: return (1.F * x + x * 1.F) + ((0.F * y + y * 0.F));
  // CHECK-NEXT: }

};

int main () {
  A a;
  B b;
  clad::differentiate(&A::f, 0);
  clad::differentiate(&A::g_1, 0);
  clad::differentiate(&A::g_1, 1);
  clad::differentiate(&A::g_2, 0);
  clad::differentiate(&A::g_2, 1);
  // clad::differentiate(&A::m, 0);
  // clad::differentiate(&A::m, 1);
  auto vm_darg0_A = clad::differentiate(&A::vm, 0);
  printf("Result is = %f\n", vm_darg0_A.execute(a, 2, 3)); // CHECK-EXEC: Result is = 1.0000
  auto vm_darg0_B = clad::differentiate(&B::vm, 0);
  printf("Result is = %f\n", vm_darg0_B.execute(b, 2, 3)); // CHECK-EXEC: Result is = 4.0000
  return 0;
}
