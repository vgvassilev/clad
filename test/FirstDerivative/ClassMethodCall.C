// RUN: %cladclang %s -I%S/../../include -oClassMethods.out 2>&1 | FileCheck %s
// RUN: ./ClassMethods.out | FileCheck -check-prefix=CHECK-EXEC %s
// Fails on clang-18 due to https://github.com/llvm/llvm-project/issues/87151
// XFAIL: clang-18
//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

extern "C" int printf(const char* fmt, ...);

class A {
public:
  virtual ~A() {}
  A() {}

   __attribute__((always_inline)) int f(int x) {
    return x;
  }

  //CHECK:{{[__attribute__((always_inline)) ]*}}int f_darg0(int x){{[ __attribute__((always_inline))]*}} {
  //CHECK-NEXT:       int _d_x = 1;
  //CHECK-NEXT:       A _d_this_obj;
  //CHECK-NEXT:       A *_d_this = &_d_this_obj;
  //CHECK-NEXT:       return _d_x;
  //CHECK-NEXT:   }


  int g_1(int x, int y) {
    return x*x + y;
  }

  //CHECK:   int g_1_darg0(int x, int y) {
  //CHECK-NEXT:       int _d_x = 1;
  //CHECK-NEXT:       int _d_y = 0;
  //CHECK-NEXT:       A _d_this_obj;
  //CHECK-NEXT:       A *_d_this = &_d_this_obj;
  //CHECK-NEXT:       return _d_x * x + x * _d_x + _d_y;
  //CHECK-NEXT:   }


  //CHECK:   int g_1_darg1(int x, int y) {
  //CHECK-NEXT:       int _d_x = 0;
  //CHECK-NEXT:       int _d_y = 1;
  //CHECK-NEXT:       A _d_this_obj;
  //CHECK-NEXT:       A *_d_this = &_d_this_obj;
  //CHECK-NEXT:       return _d_x * x + x * _d_x + _d_y;
  //CHECK-NEXT:   }

  int g_2(int x, int y) {
    return x + y*y;
  }

  //CHECK:   int g_2_darg0(int x, int y) {
  //CHECK-NEXT:       int _d_x = 1;
  //CHECK-NEXT:       int _d_y = 0;
  //CHECK-NEXT:       A _d_this_obj;
  //CHECK-NEXT:       A *_d_this = &_d_this_obj;
  //CHECK-NEXT:       return _d_x + _d_y * y + y * _d_y;
  //CHECK-NEXT:   }

  //CHECK:   int g_2_darg1(int x, int y) {
  //CHECK-NEXT:       int _d_x = 0;
  //CHECK-NEXT:       int _d_y = 1;
  //CHECK-NEXT:       A _d_this_obj;
  //CHECK-NEXT:       A *_d_this = &_d_this_obj;
  //CHECK-NEXT:       return _d_x + _d_y * y + y * _d_y;
  //CHECK-NEXT:   }

  int m(int x, int y) {
    return f(x) + g_1(x, y);
  }

  virtual float vm(float x, float y) {
    return x + y;
  }

  float vm_darg0(float x, float y);
  //CHECK:   float vm_darg0(float x, float y) {
  //CHECK-NEXT:       float _d_x = 1;
  //CHECK-NEXT:       float _d_y = 0;
  //CHECK-NEXT:       A _d_this_obj;
  //CHECK-NEXT:       A *_d_this = &_d_this_obj;
  //CHECK-NEXT:       return _d_x + _d_y;
  //CHECK-NEXT:   }

};

class B : public A {
public:
  B() {}
  virtual ~B() {}
  float vm(float x, float y) override {
    return x*x + y*y;
  }

  float vm_darg0(float x, float y);
  //CHECK:   float vm_darg0(float x, float y) override {
  //CHECK-NEXT:       float _d_x = 1;
  //CHECK-NEXT:       float _d_y = 0;
  //CHECK-NEXT:       B _d_this_obj;
  //CHECK-NEXT:       B *_d_this = &_d_this_obj;
  //CHECK-NEXT:       return _d_x * x + x * _d_x + _d_y * y + y * _d_y;
  //CHECK-NEXT:   }


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
  printf("%s\n", vm_darg0_B.getCode());
  //CHECK-EXEC:   float vm_darg0(float x, float y) override {
  //CHECK-EXEC-NEXT:       float _d_x = 1;
  //CHECK-EXEC-NEXT:       float _d_y = 0;
  //CHECK-EXEC-NEXT:       B _d_this_obj;
  //CHECK-EXEC-NEXT:       B *_d_this = &_d_this_obj;
  //CHECK-EXEC-NEXT:       return _d_x * x + x * _d_x + _d_y * y + y * _d_y;
  //CHECK-EXEC-NEXT:   }

  return 0;
}
