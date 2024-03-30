// RUN: %cladclang %s -I%S/../../include -oVirtualMethodsCall.out 2>&1 | FileCheck %s
// RUN: ./VirtualMethodsCall.out | FileCheck -check-prefix=CHECK-EXEC %s
// XFAIL: asserts
// Fails on clang-18 due to https://github.com/llvm/llvm-project/issues/87151
// XFAIL: clang-18
//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

extern "C" int printf(const char* fmt, ...);

// Classes: forward, hide, virtual and overriden methods

class A {
public:
  A() {}
  virtual ~A() {}

  // Hide/Reintroduce

  float f(float x) {
    return x;
  }

  // Virtual/Override

  virtual float vm(float x, float y) {
    return x + y;
  }

  virtual float vm1(float x, float y) {
    return x - y;
  }

  // Polimophism

  // TODO: Remove forward when VTable update is implemented
  virtual float vm_darg0(float x, float y); // forward
  virtual float vm_darg1(float x, float y); // forward

  virtual float m(float x, float y);

  virtual float ovr(float x, float y) {
    // TODO: Remove call forward when execute works with polymorphic methods
    auto vm_darg0_cf = clad::differentiate(&A::vm, 0);
// FIXME: We need to make this out-of-line
//CHECK: float vm_darg0(float x, float y) {
//CHECK-NEXT:       float _d_x = 1;
//CHECK-NEXT:       float _d_y = 0;
//CHECK-NEXT:      A _d_this_obj;
//CHECK-NEXT:      A *_d_this = &_d_this_obj;
//CHECK-NEXT:       return _d_x + _d_y;
//CHECK-NEXT: }
    auto vm_darg1_cf = clad::differentiate(&A::vm, 1);
//CHECK:   float vm_darg1(float x, float y) {
//CHECK-NEXT:       float _d_x = 0;
//CHECK-NEXT:       float _d_y = 1;
//CHECK-NEXT:       A _d_this_obj;
//CHECK-NEXT:       A *_d_this = &_d_this_obj;
//CHECK-NEXT:       return _d_x + _d_y;
//CHECK-NEXT: }
    return vm_darg0(x, y) + vm_darg1(x, y);
    //auto vm_darg0_cf = clad::differentiate(vm, 0);
    //auto vm_darg1_cf = clad::differentiate(vm, 1);
    //return vm_darg0_cf.execute(*this, x, y) + vm_darg1_cf.execute(*this, x, y);
  }
};

float A::m(float x, float y) {
  // TODO: Remove call forward when execute works with polymorphic methods
  return vm_darg0(x, y) + vm_darg1(x, y);
  //auto vm_darg0_cf = clad::differentiate(vm, 0);
  //auto vm_darg1_cf = clad::differentiate(vm, 1);
  //return vm_darg0_cf.execute(*this, x, y) + vm_darg1_cf.execute(*this, x, y);
}

class B : public A {
public:
  B() {}
  virtual ~B() {}

  // Hide/Reintroduce

  float f(float x) {
    return x*x;
  }

  // Virtual/Override

  float vm(float x, float y) override {
    return x*x + y*y;
  }

  float vm1(float x, float y) override {
    return x*x - y*y;
  }

  // Polimophism

  // TODO: Remove forward when VTable update is implemented
  float vm_darg0(float x, float y) override; // forward
  float vm_darg1(float x, float y) override; // forward

/*
  // Inherited from A:
  float m(float x, float y) override {
    // ...
  }
*/

  float ovr(float x, float y) override {
    // TODO: Remove call forward when execute works with polymorphic methods
    return vm_darg0(x+1, y+1) + 7.0;
    //auto vm_darg0_cf = clad::differentiate(/*B::*/vm, 0);
    //return vm_darg0_cf.execute(*this, x, y) + 7.0;
  }
};

class B1 : public A {
public:
  B1() {}
  virtual ~B1() {}

  // Hide/Reintroduce

  float f(float x) {
    return x*x*x;
  }

  // Virtual/Override

  float vm(float x, float y) override {
    return x*y + x*y;
  }

  // Inherited from A
  // float vm1(float x, float y)...

  // Polimophism

  // Inherited from A:
  //float m(float x, float y) ...
};

int main () {
  A a;
  B b;
  B1 b1;

  //
  printf("Result is = %f\n", a.f(2.0)); // CHECK-EXEC: Result is = 2.0000
  printf("Result is = %f\n", b.f(2.0)); // CHECK-EXEC: Result is = 4.0000
  printf("Result is = %f\n", b1.f(2.0)); // CHECK-EXEC: Result is = 8.0000

  printf("---\n"); // CHECK-EXEC: ---

  auto f_darg0_A = clad::differentiate(&A::f, 0);
//CHECK: float f_darg0(float x) {
//CHECK-NEXT:     float _d_x = 1;
// CHECK-NEXT:     A _d_this_obj;
// CHECK-NEXT:     A *_d_this = &_d_this_obj;
//CHECK-NEXT:     return _d_x;
//CHECK-NEXT: }
  auto f_darg0_B = clad::differentiate(&B::f, 0);
//CHECK: float f_darg0(float x) {
//CHECK-NEXT:     float _d_x = 1;
// CHECK-NEXT:     B _d_this_obj;
// CHECK-NEXT:     B *_d_this = &_d_this_obj;
//CHECK-NEXT:     return _d_x * x + x * _d_x;
//CHECK-NEXT: }
  auto f_darg0_B1 = clad::differentiate(&B1::f, 0);
//CHECK: float f_darg0(float x) {
//CHECK-NEXT:     float _d_x = 1;
// CHECK-NEXT:     B1 _d_this_obj;
// CHECK-NEXT:     B1 *_d_this = &_d_this_obj;
//CHECK-NEXT:     float _t0 = x * x;
//CHECK-NEXT:     return (_d_x * x + x * _d_x) * x + _t0 * _d_x;
//CHECK-NEXT: }
  printf("Result is = %f\n", f_darg0_A.execute(a, 2.0)); // CHECK-EXEC: Result is = 1.0000
  printf("Result is = %f\n", f_darg0_B.execute(b, 2.0)); // CHECK-EXEC: Result is = 4.0000
  printf("Result is = %f\n", f_darg0_B1.execute(b1, 2.0)); // CHECK-EXEC: Result is = 12.0000

  printf("---\n"); // CHECK-EXEC: ---

  auto vm_darg0_A = clad::differentiate((float(A::*)(float,float))&A::vm, 0);
//CHECK: float vm_darg0(float x, float y) override {
//CHECK-NEXT:     float _d_x = 1;
//CHECK-NEXT:     float _d_y = 0;
// CHECK-NEXT:     B _d_this_obj;
// CHECK-NEXT:     B *_d_this = &_d_this_obj;
//CHECK-NEXT:     return _d_x * x + x * _d_x + _d_y * y + y * _d_y;
//CHECK-NEXT: }
  auto vm_darg0_B = clad::differentiate((float(B::*)(float,float))&B::vm, 0);
//CHECK: float vm_darg1(float x, float y) override {
//CHECK-NEXT:     float _d_x = 0;
//CHECK-NEXT:     float _d_y = 1;
// CHECK-NEXT:     B _d_this_obj;
// CHECK-NEXT:     B *_d_this = &_d_this_obj;
//CHECK-NEXT:     return _d_x * x + x * _d_x + _d_y * y + y * _d_y;
//CHECK-NEXT: }
  auto vm_darg1_A = clad::differentiate((float(A::*)(float,float))&A::vm, 1);
//
  auto vm_darg1_B = clad::differentiate((float(B::*)(float,float))&B::vm, 1);
//
  auto vm1_darg0_A = clad::differentiate((float(A::*)(float,float))&A::vm1, 0);
//CHECK: float vm1_darg0(float x, float y) {
//CHECK-NEXT:     float _d_x = 1;
//CHECK-NEXT:     float _d_y = 0;
// CHECK-NEXT:     A _d_this_obj;
// CHECK-NEXT:     A *_d_this = &_d_this_obj;
//CHECK-NEXT:     return _d_x - _d_y;
//CHECK-NEXT: }
  auto vm1_darg0_B = clad::differentiate((float(B::*)(float,float))&B::vm1, 0);
//CHECK: float vm1_darg0(float x, float y) override {
//CHECK-NEXT:    float _d_x = 1;
//CHECK-NEXT:     float _d_y = 0;
// CHECK-NEXT:     B _d_this_obj;
// CHECK-NEXT:     B *_d_this = &_d_this_obj;
//CHECK-NEXT:     return _d_x * x + x * _d_x - (_d_y * y + y * _d_y);
//CHECK-NEXT: }
  auto vm1_darg1_A = clad::differentiate((float(A::*)(float,float))&A::vm1, 1);
//CHECK: float vm1_darg1(float x, float y) {
//CHECK-NEXT:     float _d_x = 0;
//CHECK-NEXT:     float _d_y = 1;
// CHECK-NEXT:     A _d_this_obj;
// CHECK-NEXT:     A *_d_this = &_d_this_obj;
//CHECK-NEXT:     return _d_x - _d_y;
//CHECK-NEXT: }
  auto vm1_darg1_B = clad::differentiate((float(B::*)(float,float))&B::vm1, 1);
//CHECK: float vm1_darg1(float x, float y) override {
//CHECK-NEXT:     float _d_x = 0;
//CHECK-NEXT:     float _d_y = 1;
// CHECK-NEXT:     B _d_this_obj;
// CHECK-NEXT:     B *_d_this = &_d_this_obj;
//CHECK-NEXT:     return _d_x * x + x * _d_x - (_d_y * y + y * _d_y);
//CHECK-NEXT: }
  printf("Result is = %f\n", vm_darg0_A.execute(a, 2.0f, 3.0f)); // CHECK-EXEC: Result is = 1.0000
  printf("Result is = %f\n", vm_darg0_B.execute(b, 2.0f, 3.0f)); // CHECK-EXEC: Result is = 4.0000
  printf("Result is = %f\n", vm_darg1_A.execute(a, 2.0f, 3.0f)); // CHECK-EXEC: Result is = 1.0000
  printf("Result is = %f\n", vm_darg1_B.execute(b, 2.0f, 3.0f)); // CHECK-EXEC: Result is = 6.0000
  printf("Result is = %f\n", vm1_darg0_A.execute(a, 2.0f, 3.0f)); // CHECK-EXEC: Result is = 1.0000
  printf("Result is = %f\n", vm1_darg0_B.execute(b, 2.0f, 3.0f)); // CHECK-EXEC: Result is = 4.0000
  printf("Result is = %f\n", vm1_darg1_A.execute(a, 2.0f, 3.0f)); // CHECK-EXEC: Result is = -1.0000
  printf("Result is = %f\n", vm1_darg1_B.execute(b, 2.0f, 3.0f)); // CHECK-EXEC: Result is = -6.0000

  printf("---\n"); // CHECK-EXEC: ---

  printf("Result is = %f\n", a.m(3, 4)); // CHECK-EXEC: Result is = 2.0000
  printf("Result is = %f\n", b.m(3, 4)); // CHECK-EXEC: Result is = 14.0000
  printf("Result is = %f\n", b1.m(3, 4)); // CHECK-EXEC: Result is = 14.0000

  printf("---\n"); // CHECK-EXEC: ---

  printf("Result is = %f\n", a.ovr(3, 4)); // CHECK-EXEC: Result is = 2.0000
  printf("Result is = %f\n", b.ovr(3, 4)); // CHECK-EXEC: Result is = 15.0000
  printf("Result is = %f\n", b1.ovr(3, 4)); // CHECK-EXEC: Result is = 14.0000

  printf("---\n"); // CHECK-EXEC: ---

  auto vm_darg0_B1 = clad::differentiate(&B1::vm, 0);
//CHECK: float vm_darg0(float x, float y) override {
//CHECK-NEXT:     float _d_x = 1;
//CHECK-NEXT:     float _d_y = 0;
// CHECK-NEXT:     B1 _d_this_obj;
// CHECK-NEXT:     B1 *_d_this = &_d_this_obj;
//CHECK-NEXT:     return _d_x * y + x * _d_y + _d_x * y + x * _d_y;
//CHECK-NEXT: }
  auto vm_darg1_B1 = clad::differentiate(&B1::vm, 1);
//CHECK: float vm_darg1(float x, float y) override {
//CHECK-NEXT:     float _d_x = 0;
//CHECK-NEXT:     float _d_y = 1;
// CHECK-NEXT:     B1 _d_this_obj;
// CHECK-NEXT:     B1 *_d_this = &_d_this_obj;
//CHECK-NEXT:     return _d_x * y + x * _d_y + _d_x * y + x * _d_y;
//CHECK-NEXT: }
  printf("Result is = %f\n", b1.m(3, 4)); // CHECK-EXEC: Result is = 14.0000
  printf("Result is = %f\n", b1.ovr(3, 4)); // CHECK-EXEC: Result is = 14.0000

  printf("---\n"); // CHECK-EXEC: ---

  A *obj;
  obj = &a;
  printf("Result is = %f\n", obj->ovr(3, 4)); // CHECK-EXEC: Result is = 2.0000
  obj = &b;
  printf("Result is = %f\n", obj->ovr(3, 4)); // CHECK-EXEC: Result is = 15.0000
  obj = &b1;
  printf("Result is = %f\n", obj->ovr(3, 4)); // CHECK-EXEC: Result is = 14.0000

  return 0;
}
