// RUN: %cladclang %s -I%S/../../include -oOverloads.out 2>&1 | %filecheck %s
// RUN: ./Overloads.out | %filecheck_exec %s
// XFAIL: asserts
//CHECK-NOT: {{.*error|warning|note:.*}}
// XFAIL: target={{i586.*}}

#include "clad/Differentiator/Differentiator.h"

extern "C" int printf(const char* fmt, ...);

// Overload

float func(float x) {
  return x;
}

double func(double x) {
  return x+x;
}

// Classes: forward, hide, overload, virtual and overriden methods

class A {
public:
  float f1(float x) {
    return x+x+x;
  }

  double f1(double x) {
    return x+x+x+x;
  }
};

class B : public A {
public:
  float f1(float x) {
    return x+x+x+x+x;
  }

  int f1(int x) {
    return x;
  }
};

class B1 : public A {
public:
  double f1(double x) {
    return x+x+x+x+x+x;
  }
};
int main () {
  A a;
  B b;
  B1 b1;

  // Function overloads

  auto func_darg0_float = clad::differentiate(static_cast<float(*)(float)>(&func), 0);
//CHECK: float func_darg0(float x) {
//CHECK-NEXT:    float _d_x = 1;
//CHECK-NEXT:    return _d_x;
//CHECK-NEXT: }
  auto func_darg0_double = clad::differentiate((double(*)(double))&func, 0);
//CHECK: double func_darg0(double x) {
//CHECK-NEXT:    double _d_x = 1;
//CHECK-NEXT:    return _d_x + _d_x;
//CHECK-NEXT: }
  printf("Result is = %f\n", func_darg0_float.execute(2.0)); // CHECK-EXEC: Result is = 1.0000
  printf("Result is = %f\n", func_darg0_double.execute(2.0)); // CHECK-EXEC: Result is = 2.0000

  printf("---\n"); // CHECK-EXEC: ---

  // Method overloads

  auto f1_darg0_float_A = clad::differentiate((float(A::*)(float))&A::f1, 0);
//CHECK: float f1_darg0(float x) {
//CHECK-NEXT:     float _d_x = 1;
// CHECK-NEXT:     A _d_this_obj;
// CHECK-NEXT:     A *_d_this = &_d_this_obj;
//CHECK-NEXT:     return _d_x + _d_x + _d_x;
//CHECK-NEXT: }
  auto f1_darg0_double_A = clad::differentiate((double(A::*)(double))&A::f1, 0);
//CHECK: double f1_darg0(double x) {
//CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     A _d_this_obj;
// CHECK-NEXT:     A *_d_this = &_d_this_obj;
//CHECK-NEXT:     return _d_x + _d_x + _d_x + _d_x;
//CHECK-NEXT: }
  auto f1_darg0_float_B = clad::differentiate((float(B::*)(float))&B::f1, 0);
//CHECK: float f1_darg0(float x) {
//CHECK-NEXT:     float _d_x = 1;
// CHECK-NEXT:     B _d_this_obj;
// CHECK-NEXT:     B *_d_this = &_d_this_obj;
//CHECK-NEXT:     return _d_x + _d_x + _d_x + _d_x + _d_x;
//CHECK-NEXT: }
  auto f1_darg0_int_B = clad::differentiate((int(B::*)(int))&B::f1, 0);
//CHECK: int f1_darg0(int x) {
//CHECK-NEXT:     int _d_x = 1;
// CHECK-NEXT:     B _d_this_obj;
// CHECK-NEXT:     B *_d_this = &_d_this_obj;
//CHECK-NEXT:     return _d_x;
//CHECK-NEXT: }
  // resolve to float(B::*)(float)
  //auto f1_darg0_double_B = clad::differentiate((double(B::*)(double))&B::f1, 0); // ?EXPECTED-ERROR {{address of overloaded function 'f1' does not match required type 'double (double)'}}
  // resolve to double(B1::*)(double)
  auto f1_darg0_float_B1 = clad::differentiate((float(B1::*)(float))&B1::f1, 0);
  //
  auto f1_darg0_double_B1 = clad::differentiate((double(B1::*)(double))&B1::f1, 0);
//CHECK: double f1_darg0(double x) {
//CHECK-NEXT:     double _d_x = 1;
//CHECK-NEXT:     B1 _d_this_obj;
//CHECK-NEXT:     B1 *_d_this = &_d_this_obj;
//CHECK-NEXT:     return _d_x + _d_x + _d_x + _d_x + _d_x + _d_x;
//CHECK-NEXT: }
  printf("Result is = %f\n", f1_darg0_float_A.execute(a, 2.0)); // CHECK-EXEC: Result is = 3.0000
  printf("Result is = %f\n", f1_darg0_double_A.execute(a, 2.0)); // CHECK-EXEC: Result is = 4.0000
  printf("Result is = %f\n", f1_darg0_float_B.execute(b, 2.0)); // CHECK-EXEC: Result is = 5.0000
  printf("Result is = %i\n", f1_darg0_int_B.execute(b, 2)); // CHECK-EXEC: Result is = 1
  //printf("Result is %s\n", f1_darg0_double_B.execute(b, 2.0)<1 ? "float" : "other"); // -CHECK-EXEC: Result is float
  printf("Result is %s\n", f1_darg0_float_B1.execute(b1, 2.0f)<1 ? "double" : "other"); // CHECK-EXEC: Result is double
  printf("Result is = %f\n", f1_darg0_double_B1.execute(b1, 2.0)); // CHECK-EXEC: Result is = 6.0000

  return 0;
}
