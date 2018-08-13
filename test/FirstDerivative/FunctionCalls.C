// RUN: %cladclang %s -I%S/../../include -fsyntax-only -Xclang -verify 2>&1 | FileCheck %s

//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

int printf(const char* fmt, ...);
int no_body(int x);
int custom_fn(int x);
int custom_fn(float x);
int custom_fn();

namespace custom_derivatives {
  float overloaded_darg0(float x) {
    return x;
  }

  float overloaded_darg0(int x) {
    return x;
  }

  float no_body_darg0(float x) {
    return 1;
  }

  float custom_fn_darg0(int x) {
    return x + x;
  }

  float custom_fn_darg0(float x) {
    return x * x;
  }

  int custom_fn_darg0() {
    return 5;
  }
}

int overloaded(int x) {
  printf("A was called.\n");
  return x*x;
}

int overloaded(float x) {
  return x;
}

int overloaded() {
  return 3;
} // expected-warning {{function 'overloaded' was not differentiated because it is not declared in namespace 'custom_derivatives'}}

float test_1(float x) {
  return overloaded(x) + custom_fn(x);
}

// CHECK: float test_1_darg0(float x) {
// CHECK-NEXT: float _d_x = 1;
// CHECK-NEXT: return custom_derivatives::overloaded_darg0(x) * _d_x + custom_derivatives::custom_fn_darg0(x) * _d_x;
// CHECK-NEXT: }

float test_2(float x) {
  return overloaded(x) + custom_fn(x);
}

// CHECK: float test_2_darg0(float x) {
// CHECK-NEXT: float _d_x = 1;
// CHECK-NEXT: return custom_derivatives::overloaded_darg0(x) * _d_x + custom_derivatives::custom_fn_darg0(x) * _d_x;
// CHECK-NEXT: }

float test_3() {
  return custom_fn();
}

// CHECK-NOT: float test_3_darg0() {

float test_4(int x) {
  return overloaded();
}

// CHECK: float test_4_darg0(int x) {
// CHECK-NEXT: int _d_x = 1;
// CHECK-NEXT: return 0;
// CHECK-NEXT: }

float test_5(int x) {
  return no_body(x);
}

// CHECK: float test_5_darg0(int x) {
// CHECK-NEXT: int _d_x = 1;
// CHECK-NEXT: return custom_derivatives::no_body_darg0(x) * _d_x;
// CHECK-NEXT: }

int main () {
  clad::differentiate(test_1, 0);
  clad::differentiate(test_2, 0);
  clad::differentiate(test_3, 0); //expected-error {{Trying to differentiate function 'test_3' taking no arguments}}
  clad::differentiate(test_4, 0);
  clad::differentiate(test_5, 0);

  return 0;
}
