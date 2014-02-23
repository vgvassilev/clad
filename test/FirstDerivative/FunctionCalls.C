// RUN: %clad %s -I%S/../../include -fsyntax-only -Xclang -verify 2>&1 | FileCheck %s

#include "clad/Differentiator/Differentiator.h"

int printf(const char* fmt, ...); // expected-warning {{function printf was not differentiated because it is not declared in namespace custom_derivatives}}
int no_body(int x); // expected-error {{attempted differention of function no_body, which does not have a definition}}
int custom_fn(int x);
int custom_fn(float x);
int custom_fn();

namespace custom_derivatives {
  float overloaded(int x);
  float overloaded(float x);
  
  int no_body(int x);
  
  int custom_fn_derived_x(int x) {
    return x + x;
  }
  
  int custom_fn_derived_x(float x) {
    return x * x;
  }
  
  int custom_fn_derived_x() {
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
} // expected-warning {{function overloaded was not differentiated because it is not declared in namespace custom_derivatives}}

float test_1(int x) {
  return overloaded(x) + custom_fn(x);
}

// CHECK: float test_1_derived_x(int x) {
// CHECK-NEXT: return overloaded_derived_x(x) + (custom_fn_derived_x(x));
// CHECK-NEXT: }

float test_2(float x) {
  return overloaded(x) + custom_fn(x);
}

// CHECK: float test_2_derived_x(float x) {
// CHECK-NEXT: return overloaded_derived_x(x) + (custom_fn_derived_x(x));
// CHECK-NEXT: }

float test_3() {
  return custom_fn();
}

// CHECK: float test_3_derived_x() {
// CHECK-NEXT: return custom_fn_derived_x();
// CHECK-NEXT: }

float test_4() {
  return overloaded();
}

// CHECK: float test_4_derived_x() {
// CHECK-NEXT: return overloaded();
// CHECK-NEXT: }


float test_5(int x) {
  return no_body(x);
}

// CHECK: float test_5_derived_x(int x) {
// CHECK-NEXT: return no_body(x);
// CHECK-NEXT: }

int main () {
  int x = 4;
  diff(test_1, x);
  diff(test_2, x);
  diff(test_3, x);
  diff(test_4, x);
  diff(test_5, x);
  
  return 0;
}
