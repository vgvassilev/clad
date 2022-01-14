// RUN: %cladclang %s -I%S/../../include -oFunctionCallsWithResults.out 2>&1 | FileCheck %s
// RUN: ./FunctionCallsWithResults.out | FileCheck -check-prefix=CHECK-EXEC %s

#include "clad/Differentiator/Differentiator.h"

int printf(const char* fmt, ...);

namespace custom_derivatives {
  float custom_fn_darg0(int x) {
    return x;
  }

  float custom_fn_darg0(float x) {
    return x * x;
  }

  int custom_fn_darg0() {
    return 5;
  }

  float overloaded_darg0(float x) {
    printf("A was called.\n");
    return x*x;
  }

  float overloaded_darg0() {
    int x = 2;
    printf("A was called.\n");
    return x*x;
  }
}

float custom_fn(float x) {
  return x;
}

int custom_fn(int x) {
  return x;
}

float overloaded(float x) {
  return x;
}

float overloaded() {
  return 3;
}

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

float test_4(float x) {
  return overloaded();
}

// CHECK: float test_4_darg0(float x) {
// CHECK-NEXT: float _d_x = 1;
// CHECK-NEXT: return custom_derivatives::overloaded_darg0();
// CHECK-NEXT: }

float test_1_darg0(float x);
float test_2_darg0(float x);
float test_4_darg0(float x);

int main () {
  clad::differentiate(test_1, 0);
  printf("Result is = %f\n", test_1_darg0(1.1)); // CHECK-EXEC: Result is = 2

  clad::differentiate(test_2, 0);
  printf("Result is = %f\n", test_2_darg0(1.0)); // CHECK-EXEC: Result is = 2

  clad::differentiate(test_4, 0);
  printf("Result is = %f\n", test_4_darg0(1.0)); // CHECK-EXEC: Result is = 4

  return 0;
}
