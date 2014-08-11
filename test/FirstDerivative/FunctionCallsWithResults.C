// RUN: %cladclang %s -I%S/../../include -oFunctionCallsWithResults.out 2>&1 | FileCheck %s
// RUN: ./FunctionCallsWithResults.out | FileCheck -check-prefix=CHECK-EXEC %s

#include "clad/Differentiator/Differentiator.h"

int printf(const char* fmt, ...);

namespace custom_derivatives {
  float custom_fn_dx(int x) {
    return x;
  }

  float custom_fn_dx(float x) {
    return x * x;
  }

  int custom_fn_dx() {
    return 5;
  }

  float overloaded_dx(float x) {
    printf("A was called.\n");
    return x*x;
  }

  float overloaded_dx() {
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

// CHECK: float test_1_dx(float x) {
// CHECK-NEXT: return overloaded_dx(x) * (1.F) + (custom_fn_dx(x) * (1.F));
// CHECK-NEXT: }

float test_2(float x) {
  return overloaded(x) + custom_fn(x);
}

// CHECK: float test_2_dx(float x) {
// CHECK-NEXT: return overloaded_dx(x) * (1.F) + (custom_fn_dx(x) * (1.F));
// CHECK-NEXT: }

float test_4(float x) {
  return overloaded();
}

// CHECK: float test_4_dx(float x) {
// CHECK-NEXT: return overloaded_dx();
// CHECK-NEXT: }

float test_1_dx(float x);
float test_2_dx(float x);
float test_4_dx(float x);

int main () {
  clad::differentiate(test_1, 0);
  printf("Result is = %f\n", test_1_dx(1.1)); // CHECK-EXEC: Result is = 2

  clad::differentiate(test_2, 0);
  printf("Result is = %f\n", test_2_dx(1.0)); // CHECK-EXEC: Result is = 2

  clad::differentiate(test_4, 0);
  printf("Result is = %f\n", test_4_dx(1.0)); // CHECK-EXEC: Result is = 4

  return 0;
}
