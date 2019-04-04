// RUN: %cladclang %s -I%S/../../include -oBuiltinDerivatives.out -lm 2>&1 | FileCheck %s
// RUN: ./BuiltinDerivatives.out | FileCheck -check-prefix=CHECK-EXEC %s

//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

extern "C" int printf(const char* fmt, ...);

float f1(float x) {
  return sin(x);
}

// CHECK: float f1_darg0(float x) {
// CHECK-NEXT: float _d_x = 1;
// CHECK-NEXT: return custom_derivatives::sin_darg0(x) * _d_x;
// CHECK-NEXT: }

float f2(float x) {
  return cos(x);
}

// CHECK: float f2_darg0(float x) {
// CHECK-NEXT: float _d_x = 1;
// CHECK-NEXT: custom_derivatives::cos_darg0(x) * _d_x;
// CHECK-NEXT: }

float f3(float x, float y) {
  return sin(x) + sin(y);
}

// CHECK: float f3_darg0(float x, float y) {
// CHECK-NEXT: float _d_x = 1;
// CHECK-NEXT: float _d_y = 0;
// CHECK-NEXT: return custom_derivatives::sin_darg0(x) * _d_x + custom_derivatives::sin_darg0(y) * _d_y;
// CHECK-NEXT: }

// CHECK: float f3_darg1(float x, float y) {
// CHECK-NEXT: float _d_x = 0;
// CHECK-NEXT: float _d_y = 1;
// CHECK-NEXT: return custom_derivatives::sin_darg0(x) * _d_x + custom_derivatives::sin_darg0(y) * _d_y;
// CHECK-NEXT: }

float f4(float x, float y) {
  return sin(x * x) + sin(y * y);
}

// CHECK: float f4_darg0(float x, float y) {
// CHECK-NEXT: float _d_x = 1;
// CHECK-NEXT: float _d_y = 0;
// CHECK-NEXT: return custom_derivatives::sin_darg0(x * x) * (_d_x * x + x * _d_x) + custom_derivatives::sin_darg0(y * y) * (_d_y * y + y * _d_y);
// CHECK-NEXT: }

// CHECK: float f4_darg1(float x, float y) {
// CHECK-NEXT: float _d_x = 0;
// CHECK-NEXT: float _d_y = 1;
// CHECK-NEXT: return custom_derivatives::sin_darg0(x * x) * (_d_x * x + x * _d_x) + custom_derivatives::sin_darg0(y * y) * (_d_y * y + y * _d_y);
// CHECK-NEXT: }

float f5(float x) {
  return exp(x);
}

// CHECK: float f5_darg0(float x) {
// CHECK-NEXT: float _d_x = 1;
// CHECK-NEXT: return custom_derivatives::exp_darg0(x) * _d_x;
// CHECK-NEXT: }

float f6(float x) {
  return exp(x * x);
}

// CHECK: float f6_darg0(float x) {
// CHECK-NEXT: float _d_x = 1;
// CHECK-NEXT: return custom_derivatives::exp_darg0(x * x) * (_d_x * x + x * _d_x);
// CHECK-NEXT: }


int main () { //expected-no-diagnostics
  auto f1_darg0 = clad::differentiate(f1, 0);
  printf("Result is = %f\n", f1_darg0.execute(60)); // CHECK-EXEC: Result is = -0.952413

  auto f2_darg0 = clad::differentiate(f2, 0);
  printf("Result is = %f\n", f2_darg0.execute(60)); //CHECK-EXEC: Result is = 0.304811

  auto f3_darg0 = clad::differentiate(f3, 0);
  printf("Result is = %f\n", f3_darg0.execute(60, 30)); //CHECK-EXEC: Result is = -0.952413

  auto f3_darg1 = clad::differentiate(f3, 1);
  printf("Result is = %f\n", f3_darg1.execute(60, 30)); //CHECK-EXEC: Result is = 0.154251

  auto f4_darg0 = clad::differentiate(f4, 0);
  printf("Result is = %f\n", f4_darg0.execute(60, 30)); //CHECK-EXEC: Result is = 115.805412

  auto f4_darg1 = clad::differentiate(f4, 1);
  printf("Result is = %f\n", f4_darg1.execute(60, 30)); //CHECK-EXEC: Result is = 3.974802

  auto f5_darg0 = clad::differentiate(f5, 0);
  printf("Result is = %f\n", f5_darg0.execute(3)); //CHECK-EXEC: Result is = 20.085537

  auto f6_darg0 = clad::differentiate(f6, 0);
  printf("Result is = %f\n", f6_darg0.execute(3)); //CHECK-EXEC: Result is = 48618.503906

  return 0;
}
