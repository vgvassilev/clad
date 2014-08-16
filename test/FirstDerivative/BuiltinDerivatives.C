// RUN: %cladclang %s -I%S/../../include -Xclang -verify -oBuiltinDerivatives.out -lm 2>&1 | FileCheck %s
// RUN: ./BuiltinDerivatives.out | FileCheck -check-prefix=CHECK-EXEC %s

//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

extern "C" int printf(const char* fmt, ...);

float f1(float x) {
  return sin(x);
}

// CHECK: float f1_dx(float x) {
// CHECK-NEXT: return sin_dx(x) * (1.F);
// CHECK-NEXT: }

float f2(float x) {
  return cos(x);
}

// CHECK: float f2_dx(float x) {
// CHECK-NEXT: cos_dx(x) * (1.F);
// CHECK-NEXT: }

float f3(float x, float y) {
  return sin(x) + sin(y);
}

// CHECK: float f3_dx(float x, float y) {
// CHECK-NEXT: return sin_dx(x) * (1.F) + (sin_dx(y) * (0.F));
// CHECK-NEXT: }

// CHECK: float f3_dy(float x, float y) {
// CHECK-NEXT: return sin_dy(x) * (0.F) + (sin_dy(y) * (1.F));
// CHECK-NEXT: }

float f4(float x, float y) {
  return sin(x * x) + sin(y * y);
}

// CHECK: float f4_dx(float x, float y) {
// CHECK-NEXT: return sin_dx(x * x) * ((1.F * x + x * 1.F)) + (sin_dx(y * y) * ((0.F * y + y * 0.F)));
// CHECK-NEXT: }

// CHECK: float f4_dy(float x, float y) {
// CHECK-NEXT: return sin_dy(x * x) * ((0.F * x + x * 0.F)) + (sin_dy(y * y) * ((1.F * y + y * 1.F)));
// CHECK-NEXT: }

int main () { //expected-no-diagnostics
  auto f1_dx = clad::differentiate(f1, 0);
  printf("Result is = %f\n", f1_dx.execute(60)); // CHECK-EXEC: Result is = -0.952413

  auto f2_dx = clad::differentiate(f2, 0);
  printf("Result is = %f\n", f2_dx.execute(60)); //CHECK-EXEC: Result is = 0.304811

  auto f3_dx = clad::differentiate(f3, 0);
  printf("Result is = %f\n", f3_dx.execute(60, 30)); //CHECK-EXEC: Result is = -0.952413

  auto f3_dy = clad::differentiate(f3, 1);
  printf("Result is = %f\n", f3_dy.execute(60, 30)); //CHECK-EXEC: Result is = 0.154251

  auto f4_dx = clad::differentiate(f4, 0);
  printf("Result is = %f\n", f4_dx.execute(60, 30)); //CHECK-EXEC: Result is = 115.805412

  auto f4_dy = clad::differentiate(f4, 1);
  printf("Result is = %f\n", f4_dy.execute(60, 30)); //CHECK-EXEC: Result is = 3.974802

  return 0;
}
