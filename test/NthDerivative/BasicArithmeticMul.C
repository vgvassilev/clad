// RUN: %cladclang %s -I%S/../../include -oBasicArithmeticMul2.out 2>&1 | FileCheck %s
// RUN: ./BasicArithmeticMul2.out | FileCheck -check-prefix=CHECK-EXEC %s
#include "clad/Differentiator/Differentiator.h"
#include "clad/Differentiator/BuiltinDerivatives.h"

extern "C" int printf(const char* fmt, ...);


float test_2(float x, float y) {
  return x * x + y * y;
}

// CHECK: float test_2_darg0(float x, float y) {
// CHECK-NEXT: return 1.F * x + x * 1.F + 0.F * y + y * 0.F;
// CHECK-NEXT: }

// CHECK: float test_2_d2arg0(float x, float y) {
// CHECK-NEXT: return 0.F * x + 1.F * 1.F + 1.F * 1.F + x * 0.F + 0.F * y + 0.F * 0.F + 0.F * 0.F + y * 0.F;
// CHECK-NEXT: }

// CHECK: float test_2_darg1(float x, float y) {
// CHECK-NEXT: return 0.F * x + x * 0.F + 1.F * y + y * 1.F;
// CHECK-NEXT: }

// CHECK: float test_2_d2arg1(float x, float y) {
// CHECK-NEXT: return 0.F * x + 0.F * 0.F + 0.F * 0.F + x * 0.F + 0.F * y + 1.F * 1.F + 1.F * 1.F + y * 0.F;
// CHECK-NEXT: }

// CHECK: float test_2_d3arg1(float x, float y) {
// CHECK-NEXT: return 0.F * x + 0.F * 0.F + 0.F * 0.F + 0.F * 0.F + 0.F * 0.F + 0.F * 0.F + 0.F * 0.F + x * 0.F + 0.F * y + 0.F * 1.F + 0.F * 1.F + 1.F * 0.F + 0.F * 1.F + 1.F * 0.F + 1.F * 0.F + y * 0.F;
// CHECK-NEXT: }

float test_1_darg0(float x);
float test_1_d2arg0(float x);

float test_2_darg0(float x, float y);
float test_2_d2arg0(float x, float y);
float test_2_darg1(float x, float y);
float test_2_d2arg1(float x, float y);
float test_2_d3arg1(float x, float y);

int main () {
  clad::differentiate<2>(test_2, 0);
  printf("Result is = %f\n", test_2_d2arg0(1.5, 2.5)); // CHECK-EXEC: Result is = 2.000000

  clad::differentiate<3>(test_2, 1);
  printf("Result is = %f\n", test_2_d3arg1(1.5, 2.5)); // CHECK-EXEC: Result is = 0.000000
  return 0;
}
