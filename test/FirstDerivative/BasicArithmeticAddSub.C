// RUN: %cladclang %s -I%S/../../include -oBasicArithmeticAddSub.out 2>&1 | FileCheck %s
// RUN: ./BasicArithmeticAddSub.out | FileCheck -check-prefix=CHECK-EXEC %s

//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

extern "C" int printf(const char* fmt, ...);

float a_1(float x) {
  float y = 4;
  return y + y; // == 0
}
// CHECK: float a_1_darg0(float x) {
// CHECK-NEXT: float _d_x = 1;
// CHECK-NEXT: float _d_y = 0;
// CHECK-NEXT: float y = 4;
// CHECK-NEXT: return _d_y + _d_y;
// CHECK-NEXT: }

float a_2(float x) {
  return 1 + 1; // == 0
}
// CHECK: float a_2_darg0(float x) {
// CHECK-NEXT: float _d_x = 1;
// CHECK-NEXT: return 0 + 0;
// CHECK-NEXT: }

float a_3(float x) {
  return x + x; // == 2
}
// CHECK: float a_3_darg0(float x) {
// CHECK-NEXT: float _d_x = 1;
// CHECK-NEXT: return _d_x + _d_x;
// CHECK-NEXT: }

float a_4(float x) {
  float y = 4;
  return x + y + x + 3 + x; // == 3x
}
// CHECK: float a_4_darg0(float x) {
// CHECK-NEXT: float _d_x = 1;
// CHECK-NEXT: float _d_y = 0;
// CHECK-NEXT: float y = 4;
// CHECK-NEXT: return _d_x + _d_y + _d_x + 0 + _d_x;
// CHECK-NEXT: }

float s_1(float x) {
  float y = 4;
  return y - y; // == 0
}
// CHECK: float s_1_darg0(float x) {
// CHECK-NEXT: float _d_x = 1;
// CHECK-NEXT: float _d_y = 0;
// CHECK-NEXT: float y = 4;
// CHECK-NEXT: return _d_y - _d_y;
// CHECK-NEXT: }

float s_2(float x) {
  return 1 - 1; // == 0
}
// CHECK: float s_2_darg0(float x) {
// CHECK-NEXT: float _d_x = 1;
// CHECK-NEXT: return 0 - 0;
// CHECK-NEXT: }

float s_3(float x) {
  return x - x; // == 0
}
// CHECK: float s_3_darg0(float x) {
// CHECK-NEXT: float _d_x = 1;
// CHECK-NEXT: return _d_x - _d_x;
// CHECK-NEXT: }

float s_4(float x) {
  float y = 4;
  return x - y - x - 3 - x; // == -1
}
// CHECK: float s_4_darg0(float x) {
// CHECK-NEXT: float _d_x = 1;
// CHECK-NEXT: float _d_y = 0;
// CHECK-NEXT: float y = 4;
// CHECK-NEXT: return _d_x - _d_y - _d_x - 0 - _d_x;
// CHECK-NEXT: }

float as_1(float x) {
  float y = 4;
  return x + x - x + y - y + 3 - 3; // == 1
}
// CHECK: float as_1_darg0(float x) {
// CHECK-NEXT: float _d_x = 1;
// CHECK-NEXT: float _d_y = 0;
// CHECK-NEXT: float y = 4;
// CHECK-NEXT: return _d_x + _d_x - _d_x + _d_y - _d_y + 0 - 0;
// CHECK-NEXT: }

float IntegerLiteralToFloatLiteral(float x, float y) {
  return x * x - y;
}
// CHECK: float IntegerLiteralToFloatLiteral_darg0(float x, float y) {
// CHECK-NEXT: float _d_x = 1;
// CHECK-NEXT: float _d_y = 0;
// CHECK-NEXT: return _d_x * x + x * _d_x - _d_y;
// CHECK-NEXT: }

float a_1_darg0(float x);
float a_2_darg0(float x);
float a_3_darg0(float x);
float a_4_darg0(float x);
float s_1_darg0(float x);
float s_2_darg0(float x);
float s_3_darg0(float x);
float s_4_darg0(float x);
float as_1_darg0(float x);
float IntegerLiteralToFloatLiteral_darg0(float x, float y);

int main () { // expected-no-diagnostics
  float x = 4;
  clad::differentiate(a_1, 0);
  printf("Result is = %f\n", a_1_darg0(1)); // CHECK-EXEC: Result is = 0

  clad::differentiate(a_2, 0);
  printf("Result is = %f\n", a_2_darg0(1)); // CHECK-EXEC: Result is = 0

  clad::differentiate(a_3, 0);
  printf("Result is = %f\n", a_3_darg0(1)); // CHECK-EXEC: Result is = 2

  clad::differentiate(a_4, 0);
  printf("Result is = %f\n", a_4_darg0(1)); // CHECK-EXEC: Result is = 3

  clad::differentiate(s_1, 0);
  printf("Result is = %f\n", s_1_darg0(1)); // CHECK-EXEC: Result is = 0

  clad::differentiate(s_2, 0);
  printf("Result is = %f\n", s_2_darg0(1)); // CHECK-EXEC: Result is = 0

  clad::differentiate(s_3, 0);
  printf("Result is = %f\n", s_3_darg0(1)); // CHECK-EXEC: Result is = 0

  clad::differentiate(s_4, 0);
  printf("Result is = %f\n", s_4_darg0(1)); // CHECK-EXEC: Result is = -1

  clad::differentiate(as_1, 0);
  printf("Result is = %f\n", as_1_darg0(1)); // CHECK-EXEC: Result is = 1

  clad::differentiate(IntegerLiteralToFloatLiteral, 0);
  printf("Result is = %f\n", IntegerLiteralToFloatLiteral_darg0(5., 0.)); // CHECK-EXEC: Result is = 10

  return 0;
}
