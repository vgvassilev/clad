// RUN: %cladclang %s -I%S/../../include -oBasicArithmeticAddSub.out -Xclang -verify 2>&1 | FileCheck %s
// RUN: ./BasicArithmeticAddSub.out | FileCheck -check-prefix=CHECK-EXEC %s

#include "clad/Differentiator/Differentiator.h"

extern "C" int printf(const char* fmt, ...);

int a_1(int x) {
  int y = 4;
  return y + y; // == 0
}
// CHECK: int a_1_derived_x(int x) {
// CHECK-NEXT: int y = 4;
// CHECK-NEXT: return 0 + (0);
// CHECK-NEXT: }

int a_2(int x) {
  return 1 + 1; // == 0
}
// CHECK: int a_2_derived_x(int x) {
// CHECK-NEXT: return 0 + (0);
// CHECK-NEXT: }

int a_3(int x) {
  return x + x; // == 2
}
// CHECK: int a_3_derived_x(int x) {
// CHECK-NEXT: return 1 + (1);
// CHECK-NEXT: }

int a_4(int x) {
  int y = 4;
  return x + y + x + 3 + x; // == 3
}
// CHECK: int a_4_derived_x(int x) {
// CHECK-NEXT: int y = 4;
// CHECK-NEXT: return 1 + (0) + (1) + (0) + (1);
// CHECK-NEXT: }

int s_1(int x) {
  int y = 4;
  return y - y; // == 0
}
// CHECK: int s_1_derived_x(int x) {
// CHECK-NEXT: int y = 4;
// CHECK-NEXT: return 0 - (0);
// CHECK-NEXT: }

int s_2(int x) {
  return 1 - 1; // == 0
}
// CHECK: int s_2_derived_x(int x) {
// CHECK-NEXT: return 0 - (0);
// CHECK-NEXT: }

int s_3(int x) {
  return x - x; // == 0
}
// CHECK: int s_3_derived_x(int x) {
// CHECK-NEXT: return 1 - (1);
// CHECK-NEXT: }

int s_4(int x) {
  int y = 4;
  return x - y - x - 3 - x; // == -1
}
// CHECK: int s_4_derived_x(int x) {
// CHECK-NEXT: int y = 4;
// CHECK-NEXT: return 1 - (0) - (1) - (0) - (1);
// CHECK-NEXT: }

int as_1(int x) {
  int y = 4;
  return x + x - x + y - y + 3 - 3; // == 1
}
// CHECK: int as_1_derived_x(int x) {
// CHECK-NEXT: int y = 4;
// CHECK-NEXT: return 1 + (1) - (1) + (0) - (0) + (0) - (0);
// CHECK-NEXT: }

int a_1_derived_x(int x);
int a_2_derived_x(int x);
int a_3_derived_x(int x);
int a_4_derived_x(int x);
int s_1_derived_x(int x);
int s_2_derived_x(int x);
int s_3_derived_x(int x);
int s_4_derived_x(int x);
int as_1_derived_x(int x);

int main () { // expected-no-diagnostics
  int x = 4;
  diff(a_1, 1);
  printf("Result is = %d\n", a_1_derived_x(1)); // CHECK-EXEC: Result is = 0

  diff(a_2, 1);
  printf("Result is = %d\n", a_2_derived_x(1)); // CHECK-EXEC: Result is = 0

  diff(a_3, 1);
  printf("Result is = %d\n", a_3_derived_x(1)); // CHECK-EXEC: Result is = 2

  diff(a_4, 1);
  printf("Result is = %d\n", a_4_derived_x(1)); // CHECK-EXEC: Result is = 3

  diff(s_1, 1);
  printf("Result is = %d\n", s_1_derived_x(1)); // CHECK-EXEC: Result is = 0

  diff(s_2, 1);
  printf("Result is = %d\n", s_2_derived_x(1)); // CHECK-EXEC: Result is = 0

  diff(s_3, 1);
  printf("Result is = %d\n", s_3_derived_x(1)); // CHECK-EXEC: Result is = 0

  diff(s_4, 1);
  printf("Result is = %d\n", s_4_derived_x(1)); // CHECK-EXEC: Result is = -1

  diff(as_1, 1);
  printf("Result is = %d\n", as_1_derived_x(1)); // CHECK-EXEC: Result is = 1

  return 0;
}
