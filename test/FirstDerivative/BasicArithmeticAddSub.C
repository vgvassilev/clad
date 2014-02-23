// RUN: %clad %s -I%S/../../include -fsyntax-only 2>&1 | FileCheck %s

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

int main () {
  int x = 4;
  diff(a_1, 1);
  diff(a_2, 1);
  diff(a_3, 1);
  diff(a_4, 1);
  
  diff(s_1, 1);
  diff(s_2, 1);
  diff(s_3, 1);
  diff(s_4, 1);
  
  diff(as_1, 1);
  return 0;
}
