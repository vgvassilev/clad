// RUN: %cladclang %s -I%S/../../include -fsyntax-only 2>&1 | FileCheck %s

#include "clad/Differentiator/Differentiator.h"

extern "C" int printf(const char* fmt, ...);

int m_1(int x) {
  int y = 4;
  return y * y; // == 0
}
// CHECK: int m_1_derived_x(int x) {
// CHECK-NEXT: int y = 4;
// CHECK-NEXT: return (0 * y + y * 0);
// CHECK-NEXT: }

int m_2(int x) {
  return 1 * 1; // == 0
}
// CHECK: int m_2_derived_x(int x) {
// CHECK-NEXT: return (0 * 1 + 1 * 0);
// CHECK-NEXT: }

int m_3(int x) {
  return x * x; // == 2 * x
}
// CHECK: int m_3_derived_x(int x) {
// CHECK-NEXT: return (1 * x + x * 1);
// CHECK-NEXT: }

int m_4(int x) {
  int y = 4;
  return x * y * x * 3 * x; // == 9 * x * x * y
}
// CHECK: int m_4_derived_x(int x) {
// CHECK-NEXT: int y = 4;
// CHECK-NEXT: return ((((1 * y + x * 0) * x + x * y * 1) * 3 + x * y * x * 0) * x + x * y * x * 3 * 1);
// CHECK-NEXT: }

int d_1(int x) {
  int y = 4;
  return y / y; // == 0
}
// CHECK: int d_1_derived_x(int x) {
// CHECK-NEXT: int y = 4;
// CHECK-NEXT: return (0 * y - y * 0) / (y * y);
// CHECK-NEXT: }

int d_2(int x) {
  return 1 / 1; // == 0
}
// CHECK: int d_2_derived_x(int x) {
// CHECK-NEXT: return (0 * 1 - 1 * 0) / (1 * 1);
// CHECK-NEXT: }

int d_3(int x) {
  return x / x; // == 0
}
// CHECK: int d_3_derived_x(int x) {
// CHECK-NEXT: return (1 * x - x * 1) / (x * x);
// CHECK-NEXT: }

int d_4(int x) {
  int y = 4;
  return x / y / x / 3 / x; // == -1 / 3 / x / x / y
}
// CHECK: int d_4_derived_x(int x) {
// CHECK-NEXT: int y = 4;
// CHECK-NEXT: return ((((1 * y - x * 0) / (y * y) * x - x / y * 1) / (x * x) * 3 - x / y / x * 0) / (3 * 3) * x - x / y / x / 3 * 1) / (x * x);
// CHECK-NEXT: }

int md_1(int x) {
  int y = 4;
  return x * x / x * y / y * 3 / 3; // == 1
}
// CHECK: int md_1_derived_x(int x) {
// CHECK-NEXT: int y = 4;
// CHECK-NEXT: return ((((((1 * x + x * 1) * x - x * x * 1) / (x * x) * y + x * x / x * 0) * y - x * x / x * y * 0) / (y * y) * 3 + x * x / x * y / y * 0) * 3 - x * x / x * y / y * 3 * 0) / (3 * 3);
// CHECK-NEXT: }

int main () {
  int x = 4;
  diff(m_1, 1);
  diff(m_2, 1);
  diff(m_3, 1);
  diff(m_4, 1);
  
  diff(d_1, 1);
  diff(d_2, 1);
  diff(d_3, 1);
  diff(d_4, 1);
  
  diff(md_1, 1);
  return 0;
}
