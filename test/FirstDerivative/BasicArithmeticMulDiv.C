// RUN: %cladclang %s -I%S/../../include -oBasicArithmeticMulDiv.out 2>&1 | FileCheck  %s
// RUN: ./BasicArithmeticMulDiv.out | FileCheck -check-prefix=CHECK-EXEC %s

//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

extern "C" int printf(const char* fmt, ...);

int m_1(int x) {
  int y = 4;
  return y * y; // == 0
}
// CHECK: int m_1_darg0(int x) {
// CHECK-NEXT: int y = 4;
// CHECK-NEXT: return (0 * y + y * 0);
// CHECK-NEXT: }

int m_2(int x) {
  return 1 * 1; // == 0
}
// CHECK: int m_2_darg0(int x) {
// CHECK-NEXT: return (0 * 1 + 1 * 0);
// CHECK-NEXT: }

int m_3(int x) {
  return x * x; // == 2 * x
}
// CHECK: int m_3_darg0(int x) {
// CHECK-NEXT: return (1 * x + x * 1);
// CHECK-NEXT: }

int m_4(int x) {
  int y = 4;
  return x * y * x * 3 * x; // == 9 * x * x * y
}
// CHECK: int m_4_darg0(int x) {
// CHECK-NEXT: int y = 4;
// CHECK-NEXT: return ((((1 * y + x * 0) * x + x * y * 1) * 3 + x * y * x * 0) * x + x * y * x * 3 * 1);
// CHECK-NEXT: }

double m_5(int x) {
  return 3.14 * x;
}
// CHECK: double m_5_darg0(int x) {
// CHECK-NEXT: return (0. * x + 3.1400000000000001 * 1);
// CHECK-NEXT: }

float m_6(int x) {
  return 3.f * x;
}
// CHECK: float m_6_darg0(int x) {
// CHECK-NEXT: return (0.F * x + 3.F * 1);
// CHECK-NEXT: }

int d_1(int x) {
  int y = 4;
  return y / y; // == 0
}
// CHECK: int d_1_darg0(int x) {
// CHECK-NEXT: int y = 4;
// CHECK-NEXT: return (0 * y - y * 0) / (y * y);
// CHECK-NEXT: }

int d_2(int x) {
  return 1 / 1; // == 0
}
// CHECK: int d_2_darg0(int x) {
// CHECK-NEXT: return (0 * 1 - 1 * 0) / (1 * 1);
// CHECK-NEXT: }

int d_3(int x) {
  return x / x; // == 0
}
// CHECK: int d_3_darg0(int x) {
// CHECK-NEXT: return (1 * x - x * 1) / (x * x);
// CHECK-NEXT: }

int d_4(int x) {
  int y = 4;
  return x / y / x / 3 / x; // == -1 / 3 / x / x / y
}
// CHECK: int d_4_darg0(int x) {
// CHECK-NEXT: int y = 4;
// CHECK-NEXT: return ((((1 * y - x * 0) / (y * y) * x - x / y * 1) / (x * x) * 3 - x / y / x * 0) / (3 * 3) * x - x / y / x / 3 * 1) / (x * x);
// CHECK-NEXT: }

double issue25(double x, double y) {
  x+=y;
  x-=y;
  x/=y;
  x*=y;

  return x;
}

// CHECK: double issue25_darg0(double x, double y) {
// CHECK-NEXT: x += 0.;
// CHECK-NEXT: x -= 0.;
// CHECK-NEXT: x /= 0.;
// CHECK-NEXT: x *= 0.;
// CHECK-NEXT: return 1.;
// CHECK-NEXT: }

int md_1(int x) {
  int y = 4;
  return x * x / x * y / y * 3 / 3; // == 1
}
// CHECK: int md_1_darg0(int x) {
// CHECK-NEXT: int y = 4;
// CHECK-NEXT: return ((((((1 * x + x * 1) * x - x * x * 1) / (x * x) * y + x * x / x * 0) * y - x * x / x * y * 0) / (y * y) * 3 + x * x / x * y / y * 0) * 3 - x * x / x * y / y * 3 * 0) / (3 * 3);
// CHECK-NEXT: }


int m_1_darg0(int x);
int m_2_darg0(int x);
int m_3_darg0(int x);
int m_4_darg0(int x);
double m_5_darg0(int x);
float m_6_darg0(int x);
int d_1_darg0(int x);
int d_2_darg0(int x);
int d_3_darg0(int x);
int d_4_darg0(int x);
double issue25_darg0(double x, double y);
int md_1_darg0(int x);

int main () {
  int x = 4;
  clad::differentiate(m_1, 0);
  printf("Result is = %d\n", m_1_darg0(1)); // CHECK-EXEC: Result is = 0

  clad::differentiate(m_2, 0);
  printf("Result is = %d\n", m_2_darg0(1)); // CHECK-EXEC: Result is = 0

  clad::differentiate(m_3, 0);
  printf("Result is = %d\n", m_3_darg0(1)); // CHECK-EXEC: Result is = 2

  clad::differentiate(m_4, 0);
  printf("Result is = %d\n", m_4_darg0(1)); // CHECK-EXEC: Result is = 36

  clad::differentiate(m_5, 0);
  printf("Result is = %f\n", m_5_darg0(1)); // CHECK-EXEC: Result is = 3.14

  clad::differentiate(m_6, 0);
  printf("Result is = %f\n", m_6_darg0(1)); // CHECK-EXEC: Result is = 3

  clad::differentiate(d_1, 0);
  printf("Result is = %d\n", d_1_darg0(1)); // CHECK-EXEC: Result is = 0

  clad::differentiate(d_2, 0);
  printf("Result is = %d\n", d_2_darg0(1)); // CHECK-EXEC: Result is = 0

  clad::differentiate(d_3, 0);
  printf("Result is = %d\n", d_3_darg0(1)); // CHECK-EXEC: Result is = 0

  clad::differentiate(d_4, 0);
  printf("Result is = %d\n", d_4_darg0(1)); // CHECK-EXEC: Result is = 0

  clad::differentiate(issue25, 0);
  printf("Result is = %f\n", issue25_darg0(1.4, 2.3)); // CHECK-EXEC: Result is = 1

  clad::differentiate(md_1, 0);
  printf("Result is = %d\n", md_1_darg0(1)); // CHECK-EXEC: Result is = 1

  return 0;
}
