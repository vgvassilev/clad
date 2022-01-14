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
// CHECK-NEXT: int _d_x = 1;
// CHECK-NEXT: int _d_y = 0;
// CHECK-NEXT: int y = 4;
// CHECK-NEXT: return _d_y * y + y * _d_y;
// CHECK-NEXT: }

int m_2(int x) {
  return 1 * 1; // == 0
}
// CHECK: int m_2_darg0(int x) {
// CHECK-NEXT: int _d_x = 1;
// CHECK-NEXT: return 0 * 1 + 1 * 0;
// CHECK-NEXT: }

int m_3(int x) {
  return x * x; // == 2 * x
}
// CHECK: int m_3_darg0(int x) {
// CHECK-NEXT: int _d_x = 1;
// CHECK-NEXT: return _d_x * x + x * _d_x;
// CHECK-NEXT: }

int m_4(int x) {
  int y = 4;
  return x * y * x * 3 * x; // == 9 * x * x * y
}
// CHECK: int m_4_darg0(int x) {
// CHECK-NEXT: int _d_x = 1;
// CHECK-NEXT: int _d_y = 0;
// CHECK-NEXT: int y = 4;
// CHECK-NEXT: int _t0 = x * y;
// CHECK-NEXT: int _t1 = _t0 * x;
// CHECK-NEXT: int _t2 = _t1 * 3;
// CHECK-NEXT: return (((_d_x * y + x * _d_y) * x + _t0 * _d_x) * 3 + _t1 * 0) * x + _t2 * _d_x;
// CHECK-NEXT: }

double m_5(int x) {
  return 3.14 * x;
}
// CHECK: double m_5_darg0(int x) {
// CHECK-NEXT: int _d_x = 1;
// CHECK-NEXT: return 0. * x + 3.1400000000000001 * _d_x;
// CHECK-NEXT: }

float m_6(int x) {
  return 3.f * x;
}
// CHECK: float m_6_darg0(int x) {
// CHECK-NEXT: int _d_x = 1;
// CHECK-NEXT: return 0.F * x + 3.F * _d_x;
// CHECK-NEXT: }

int d_1(int x) {
  int y = 4;
  return y / y; // == 0
}
// CHECK: int d_1_darg0(int x) {
// CHECK-NEXT: int _d_x = 1;
// CHECK-NEXT: int _d_y = 0;
// CHECK-NEXT: int y = 4;
// CHECK-NEXT: return (_d_y * y - y * _d_y) / (y * y);
// CHECK-NEXT: }

int d_2(int x) {
  return 1 / 1; // == 0
}
// CHECK: int d_2_darg0(int x) {
// CHECK-NEXT: int _d_x = 1;
// CHECK-NEXT: return (0 * 1 - 1 * 0) / (1 * 1);
// CHECK-NEXT: }

int d_3(int x) {
  return x / x; // == 0
}
// CHECK: int d_3_darg0(int x) {
// CHECK-NEXT: int _d_x = 1;
// CHECK-NEXT: return (_d_x * x - x * _d_x) / (x * x);
// CHECK-NEXT: }

int d_4(int x) {
  int y = 4;
  return x / y / x / 3 / x; // == -1 / 3 / x / x / y
}
// CHECK: int d_4_darg0(int x) {
// CHECK-NEXT: int _d_x = 1;
// CHECK-NEXT: int _d_y = 0;
// CHECK-NEXT: int y = 4;
// CHECK-NEXT: int _t0 = x / y;
// CHECK-NEXT: int _t1 = _t0 / x;
// CHECK-NEXT: int _t2 = _t1 / 3;
// CHECK-NEXT: return (((((((_d_x * y - x * _d_y) / (y * y)) * x - _t0 * _d_x) / (x * x)) * 3 - _t1 * 0) / (3 * 3)) * x - _t2 * _d_x) / (x * x);
// CHECK-NEXT: }

double issue25(double x, double y) {
  x+=y;
  x-=y;
  x/=y;
  x*=y;

  return x;
}

// FIXME: +=, etc. operators are not currently supported, ignore for now.
// double issue25_darg0(double x, double y) {
//   x += 0.;
//   x -= 0.;
//   x /= 0.;
//   x *= 0.;
//   return 1.;
// }

int md_1(int x) {
  int y = 4;
  return x * x / x * y / y * 3 / 3; // == 1
}
// CHECK: int md_1_darg0(int x) {
// CHECK-NEXT: int _d_x = 1;
// CHECK-NEXT: int _d_y = 0;
// CHECK-NEXT: int y = 4;
// CHECK-NEXT: int _t0 = x * x;
// CHECK-NEXT: int _t1 = _t0 / x;
// CHECK-NEXT: int _t2 = _t1 * y;
// CHECK-NEXT: int _t3 = _t2 / y;
// CHECK-NEXT: int _t4 = _t3 * 3;
// CHECK-NEXT: return ((((((((_d_x * x + x * _d_x) * x - _t0 * _d_x) / (x * x)) * y + _t1 * _d_y) * y - _t2 * _d_y) / (y * y)) * 3 + _t3 * 0) * 3 - _t4 * 0) / (3 * 3);
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

  //clad::differentiate(issue25, 0);
  //printf("Result is = %f\n", issue25_darg0(1.4, 2.3)); Result is = 1

  clad::differentiate(md_1, 0);
  printf("Result is = %d\n", md_1_darg0(1)); // CHECK-EXEC: Result is = 1

  return 0;
}
