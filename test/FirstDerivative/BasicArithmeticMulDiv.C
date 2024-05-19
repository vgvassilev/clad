// RUN: %cladclang %s -I%S/../../include -oBasicArithmeticMulDiv.out 2>&1 | FileCheck  %s
// RUN: ./BasicArithmeticMulDiv.out | FileCheck -check-prefix=CHECK-EXEC %s

//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

extern "C" int printf(const char* fmt, ...);

float m_1(float x) {
  float y = 4;
  return y * y; // == 0
}
// CHECK: float m_1_darg0(float x) {
// CHECK-NEXT: float _d_x = 1;
// CHECK-NEXT: float _d_y = 0;
// CHECK-NEXT: float y = 4;
// CHECK-NEXT: return _d_y * y + y * _d_y;
// CHECK-NEXT: }

float m_2(float x) {
  return 1 * 1; // == 0
}
// CHECK: float m_2_darg0(float x) {
// CHECK-NEXT: float _d_x = 1;
// CHECK-NEXT: return 0 * 1 + 1 * 0;
// CHECK-NEXT: }

float m_3(float x) {
  return x * x; // == 2 * x
}
// CHECK: float m_3_darg0(float x) {
// CHECK-NEXT: float _d_x = 1;
// CHECK-NEXT: return _d_x * x + x * _d_x;
// CHECK-NEXT: }

float m_4(float x) {
  float y = 4;
  return x * y * x * 3 * x; // == 9 * x * x * y
}
// CHECK: float m_4_darg0(float x) {
// CHECK-NEXT: float _d_x = 1;
// CHECK-NEXT: float _d_y = 0;
// CHECK-NEXT: float y = 4;
// CHECK-NEXT: float _t0 = x * y;
// CHECK-NEXT: float _t1 = _t0 * x;
// CHECK-NEXT: float _t2 = _t1 * 3;
// CHECK-NEXT: return (((_d_x * y + x * _d_y) * x + _t0 * _d_x) * 3 + _t1 * 0) * x + _t2 * _d_x;
// CHECK-NEXT: }

double m_5(float x) {
  return 3.14 * x;
}
// CHECK: double m_5_darg0(float x) {
// CHECK-NEXT: float _d_x = 1;
// CHECK-NEXT: return 0. * x + 3.1400000000000001 * _d_x;
// CHECK-NEXT: }

float m_6(float x) {
  return 3.f * x;
}
// CHECK: float m_6_darg0(float x) {
// CHECK-NEXT: float _d_x = 1;
// CHECK-NEXT: return 0.F * x + 3.F * _d_x;
// CHECK-NEXT: }

double m_7(double x) {
  // returns (x+1)^2
  return (++x, x * x);
}
// CHECK: double m_7_darg0(double x) {
// CHECK-NEXT:   double _d_x = 1;
// CHECK-NEXT:   return (++x , (_d_x * x + x * _d_x));
// CHECK-NEXT: }

double m_8(double x) {
  // returns (x+1)^2
  double temp = (++x, x * x);
  return temp;
}
// CHECK: double m_8_darg0(double x) {
// CHECK-NEXT:   double _d_x = 1;
// CHECK-NEXT:   double _d_temp = (++x , (_d_x * x + x * _d_x));
// CHECK-NEXT:   double temp = (x * x);
// CHECK-NEXT:   return _d_temp;
// CHECK-NEXT: }

double m_9(double x) {
  // returns (2x)^2
  return (x*=2, x * x);
}
// CHECK: double m_9_darg0(double x) {
// CHECK-NEXT:   double _d_x = 1;
// CHECK-NEXT:   return (((_d_x = _d_x * 2 + x * 0) , (x *= 2)) , (_d_x * x + x * _d_x));
// CHECK-NEXT: }

double m_10(double x, bool flag) {
  // if flag is true, return 4x^2, else return (x+1)^2
  return flag ? (x*=2, x * x) : (x+=1, x * x);
}
// CHECK: double m_10_darg0(double x, bool flag) {
// CHECK-NEXT:   double _d_x = 1;
// CHECK-NEXT:   return flag ? (((_d_x = _d_x * 2 + x * 0) , (x *= 2)) , (_d_x * x + x * _d_x)) : (((_d_x += 0) , (x += 1)) , (_d_x * x + x * _d_x));
// CHECK-NEXT: }

template<size_t N>
double m_11(double x) {
  const size_t maxN = 53;
  const size_t m = maxN < N ? maxN : N;
  return x*m;
}

// CHECK: double m_11_darg0(double x) {
// CHECK-NEXT:   double _d_x = 1;
// CHECK-NEXT:   const size_t maxN = 53;
// CHECK-NEXT:   bool _t0 = maxN < {{64U|64UL}};
// CHECK-NEXT:   const size_t m = _t0 ? maxN : {{64U|64UL}};
// CHECK-NEXT:   return _d_x * m + x * 0;
// CHECK-NEXT: }

float d_1(float x) {
  float y = 4;
  return y / y; // == 0
}
// CHECK: float d_1_darg0(float x) {
// CHECK-NEXT: float _d_x = 1;
// CHECK-NEXT: float _d_y = 0;
// CHECK-NEXT: float y = 4;
// CHECK-NEXT: return (_d_y * y - y * _d_y) / (y * y);
// CHECK-NEXT: }

float d_2(float x) {
  return 1 / 1; // == 0
}
// CHECK: float d_2_darg0(float x) {
// CHECK-NEXT: float _d_x = 1;
// CHECK-NEXT: return (0 * 1 - 1 * 0) / (1 * 1);
// CHECK-NEXT: }

float d_3(float x) {
  return x / x; // == 0
}
// CHECK: float d_3_darg0(float x) {
// CHECK-NEXT: float _d_x = 1;
// CHECK-NEXT: return (_d_x * x - x * _d_x) / (x * x);
// CHECK-NEXT: }

float d_4(float x) {
  float y = 4;
  return x / y / x / 3 / x; // == -1 / 3 / x / x / y
}
// CHECK: float d_4_darg0(float x) {
// CHECK-NEXT: float _d_x = 1;
// CHECK-NEXT: float _d_y = 0;
// CHECK-NEXT: float y = 4;
// CHECK-NEXT: float _t0 = x / y;
// CHECK-NEXT: float _t1 = _t0 / x;
// CHECK-NEXT: float _t2 = _t1 / 3;
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

float md_1(float x) {
  float y = 4;
  return x * x / x * y / y * 3 / 3; // == 1
}
// CHECK: float md_1_darg0(float x) {
// CHECK-NEXT: float _d_x = 1;
// CHECK-NEXT: float _d_y = 0;
// CHECK-NEXT: float y = 4;
// CHECK-NEXT: float _t0 = x * x;
// CHECK-NEXT: float _t1 = _t0 / x;
// CHECK-NEXT: float _t2 = _t1 * y;
// CHECK-NEXT: float _t3 = _t2 / y;
// CHECK-NEXT: float _t4 = _t3 * 3;
// CHECK-NEXT: return ((((((((_d_x * x + x * _d_x) * x - _t0 * _d_x) / (x * x)) * y + _t1 * _d_y) * y - _t2 * _d_y) / (y * y)) * 3 + _t3 * 0) * 3 - _t4 * 0) / (3 * 3);
// CHECK-NEXT: }


float m_1_darg0(float x);
float m_2_darg0(float x);
float m_3_darg0(float x);
float m_4_darg0(float x);
double m_5_darg0(float x);
float m_6_darg0(float x);
double m_7_darg0(double x);
double m_8_darg0(double x);
double m_9_darg0(double x);
double m_10_darg0(double x, bool flag);
double m_11_darg0(double x);
float d_1_darg0(float x);
float d_2_darg0(float x);
float d_3_darg0(float x);
float d_4_darg0(float x);
double issue25_darg0(double x, double y);
float md_1_darg0(float x);

int main () {
  float x = 4;
  clad::differentiate(m_1, 0);
  printf("Result is = %.2f\n", m_1_darg0(1)); // CHECK-EXEC: Result is = 0.00

  clad::differentiate(m_2, 0);
  printf("Result is = %.2f\n", m_2_darg0(1)); // CHECK-EXEC: Result is = 0.00

  clad::differentiate(m_3, 0);
  printf("Result is = %.2f\n", m_3_darg0(1)); // CHECK-EXEC: Result is = 2.00

  clad::differentiate(m_4, 0);
  printf("Result is = %.2f\n", m_4_darg0(1)); // CHECK-EXEC: Result is = 36.00

  clad::differentiate(m_5, 0);
  printf("Result is = %f\n", m_5_darg0(1)); // CHECK-EXEC: Result is = 3.14

  clad::differentiate(m_6, 0);
  printf("Result is = %f\n", m_6_darg0(1)); // CHECK-EXEC: Result is = 3.00

  clad::differentiate(m_7, 0);
  printf("Result is = %f\n", m_7_darg0(1)); // CHECK-EXEC: Result is = 4.00

  clad::differentiate(m_8, 0);
  printf("Result is = %f\n", m_8_darg0(1)); // CHECK-EXEC: Result is = 4.00

  clad::differentiate(m_9, 0);
  printf("Result is = %f\n", m_9_darg0(1)); // CHECK-EXEC: Result is = 8.00

  clad::differentiate(m_10, 0);
  printf("Result is = %f\n", m_10_darg0(1, true)); // CHECK-EXEC: Result is = 8.00
  printf("Result is = %f\n", m_10_darg0(1, false)); // CHECK-EXEC: Result is = 4.00

  clad::differentiate(m_11<64>, 0);
  printf("Result is = %f\n", m_11_darg0(1)); // CHECK-EXEC: Result is = 53

  clad::differentiate(d_1, 0);
  printf("Result is = %.2f\n", d_1_darg0(1)); // CHECK-EXEC: Result is = 0.00

  clad::differentiate(d_2, 0);
  printf("Result is = %.2f\n", d_2_darg0(1)); // CHECK-EXEC: Result is = 0.00

  clad::differentiate(d_3, 0);
  printf("Result is = %.2f\n", d_3_darg0(1)); // CHECK-EXEC: Result is = 0.00

  clad::differentiate(d_4, 0);
  printf("Result is = %.2f\n", d_4_darg0(1)); // CHECK-EXEC: Result is = -0.08

  //clad::differentiate(issue25, 0);
  //printf("Result is = %f\n", issue25_darg0(1.4, 2.3)); Result is = 1

  clad::differentiate(md_1, 0);
  printf("Result is = %.2f\n", md_1_darg0(1)); // CHECK-EXEC: Result is = 1.00

  return 0;
}
