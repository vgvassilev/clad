// RUN: %cladclang %s -I%S/../../include -oBasicArithmeticAll.out 
// RUN: ./BasicArithmeticAll.out | FileCheck -check-prefix=CHECK-EXEC %s

//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

extern "C" int printf(const char* fmt, ...);

float basic_1(int x) {
  int y = 4;
  int z = 3;
  return (y + x) / (x - z) * ((x * y * z) / 5); // == y * z * (x * x - 2 * x * z - y * z) / (5 * (x - z) * (x - z))
}
// CHECK: float basic_1_darg0(int x) {
// CHECK-NEXT: int _d_x = 1;
// CHECK-NEXT: int _d_y = 0;
// CHECK-NEXT: int y = 4;
// CHECK-NEXT: int _d_z = 0;
// CHECK-NEXT: int z = 3;
// CHECK-NEXT: int _t0 = (y + x);
// CHECK-NEXT: int _t1 = (x - z);
// CHECK-NEXT: int _t2 = x * y;
// CHECK-NEXT: int _t3 = (_t2 * z);
// CHECK-NEXT: int _t4 = _t0 / _t1;
// CHECK-NEXT: int _t5 = (_t3 / 5);
// CHECK-NEXT: return (((_d_y + _d_x) * _t1 - _t0 * (_d_x - _d_z)) / (_t1 * _t1)) * _t5 + _t4 * ((((_d_x * y + x * _d_y) * z + _t2 * _d_z) * 5 - _t3 * 0) / (5 * 5));
// CHECK-NEXT: }

float basic_1_darg0(int x);

double fn1(double i, double j) {
  double t = 1;
  (t *= j) *= (i *= i);
  return t; // return i*i*j;
}

// CHECK:  double fn1_darg0(double i, double j) {
// CHECK-NEXT:      double _d_i = 1;
// CHECK-NEXT:      double _d_j = 0;
// CHECK-NEXT:      double _d_t = 0;
// CHECK-NEXT:      double t = 1;
// CHECK-NEXT:      double &_t0 = (_d_t = _d_t * j + t * _d_j);
// CHECK-NEXT:      double &_t1 = (t *= j);
// CHECK-NEXT:      double &_t2 = (_d_i = _d_i * i + i * _d_i);
// CHECK-NEXT:      double &_t3 = (i *= i);
// CHECK-NEXT:      _t0 = _t0 * _t3 + _t1 * _t2;
// CHECK-NEXT:      _t1 *= _t3;
// CHECK-NEXT:      return _d_t;
// CHECK-NEXT:  }

double fn2(double i, double j) {
  double t = i*i*j*j;
  (t /= i*j)*=(j);
  return t; // return i*j*j;
}

// CHECK:  double fn2_darg0(double i, double j) {
// CHECK-NEXT:      double _d_i = 1;
// CHECK-NEXT:      double _d_j = 0;
// CHECK-NEXT:      double _t0 = i * i;
// CHECK-NEXT:      double _t1 = _t0 * j;
// CHECK-NEXT:      double _d_t = ((_d_i * i + i * _d_i) * j + _t0 * _d_j) * j + _t1 * _d_j;
// CHECK-NEXT:      double t = _t1 * j;
// CHECK-NEXT:      double _t2 = _d_i * j + i * _d_j;
// CHECK-NEXT:      double _t3 = i * j;
// CHECK-NEXT:      double &_t4 = (_d_t = (_d_t * _t3 - t * _t2) / (_t3 * _t3));
// CHECK-NEXT:      double &_t5 = (t /= _t3);
// CHECK-NEXT:      _t4 = _t4 * j + _t5 * _d_j;
// CHECK-NEXT:      _t5 *= j;
// CHECK-NEXT:      return _d_t;
// CHECK-NEXT:  }

double fn3(double i, double j) {
  double t = 1;
  ((t*=(i*j))*=j)*=i;
  return t; // i*i*j*j
}

// CHECK:  double fn3_darg0(double i, double j) {
// CHECK-NEXT:      double _d_i = 1;
// CHECK-NEXT:      double _d_j = 0;
// CHECK-NEXT:      double _d_t = 0;
// CHECK-NEXT:      double t = 1;
// CHECK-NEXT:      double _t0 = (_d_i * j + i * _d_j);
// CHECK-NEXT:      double _t1 = (i * j);
// CHECK-NEXT:      double &_t2 = (_d_t = _d_t * _t1 + t * _t0);
// CHECK-NEXT:      double &_t3 = (t *= _t1);
// CHECK-NEXT:      double &_t4 = (_t2 = _t2 * j + _t3 * _d_j);
// CHECK-NEXT:      double &_t5 = (_t3 *= j);
// CHECK-NEXT:      _t4 = _t4 * i + _t5 * _d_i;
// CHECK-NEXT:      _t5 *= i;
// CHECK-NEXT:      return _d_t;
// CHECK-NEXT:  }

#define INIT(fn, arg) auto d_##fn = clad::differentiate(fn, arg);
#define TEST(fn, ...) printf("%.2f\n", d_##fn.execute(__VA_ARGS__));

int main () {
  clad::differentiate(basic_1, 0);
  printf("Result is = %f\n", basic_1_darg0(1)); // CHECK-EXEC: Result is = -6
  INIT(fn1, "i");
  INIT(fn2, "i");
  INIT(fn3, "i");

  TEST(fn1, 3, 5);  // CHECK-EXEC: 30.00
  TEST(fn1, 5, 7);  // CHECK-EXEC: 70.00
  TEST(fn2, 3, 5);  // CHECK-EXEC: 25.00
  TEST(fn2, 5, 7);  // CHECK-EXEC: 49.00
  TEST(fn3, 3, 5);  // CHECK-EXEC: 150.00
  TEST(fn3, 5, 7);  // CHECK-EXEC: 490.00
  return 0;
}
