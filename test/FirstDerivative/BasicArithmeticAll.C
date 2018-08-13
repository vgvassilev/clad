// RUN: %cladclang %s -I%S/../../include -oBasicArithmeticAll.out 2>&1 | FileCheck %s
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

int main () {
  clad::differentiate(basic_1, 0);
  printf("Result is = %f\n", basic_1_darg0(1)); // CHECK-EXEC: Result is = -6
  return 0;
}
