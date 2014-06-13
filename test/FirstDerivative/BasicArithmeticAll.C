// RUN: %cladclang %s -I%S/../../include -oBasicArithmeticAll.out 2>&1 | FileCheck %s
// RUN: ./BasicArithmeticAll.out | FileCheck -check-prefix=CHECK-EXEC %s
#include "clad/Differentiator/Differentiator.h"

extern "C" int printf(const char* fmt, ...);

float basic_1(int x) {
  int y = 4;
  int z = 3;
  return (y + x) / (x - z) * ((x * y * z) / 5); // == y * z * (x * x - 2 * x * z - y * z) / (5 * (x - z) * (x - z))
}
// CHECK: float basic_1_derived_x(int x) {
// CHECK-NEXT: int y = 4;
// CHECK-NEXT: int z = 3;
// CHECK-NEXT: return (((0 + (1)) * (x - z) - (y + x) * (1 - (0))) / ((x - z) * (x - z)) * ((x * y * z) / 5) + (y + x) / (x - z) * (((((1 * y + x * 0) * z + x * y * 0)) * 5 - (x * y * z) * 0) / (5 * 5)));
// CHECK-NEXT: }

float basic_1_derived_x(int x);

int main () {
  clad::differentiate(basic_1, 1);
  printf("Result is = %f\n", basic_1_derived_x(1)); // CHECK-EXEC: Result is = -6
  return 0;
}
