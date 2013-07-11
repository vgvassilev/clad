// RUN: %autodiff %s -I%S/../../include -fsyntax-only 2>&1 | FileCheck %s

#include "autodiff/Differentiator/Differentiator.h"

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

int main () {
  int x = 4;
  diff(basic_1, 1);

  return 0;
}