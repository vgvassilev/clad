// RUN: %cladclang %s -I%S/../../include -oCodeGenSimple.out 2>&1 | FileCheck %s
// RUN: ./CodeGenSimple.out | FileCheck -check-prefix=CHECK-EXEC %s

//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"
extern "C" int printf(const char* fmt, ...);

float f_1(float x) {
   printf("I am being run!\n");
   return x * x;
}
// CHECK: float f_1_darg0(float x) {
// CHECK-NEXT: float _d_x = 1;
// CHECK-NEXT: printf("I am being run!\n");
// CHECK-NEXT: return _d_x * x + x * _d_x;
// CHECK-NEXT: }


float f_2(float x) {
  return (1 * x + x * 1);
}

float f_3(float x) {
  return 1 * x;
}

float f_4(float x) {
  return (0 * x + 1 * 1);
}

extern "C" int printf(const char* fmt, ...);

float f_1_darg0(float x);

double sq_defined_later(double);

int main() {
  clad::differentiate(f_1, 0);
  auto df = clad::differentiate(sq_defined_later, "x");
  printf("Result is = %.2f\n", f_1_darg0(1)); // CHECK-EXEC: Result is = 2.00
  printf("Result is = %.2f\n", df.execute(3)); // CHECK-EXEC: Result is = 6.00
  return 0;
}

double sq_defined_later(double x) {
  return x * x;
}
