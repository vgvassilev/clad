// RUN: %cladclang %s -I%S/../../include -oCodeGenSimple.out 2>&1 | %filecheck %s
// RUN: ./CodeGenSimple.out | %filecheck_exec %s

//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"
extern "C" int printf(const char* fmt, ...);

int f_1(int x) {
   printf("I am being run!\n");
   return x * x;
}
// CHECK: int f_1_darg0(int x) {
// CHECK-NEXT: int _d_x = 1;
// CHECK-NEXT: printf("I am being run!\n");
// CHECK-NEXT: return _d_x * x + x * _d_x;
// CHECK-NEXT: }


int f_2(int x) {
  return (1 * x + x * 1);
}

int f_3(int x) {
  return 1 * x;
}

int f_4(int x) {
  return (0 * x + 1 * 1);
}

extern "C" int printf(const char* fmt, ...);

int f_1_darg0(int x);

double sq_defined_later(double);

int main() {
  int x = 4;
  clad::differentiate(f_1, 0);
  auto df = clad::differentiate(sq_defined_later, "x");
  printf("Result is = %d\n", f_1_darg0(1)); // CHECK-EXEC: Result is = 2
  printf("Result is = %f\n", df.execute(3)); // CHECK-EXEC: Result is = 6
  return 0;
}

double sq_defined_later(double x) {
  return x * x;
}
