// RUN: %autodiff %s -I%S/../../include -oCodeGenSimple.out -Xclang -verify 2>&1 | FileCheck %s
// RUN: ./CodeGenSimple.out | FileCheck -check-prefix=CHECK-EXEC %s

#include "autodiff/Differentiator/Differentiator.h"
extern "C" int printf(const char* fmt, ...);
int f_1(int x) {
   printf("I am being run!\n");
   return x * x;
}
// CHECK: int f_1_derived_x(int x) {
// CHECK-NEXT: printf("I am being run!\n");
// CHECK-NEXT: return (1 * x + x * 1);
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

int f_1_derived_x(int x);

int main() {
  int x = 4;
  diff(f_1, 1);
  printf("Result is = %d\n", f_1_derived_x(1)); // CHECK-EXEC: Result is = 2
  return 0;
}
