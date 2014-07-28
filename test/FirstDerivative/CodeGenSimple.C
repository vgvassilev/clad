// RUN: %cladclang %s -I%S/../../include -oCodeGenSimple.out -Xclang -verify 2>&1 | FileCheck %s
// RUN: ./CodeGenSimple.out | FileCheck -check-prefix=CHECK-EXEC %s

//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"
extern "C" int printf(const char* fmt, ...); //expected-warning{{function 'printf' was not differentiated because it is not declared in namespace 'custom_derivatives'}}
int f_1(int x) {
   printf("I am being run!\n");
   return x * x;
}
// CHECK: int f_1_dx(int x) {
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

int f_1_dx(int x);

int main() {
  int x = 4;
  clad::differentiate(f_1, 0);
  printf("Result is = %d\n", f_1_dx(1)); // CHECK-EXEC: Result is = 2
  return 0;
}
