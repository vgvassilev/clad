// RUN: %cladclang %s -I%S/../../include -oCodeGenSimple.out -Xclang -verify 
// RUN: ./CodeGenSimple.out | FileCheck -check-prefix=CHECK-EXEC %s

//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"
extern "C" int printf(const char* fmt, ...);

int f_1(int x) {
   printf("I am being run!\n"); //expected-warning{{attempted to differentiate unsupported statement, no changes applied}} //expected-warning{{function 'printf' was not differentiated because clad failed to differentiate it and no suitable overload was found in namespace 'custom_derivatives', and function may not be eligible for numerical differentiation.}}
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

int main() {
  int x = 4;
  clad::differentiate(f_1, 0);
  printf("Result is = %d\n", f_1_darg0(1)); // CHECK-EXEC: Result is = 2
  return 0;
}
