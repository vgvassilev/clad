// RUN: %cladclang %s -I%S/../../include -oSimple.out -Xclang -verify 2>&1 | FileCheck %s
// RUN: ./Simple.out | FileCheck -check-prefix=CHECK-EXEC %s

//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

extern "C" int printf(const char* fmt, ...);

float f1(float x, float y) {
  return x * x + y * y;
}
// CHECK: float f1_darg0(float x, float y) {
// CHECK-NEXT: return (1.F * x + x * 1.F) + ((0.F * y + y * 0.F));
// CHECK-NEXT: }

// CHECK: float f1_darg0_darg1(float x, float y) {
// CHECK-NEXT: return ((0.F * x + 1.F * 0.F) + ((0.F * 1.F + x * 0.F))) + ((((0.F * y + 0.F * 1.F) + ((1.F * 0.F + y * 0.F)))));
// CHECK-NEXT: }

float f1_darg0(float x, float y);

int main () { // expected-no-diagnostics
  auto f1_darg0_clad_func = clad::differentiate(f1, 0);
  printf("Result is = %f\n", f1_darg0_clad_func.execute(1.1, 2.1)); // CHECK-EXEC: Result is = 2.2

  auto f1_darg0_darg1_clad_func = clad::differentiate(f1_darg0, 1);
  printf("Result is = %f\n", f1_darg0_darg1_clad_func.execute(1.1, 2.1)); // CHECK-EXEC: Result is = 0

  return 0;
}
