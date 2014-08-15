// RUN: %cladclang %s -I%S/../../include -Xclang -verify -oBuiltinDerivatives.out -lm 2>&1 | FileCheck %s
// RUN: ./BuiltinDerivatives.out | FileCheck -check-prefix=CHECK-EXEC %s

//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

extern "C" int printf(const char* fmt, ...);

float f1(float x) {
  return sin(x);
}

// CHECK: float f1_dx(float x) {
// CHECK-NEXT: return sin_dx(x) * (1.F);
// CHECK-NEXT: }

float f2(float x) {
  return cos(x);
}

// CHECK: float f2_dx(float x) {
// CHECK-NEXT: cos_dx(x) * (1.F);
// CHECK-NEXT: }

int main () { //expected-no-diagnostics
  auto f1_dx = clad::differentiate(f1, 0);
  printf("Result is = %f\n", f1_dx.execute(60)); // CHECK-EXEC: Result is = -0.952413

  auto f2_dx = clad::differentiate(f2, 0);
  printf("Result is = %f\n", f2_dx.execute(60)); //CHECK-EXEC: Result is = 0.304811

  return 0;
}
