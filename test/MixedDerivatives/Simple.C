// RUN: %cladclang %s -I%S/../../include -oSimple.out -Xclang -verify 
// RUN: ./Simple.out | FileCheck -check-prefix=CHECK-EXEC %s

//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

extern "C" int printf(const char* fmt, ...);

float f1(float x, float y) {
  return x * x + y * y;
}

// CHECK: float f1_darg0(float x, float y) {
// CHECK-NEXT:    float _d_x = 1;
// CHECK-NEXT:    float _d_y = 0;
// CHECK-NEXT:    return _d_x * x + x * _d_x + _d_y * y + y * _d_y;
// CHECK-NEXT: }

//CHECK:   float f1_darg0_darg1(float x, float y) {
//CHECK-NEXT:       float _d_x = 0;
//CHECK-NEXT:       float _d_y = 1;
//CHECK-NEXT:       float _d__d_x = 0;
//CHECK-NEXT:       float _d_x0 = 1;
//CHECK-NEXT:       float _d__d_y = 0;
//CHECK-NEXT:       float _d_y0 = 0;
//CHECK-NEXT:       return _d__d_x * x + _d_x0 * _d_x + _d_x * _d_x0 + x * _d__d_x + _d__d_y * y + _d_y0 * _d_y + _d_y * _d_y0 + y * _d__d_y;
//CHECK-NEXT:   }

float f1_darg0(float x, float y);

int main () { // expected-no-diagnostics
  auto f1_darg0_clad_func = clad::differentiate(f1, 0);
  printf("Result is = %f\n", f1_darg0_clad_func.execute(1.1, 2.1)); // CHECK-EXEC: Result is = 2.2

  auto f1_darg0_darg1_clad_func = clad::differentiate(f1_darg0, 1);
  printf("Result is = %f\n", f1_darg0_darg1_clad_func.execute(1.1, 2.1)); // CHECK-EXEC: Result is = 0

  return 0;
}
