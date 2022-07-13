// RUN: %cladclang %s -I%S/../../include -oBasicArithmeticMul2.out 
// RUN: ./BasicArithmeticMul2.out | FileCheck -check-prefix=CHECK-EXEC %s
#include "clad/Differentiator/Differentiator.h"
#include "clad/Differentiator/BuiltinDerivatives.h"

extern "C" int printf(const char* fmt, ...);


float test_2(float x, float y) {
  return x * x + y * y;
}

// CHECK: float test_2_darg0(float x, float y) {
// CHECK-NEXT: float _d_x = 1;
// CHECK-NEXT: float _d_y = 0;
// CHECK-NEXT: return _d_x * x + x * _d_x + _d_y * y + y * _d_y;
// CHECK-NEXT: }

//CHECK:   float test_2_d2arg0(float x, float y) {
//CHECK-NEXT:       float _d_x = 1;
//CHECK-NEXT:       float _d_y = 0;
//CHECK-NEXT:       float _d__d_x = 0;
//CHECK-NEXT:       float _d_x0 = 1;
//CHECK-NEXT:       float _d__d_y = 0;
//CHECK-NEXT:       float _d_y0 = 0;
//CHECK-NEXT:       return _d__d_x * x + _d_x0 * _d_x + _d_x * _d_x0 + x * _d__d_x + _d__d_y * y + _d_y0 * _d_y + _d_y * _d_y0 + y * _d__d_y;
//CHECK-NEXT:   }

// CHECK: float test_2_darg1(float x, float y) {
// CHECK-NEXT: float _d_x = 0;
// CHECK-NEXT: float _d_y = 1;
// CHECK-NEXT: return _d_x * x + x * _d_x + _d_y * y + y * _d_y;
// CHECK-NEXT: }

//CHECK:   float test_2_d2arg1(float x, float y) {
//CHECK-NEXT:       float _d_x = 0;
//CHECK-NEXT:       float _d_y = 1;
//CHECK-NEXT:       float _d__d_x = 0;
//CHECK-NEXT:       float _d_x0 = 0;
//CHECK-NEXT:       float _d__d_y = 0;
//CHECK-NEXT:       float _d_y0 = 1;
//CHECK-NEXT:       return _d__d_x * x + _d_x0 * _d_x + _d_x * _d_x0 + x * _d__d_x + _d__d_y * y + _d_y0 * _d_y + _d_y * _d_y0 + y * _d__d_y;
//CHECK-NEXT:   } 
//CHECK:   float test_2_d3arg1(float x, float y) {
//CHECK-NEXT:       float _d_x = 0;
//CHECK-NEXT:       float _d_y = 1;
//CHECK-NEXT:       float _d__d_x = 0;
//CHECK-NEXT:       float _d_x0 = 0;
//CHECK-NEXT:       float _d__d_y = 0;
//CHECK-NEXT:       float _d_y0 = 1;
//CHECK-NEXT:       float _d__d__d_x = 0;
//CHECK-NEXT:       float _d__d_x0 = 0;
//CHECK-NEXT:       float _d__d_x00 = 0;
//CHECK-NEXT:       float _d_x00 = 0;
//CHECK-NEXT:       float _d__d__d_y = 0;
//CHECK-NEXT:       float _d__d_y0 = 0;
//CHECK-NEXT:       float _d__d_y00 = 0;
//CHECK-NEXT:       float _d_y00 = 1;
//CHECK-NEXT:       return _d__d__d_x * x + _d__d_x0 * _d_x + _d__d_x00 * _d_x0 + _d_x00 * _d__d_x + _d__d_x * _d_x00 + _d_x0 * _d__d_x00 + _d_x * _d__d_x0 + x * _d__d__d_x + _d__d__d_y * y + _d__d_y0 * _d_y + _d__d_y00 * _d_y0 + _d_y00 * _d__d_y + _d__d_y * _d_y00 + _d_y0 * _d__d_y00 + _d_y * _d__d_y0 + y * _d__d__d_y;
//CHECK-NEXT:   } 


float test_1_darg0(float x);
float test_1_d2arg0(float x);

float test_2_darg0(float x, float y);
float test_2_d2arg0(float x, float y);
float test_2_darg1(float x, float y);
float test_2_d2arg1(float x, float y);
float test_2_d3arg1(float x, float y);

int main () {
  clad::differentiate<2>(test_2, 0);
  printf("Result is = %f\n", test_2_d2arg0(1.5, 2.5)); // CHECK-EXEC: Result is = 2.000000

  clad::differentiate<3>(test_2, 1);
  printf("Result is = %f\n", test_2_d3arg1(1.5, 2.5)); // CHECK-EXEC: Result is = 0.000000
  return 0;
}
