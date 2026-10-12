// RUN: %cladclang %s -I%S/../../include -oSimple.out -Xclang -verify 2>&1 | %filecheck %s
// RUN: ./Simple.out | %filecheck_exec %s

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

double f2(double x, double y) { return x * x * y; }

void f2_grad_0(double x, double y, double *_d_x);

double f2_dx(double x) {
  double d = 0;
  f2_grad_0(x, 2., &d);
  return d;
}

// CHECK: void f2_grad_0_pushforward(double x, double y, double *_d_x, double _d_x0, double _d_y, double *_d__d_x) {
// CHECK-NEXT:     double _d__d_y = 0.;
// CHECK-NEXT:     double _d_y0 = 0.;
// CHECK-NEXT:     {
// CHECK-NEXT:         double _t0 = 1 * y;
// CHECK-NEXT:         *_d__d_x += _d_y * x + _t0 * _d_x0;
// CHECK-NEXT:         *_d_x += _t0 * x;
// CHECK-NEXT:         double _t1 = 1 * y;
// CHECK-NEXT:         *_d__d_x += _d_x0 * _t1 + x * _d_y;
// CHECK-NEXT:         *_d_x += x * _t1;
// CHECK-NEXT:         _d__d_y += (_d_x0 * x + x * _d_x0);
// CHECK-NEXT:         _d_y0 += x * x * 1;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: double f2_dx_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     double _d_d = 0;
// CHECK-NEXT:     double d = 0;
// CHECK-NEXT:     f2_grad_0_pushforward(x, 2., &d, _d_x, 0., &_d_d);
// CHECK-NEXT:     return _d_d;
// CHECK-NEXT: }

int main () { // expected-no-diagnostics
  auto f1_darg0_clad_func = clad::differentiate(f1, 0);
  printf("Result is = %f\n", f1_darg0_clad_func.execute(1.1, 2.1)); // CHECK-EXEC: Result is = 2.2

  auto f1_darg0_darg1_clad_func = clad::differentiate(f1_darg0, 1);
  printf("Result is = %f\n", f1_darg0_darg1_clad_func.execute(1.1, 2.1)); // CHECK-EXEC: Result is = 0

  auto f2_grad_0_clad_func = clad::gradient(f2, "x");
  auto f2_dx_darg0_clad_func = clad::differentiate(f2_dx, "x");
  printf("Result is = %f\n", f2_dx_darg0_clad_func.execute(3)); // CHECK-EXEC: Result is = 4.000000

  return 0;
}
