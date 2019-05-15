// RUN: %cladclang %s -lm -I%S/../../include -oNestedCalls.out 2>&1 | FileCheck %s
// RUN: ./NestedCalls.out | FileCheck -check-prefix=CHECK-EXEC %s

//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"
#include <cmath>

extern "C" int printf(const char* fmt, ...);

double sq(double x) { return x * x; }
//CHECK:   double sq_darg0(double x) {
//CHECK-NEXT:       double _d_x = 1;
//CHECK-NEXT:       return _d_x * x + x * _d_x;
//CHECK-NEXT:   } 

//CHECK:   double sq_darg0(double x) {
//CHECK-NEXT:       double _d_x = 1;
//CHECK-NEXT:       return _d_x * x + x * _d_x;
//CHECK-NEXT:   } 

double one(double x) { return sq(std::sin(x)) + sq(std::cos(x)); }
//CHECK:   double one_darg0(double x) {
//CHECK-NEXT:       double _d_x = 1;
//CHECK-NEXT:       return sq_darg0(std::sin(x)) * (custom_derivatives::sin_darg0(x) * _d_x) + sq_darg0(std::cos(x)) * (custom_derivatives::cos_darg0(x) * _d_x);
//CHECK-NEXT:   }

double f(double x, double y) {
  double t = one(x);
  return t * y;
}
//CHECK:   double f_darg0(double x, double y) {
//CHECK-NEXT:       double _d_x = 1;
//CHECK-NEXT:       double _d_y = 0;
//CHECK-NEXT:       double _d_t = one_darg0(x) * _d_x;
//CHECK-NEXT:       double t = one(x);
//CHECK-NEXT:       return _d_t * y + t * _d_y;
//CHECK-NEXT:   }

//CHECK:   void sq_grad(double x, double *_result) {
//CHECK-NEXT:       double _t0;
//CHECK-NEXT:       double _t1;
//CHECK-NEXT:       _t1 = x;
//CHECK-NEXT:       _t0 = x;
//CHECK-NEXT:       double sq_return = _t1 * _t0;
//CHECK-NEXT:       goto _label0;
//CHECK-NEXT:     _label0:
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r0 = 1 * _t0;
//CHECK-NEXT:           _result[0UL] += _r0;
//CHECK-NEXT:           double _r1 = _t1 * 1;
//CHECK-NEXT:           _result[0UL] += _r1;
//CHECK-NEXT:       }
//CHECK-NEXT:   }

//CHECK:   void sq_grad(double x, double *_result) {
//CHECK-NEXT:       double _t0;
//CHECK-NEXT:       double _t1;
//CHECK-NEXT:       _t1 = x;
//CHECK-NEXT:       _t0 = x;
//CHECK-NEXT:       double sq_return = _t1 * _t0;
//CHECK-NEXT:       goto _label0;
//CHECK-NEXT:     _label0:
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r0 = 1 * _t0;
//CHECK-NEXT:           _result[0UL] += _r0;
//CHECK-NEXT:           double _r1 = _t1 * 1;
//CHECK-NEXT:           _result[0UL] += _r1;
//CHECK-NEXT:       }
//CHECK-NEXT:   }

//CHECK:   void one_grad(double x, double *_result) {
//CHECK-NEXT:       double _t0;
//CHECK-NEXT:       double _t1;
//CHECK-NEXT:       double _t2;
//CHECK-NEXT:       double _t3;
//CHECK-NEXT:       _t0 = x;
//CHECK-NEXT:       _t1 = std::sin(_t0);
//CHECK-NEXT:       _t2 = x;
//CHECK-NEXT:       _t3 = std::cos(_t2);
//CHECK-NEXT:       double one_return = sq(_t1) + sq(_t3);
//CHECK-NEXT:       goto _label0;
//CHECK-NEXT:     _label0:
//CHECK-NEXT:       {
//CHECK-NEXT:           double _grad0[1] = {};
//CHECK-NEXT:           sq_grad(_t1, _grad0);
//CHECK-NEXT:           double _r0 = 1 * _grad0[0UL];
//CHECK-NEXT:           double _r1 = _r0 * custom_derivatives::sin_darg0(_t0);
//CHECK-NEXT:           _result[0UL] += _r1;
//CHECK-NEXT:           double _grad1[1] = {};
//CHECK-NEXT:           sq_grad(_t3, _grad1);
//CHECK-NEXT:           double _r2 = 1 * _grad1[0UL];
//CHECK-NEXT:           double _r3 = _r2 * custom_derivatives::cos_darg0(_t2);
//CHECK-NEXT:           _result[0UL] += _r3;
//CHECK-NEXT:       }
//CHECK-NEXT:   }

//CHECK:   void f_grad(double x, double y, double *_result) {
//CHECK-NEXT:       double _t0;
//CHECK-NEXT:       double _d_t = 0;
//CHECK-NEXT:       double _t1;
//CHECK-NEXT:       double _t2;
//CHECK-NEXT:       _t0 = x;
//CHECK-NEXT:       double t = one(_t0);
//CHECK-NEXT:       _t2 = t;
//CHECK-NEXT:       _t1 = y;
//CHECK-NEXT:       double f_return = _t2 * _t1;
//CHECK-NEXT:       goto _label0;
//CHECK-NEXT:     _label0:
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r1 = 1 * _t1;
//CHECK-NEXT:           _d_t += _r1;
//CHECK-NEXT:           double _r2 = _t2 * 1;
//CHECK-NEXT:           _result[1UL] += _r2;
//CHECK-NEXT:       }
//CHECK-NEXT:       {
//CHECK-NEXT:           double _grad0[1] = {};
//CHECK-NEXT:           one_grad(_t0, _grad0);
//CHECK-NEXT:           double _r0 = _d_t * _grad0[0UL];
//CHECK-NEXT:           _result[0UL] += _r0;
//CHECK-NEXT:       }
//CHECK-NEXT:   }

int main () { // expected-no-diagnostics
  auto df = clad::differentiate(f, 0);
  printf("%.2f\n", df.execute(1, 2)); // CHECK-EXEC: 0.00
  printf("%.2f\n", df.execute(10, 11)); // CHECK-EXEC: 0.00
   
  auto gradf = clad::gradient(f);
  double result[2] = {};
  gradf.execute(2, 3, result);
  printf("{%.2f, %.2f}\n", result[0], result[1]); // CHECK-EXEC: {0.00, 1.00}
  return 0;
}
