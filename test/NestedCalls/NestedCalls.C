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
//CHECK-NEXT:       double _t0 = 1 * x;
//CHECK-NEXT:       _result[0UL] += _t0;
//CHECK-NEXT:       double _t1 = x * 1;
//CHECK-NEXT:       _result[0UL] += _t1;
//CHECK-NEXT:       return;
//CHECK-NEXT:   }

//CHECK:   void sq_grad(double x, double *_result) {
//CHECK-NEXT:       double _t0 = 1 * x;
//CHECK-NEXT:       _result[0UL] += _t0;
//CHECK-NEXT:       double _t1 = x * 1;
//CHECK-NEXT:       _result[0UL] += _t1;
//CHECK-NEXT:       return;
//CHECK-NEXT:   }

//CHECK:   void one_grad(double x, double *_result) {
//CHECK-NEXT:       double _grad[1] = {};
//CHECK-NEXT:       sq_grad(std::sin(x), _grad);
//CHECK-NEXT:       double _t0 = 1 * _grad[0UL];
//CHECK-NEXT:       double _t1 = custom_derivatives::sin_darg0(x);
//CHECK-NEXT:       double _t2 = _t0 * _t1;
//CHECK-NEXT:       _result[0UL] += _t2;
//CHECK-NEXT:       double _grad3[1] = {};
//CHECK-NEXT:       sq_grad(std::cos(x), _grad3);
//CHECK-NEXT:       double _t4 = 1 * _grad3[0UL];
//CHECK-NEXT:       double _t5 = custom_derivatives::cos_darg0(x);
//CHECK-NEXT:       double _t6 = _t4 * _t5;
//CHECK-NEXT:       _result[0UL] += _t6;
//CHECK-NEXT:       return;
//CHECK-NEXT:   }

//CHECK:   void f_grad(double x, double y, double *_result) {
//CHECK-NEXT:       double _d_t = 0;
//CHECK-NEXT:       double t = one(x);
//CHECK-NEXT:       double _t1 = 1 * y;
//CHECK-NEXT:       _d_t += _t1;
//CHECK-NEXT:       double _t2 = t * 1;
//CHECK-NEXT:       _result[1UL] += _t2;
//CHECK-NEXT:       double _grad[1] = {};
//CHECK-NEXT:       one_grad(x, _grad);
//CHECK-NEXT:       double _t0 = _d_t * _grad[0UL];
//CHECK-NEXT:       _result[0UL] += _t0;
//CHECK-NEXT:       return;
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
