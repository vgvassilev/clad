// RUN: %cladclang %s -I%S/../../include 2>&1 -fsyntax-only 

//CHECK-NOT: {{.*error|warning|note:.*}}
#include "clad/Differentiator/Differentiator.h"

// f(x) = | +x*x, x >= 0
//        | -x*x, x < 0
//
// f'(x)= | 2*x, x >= 0
//        | -2*x, x < 0

double f(double x) {
  double result = 0.;
  if (x < 0)
    result = -x*x;
  else
    result = x*x;
  return result;
}

//CHECK:   double f_darg0(double x) {
//CHECK-NEXT:       double _d_x = 1;
//CHECK-NEXT:       double _d_result = 0.;
//CHECK-NEXT:       double result = 0.;
//CHECK-NEXT:       if (x < 0) {
//CHECK-NEXT:           _d_result = -_d_x * x + -x * _d_x;
//CHECK-NEXT:           result = -x * x;
//CHECK-NEXT:       } else {
//CHECK-NEXT:           _d_result = _d_x * x + x * _d_x;
//CHECK-NEXT:           result = x * x;
//CHECK-NEXT:       }
//CHECK-NEXT:       return _d_result;
//CHECK-NEXT:   }

int main () {
  clad::differentiate(f, 0);
  return 0;
}
