// RUN: %cladclang %s -I%S/../../include -oNoNumDiff.out -Xclang -verify 2>&1 | FileCheck -check-prefix=CHECK %s

#include "clad/Differentiator/Differentiator.h"

#include <cmath>

double func(double x) { return std::tgamma(x); } // expected-warning 2{{attempted differentiation of function 'tgamma' without definition and no suitable overload was found in namespace 'custom_derivatives'}}
// expected-note@7 2{{the numerical-diff fallback is off for this request}}

//CHECK: double func_darg0(double x) {
//CHECK-NEXT:     double _d_x = 1;
//CHECK-NEXT:     return 0.;
//CHECK-NEXT: }

//CHECK: void func_grad(double x, double *_d_x) {
//CHECK-NEXT:     {
//CHECK-NEXT:         double _r0 = 0.;
//CHECK-NEXT:         *_d_x += _r0;
//CHECK-NEXT:     }
//CHECK-NEXT: }


int main(){
  clad::differentiate(func, "x");
  clad::gradient(func);
}
