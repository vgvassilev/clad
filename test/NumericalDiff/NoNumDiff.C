// RUN: %cladclang %s -I%S/../../include -oNoNumDiff.out -Xclang -verify 2>&1 | FileCheck -check-prefix=CHECK %s

//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

#include <cmath>

double func(double x) { return std::tanh(x); } // expected-warning 2{{function 'tanh' was not differentiated because clad failed to differentiate it and no suitable overload was found in namespace 'custom_derivatives'}}
// expected-note@9 2{{fallback to numerical differentiation is disabled by the 'CLAD_NO_NUM_DIFF' macro}}

//CHECK: double func_darg0(double x) {
//CHECK-NEXT:     double _d_x = 1;
//CHECK-NEXT:     return 0;
//CHECK-NEXT: }

//CHECK: void func_grad(double x, double *_d_x) {
//CHECK-NEXT:     {
//CHECK-NEXT:         double _r0 = 0;
//CHECK-NEXT:         *_d_x += _r0;
//CHECK-NEXT:     }
//CHECK-NEXT: }


int main(){
  clad::differentiate(func, "x");
  clad::gradient(func);
}
