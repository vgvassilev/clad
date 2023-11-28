// RUN: %cladclang %s -I%S/../../include -oNoNumDiff.out 2>&1 | FileCheck -check-prefix=CHECK %s

//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

#include <cmath>

double func(double x) { return std::tanh(x); }

//CHECK: warning: Numerical differentiation is diabled using the -DCLAD_NO_NUM_DIFF flag, this means that every try to numerically differentiate a function will fail! Remove the flag to revert to default behaviour.
//CHECK: warning: Numerical differentiation is diabled using the -DCLAD_NO_NUM_DIFF flag, this means that every try to numerically differentiate a function will fail! Remove the flag to revert to default behaviour.
//CHECK: double func_darg0(double x) {
//CHECK-NEXT:     double _d_x = 1;
//CHECK-NEXT:     return 0;
//CHECK-NEXT: }

//CHECK: void func_grad(double x, clad::array_ref<double> _d_x) {
//CHECK-NEXT:     double _t0;
//CHECK-NEXT:     _t0 = x;
//CHECK-NEXT:     {
//CHECK-NEXT:         double _grad0 = 0.;
//CHECK-NEXT:         double _r0;
//CHECK-NEXT:         * _d_x += _r0;
//CHECK-NEXT:     }
//CHECK-NEXT: }


int main(){
    clad::differentiate(func, "x");
    clad::gradient(func);
}
