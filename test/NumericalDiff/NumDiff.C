// RUN: %cladnumdiffclang %s -I%S/../../include -oNumDiff.out 2>&1 | FileCheck -check-prefix=CHECK %s

//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

double test_1(double x){
   return tanh(x);
}
//CHECK: warning: Falling back to numerical differentiation for 'tanh' since no suitable overload was found and clad could not derive it. To disable this feature, compile your programs with -DCLAD_NO_NUM_DIFF.
//CHECK: warning: Falling back to numerical differentiation for 'log10' since no suitable overload was found and clad could not derive it. To disable this feature, compile your programs with -DCLAD_NO_NUM_DIFF.

//CHECK: void test_1_grad(double x, clad::array_ref<double> _d_x) {
//CHECK-NEXT:     double _t0;
//CHECK-NEXT:     _t0 = x;
//CHECK-NEXT:     {
//CHECK-NEXT:         double _r0 = 1 * numerical_diff::forward_central_difference(tanh, _t0, 0, 0, _t0);
//CHECK-NEXT:         * _d_x += _r0;
//CHECK-NEXT:     }
//CHECK-NEXT: }


double test_2(double x){
   return std::log10(x);
}
//CHECK: double test_2_darg0(double x) {
//CHECK-NEXT:     double _d_x = 1;
//CHECK-NEXT:     return numerical_diff::forward_central_difference(std::log10, x, 0, 0, x) * _d_x;
//CHECK-NEXT: }

int main(){
    clad::gradient(test_1);
    clad::differentiate(test_2, 0);
}
