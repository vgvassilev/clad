// RUN: %cladclang %s -I%S/../../include -oDifferentCladEnzymeDerivatives.out | FileCheck %s
// RUN: ./DifferentCladEnzymeDerivatives.out
// CHECK-NOT: {{.*error|warning|note:.*}}
// REQUIRES: Enzyme

#include "clad/Differentiator/Differentiator.h"


double foo(double x, double y){
    return x*y;
}

// CHECK: void foo_grad(double x, double y, clad::array_ref<double> _d_x, clad::array_ref<double> _d_y) {
// CHECK-NEXT:     double _t0;
// CHECK-NEXT:     double _t1;
// CHECK-NEXT:     _t1 = x;
// CHECK-NEXT:     _t0 = y;
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r0 = 1 * _t0;
// CHECK-NEXT:         * _d_x += _r0;
// CHECK-NEXT:         double _r1 = _t1 * 1;
// CHECK-NEXT:         * _d_y += _r1;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: void foo_grad_enzyme(double x, double y, clad::array_ref<double> _d_x, clad::array_ref<double> _d_y) {
// CHECK-NEXT:     clad::EnzymeGradient<2> grad = __enzyme_autodiff_foo(foo, x, y);
// CHECK-NEXT:     * _d_x = grad.d_arr[0U];
// CHECK-NEXT:     * _d_y = grad.d_arr[1U];
// CHECK-NEXT: }

int main(){
    auto grad = clad::gradient(foo);    
    auto gradEnzyme = clad::gradient<clad::opts::use_enzyme>(foo);
}