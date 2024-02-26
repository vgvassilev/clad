// RUN: %cladclang %s -I%S/../../include -oDifferentCladEnzymeDerivatives.out | FileCheck %s
// RUN: ./DifferentCladEnzymeDerivatives.out
// CHECK-NOT: {{.*error|warning|note:.*}}
// REQUIRES: Enzyme

#include "clad/Differentiator/Differentiator.h"


double foo(double x, double y){
    return x*y;
}

// CHECK: void foo_grad(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:     goto _label0;
// CHECK-NEXT:   _label0:
// CHECK-NEXT:     {
// CHECK-NEXT:         *_d_x += 1 * y;
// CHECK-NEXT:         *_d_y += x * 1;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: void foo_grad_enzyme(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:     clad::EnzymeGradient<2> grad = __enzyme_autodiff_foo(foo, x, y);
// CHECK-NEXT:     *_d_x = grad.d_arr[0U];
// CHECK-NEXT:     *_d_y = grad.d_arr[1U];
// CHECK-NEXT: }

int main(){
    auto grad = clad::gradient(foo);
    auto gradEnzyme = clad::gradient<clad::opts::use_enzyme>(foo);
}
