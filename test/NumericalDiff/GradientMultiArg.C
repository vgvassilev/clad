// RUN: %cladnumdiffclang %s -I%S/../../include -oGradientMultiArg.out 2>&1 | FileCheck -check-prefix=CHECK %s
// RUN: ./GradientMultiArg.out | FileCheck -check-prefix=CHECK-EXEC %s

//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

#include <cmath>
#include <algorithm>

double test_1(double x, double y){
   return std::hypot(x, y);
}
// CHECK: warning: Falling back to numerical differentiation for 'hypot' since no suitable overload was found and clad could not derive it. To disable this feature, compile your programs with -DCLAD_NO_NUM_DIFF.
// CHECK: void test_1_grad(double x, double y, clad::array_ref<double> _d_x, clad::array_ref<double> _d_y) {
// CHECK-NEXT:     double _t0;
// CHECK-NEXT:     double _t1;
// CHECK-NEXT:     _t0 = x;
// CHECK-NEXT:     _t1 = y;
// CHECK-NEXT:     {
// CHECK-NEXT:         double _grad0 = 0.;
// CHECK-NEXT:         double _grad1 = 0.;
// CHECK-NEXT:         clad::tape<clad::array_ref<double> > _t2 = {};
// CHECK-NEXT:         clad::push(_t2, &_grad0);
// CHECK-NEXT:         clad::push(_t2, &_grad1);
// CHECK-NEXT:         numerical_diff::central_difference(std::hypot, _t2, 0, _t0, _t1);
// CHECK-NEXT:         double _r0 = 1 * _grad0;
// CHECK-NEXT:         * _d_x += _r0;
// CHECK-NEXT:         double _r1 = 1 * _grad1;
// CHECK-NEXT:         * _d_y += _r1;
// CHECK-NEXT:     }
// CHECK-NEXT: }


int main(){
    auto df = clad::gradient(test_1);
    double dx = 0, dy = 0;
    df.execute(3, 4, &dx, &dy);
    printf("Result is = %f\n", dx); // CHECK-EXEC: Result is = 0.600000
    printf("Result is = %f\n", dy); // CHECK-EXEC: Result is = 0.800000
}
