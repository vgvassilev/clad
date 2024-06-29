// RUN: %cladnumdiffclang %s -I%S/../../include -oGradientMultiArg.out 2>&1 | FileCheck -check-prefix=CHECK %s
// RUN: ./GradientMultiArg.out | %filecheck_exec %s
// RUN: %cladnumdiffclang -Xclang -plugin-arg-clad -Xclang -enable-tbr %s -I%S/../../include -oGradientMultiArg.out
// RUN: ./GradientMultiArg.out | %filecheck_exec %s

//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

#include <cmath>
#include <algorithm>

double test_1(double x, double y){
   return std::hypot(x, y);
}
// CHECK: warning: Falling back to numerical differentiation for 'hypot' since no suitable overload was found and clad could not derive it. To disable this feature, compile your programs with -DCLAD_NO_NUM_DIFF.
// CHECK: void test_1_pullback(double x, double y, double _d_y0, double *_d_x, double *_d_y) {
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r0 = 0;
// CHECK-NEXT:         double _r1 = 0;
// CHECK-NEXT:         double _grad0[2] = {0};
// CHECK-NEXT:         numerical_diff::central_difference(std::hypot, _grad0, 0, x, y);
// CHECK-NEXT:         _r0 += _d_y0 * _grad0[0];
// CHECK-NEXT:         _r1 += _d_y0 * _grad0[1];
// CHECK-NEXT:         *_d_x += _r0;
// CHECK-NEXT:         *_d_y += _r1;
// CHECK-NEXT:     }
// CHECK-NEXT: }


int main(){
    auto df = clad::gradient(test_1);
    double dx = 0, dy = 0;
    df.execute(3, 4, &dx, &dy);
    printf("Result is = %f\n", dx); // CHECK-EXEC: Result is = 0.600000
    printf("Result is = %f\n", dy); // CHECK-EXEC: Result is = 0.800000
}
