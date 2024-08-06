// RUN: %cladnumdiffclang %s -I%S/../../include -oGradientMultiArg.out -Xclang -verify 2>&1 | FileCheck -check-prefix=CHECK %s
// RUN: ./GradientMultiArg.out | %filecheck_exec %s
// RUN: %cladnumdiffclang -Xclang -plugin-arg-clad -Xclang -enable-tbr %s -I%S/../../include -oGradientMultiArg.out -Xclang -verify
// RUN: ./GradientMultiArg.out | %filecheck_exec %s

//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

#include <cmath>
#include <algorithm>

double test_1(double x, double y){
  return std::hypot(x, y); // expected-warning {{function 'hypot' was not differentiated}}
  // expected-note@14 {{falling back to numerical differentiation}}
}
// CHECK: void test_1_grad(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r0 = 0;
// CHECK-NEXT:         double _r1 = 0;
// CHECK-NEXT:         double _grad0[2] = {0};
// CHECK-NEXT:         numerical_diff::central_difference(std::hypot, _grad0, 0, x, y);
// CHECK-NEXT:         _r0 += 1 * _grad0[0];
// CHECK-NEXT:         _r1 += 1 * _grad0[1];
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
