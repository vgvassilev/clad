// RUN: %cladnumdiffclang -Xclang -plugin-arg-clad -Xclang -fprint-num-diff-errors %s -I%S/../../include -oPrintErrorNumDiff.out -Xclang -verify 2>&1 | FileCheck -check-prefix=CHECK %s
// RUN: ./PrintErrorNumDiff.out | %filecheck_exec %s
// RUN: %cladnumdiffclang -Xclang -plugin-arg-clad -Xclang -fprint-num-diff-errors -Xclang -plugin-arg-clad -Xclang -enable-tbr %s -I%S/../../include -oPrintErrorNumDiff.out -Xclang -verify
// RUN: ./PrintErrorNumDiff.out | %filecheck_exec %s

//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

#include <cmath>

extern "C" int printf(const char* fmt, ...);

double test_1(double x){
  return tanh(x); // expected-warning {{function 'tanh' was not differentiated because}}
  // expected-note@15 {{falling back to numerical differentiation for 'tanh}}
}

//CHECK: void test_1_grad(double x, double *_d_x) {
//CHECK-NEXT:     {
//CHECK-NEXT:         double _r0 = 0;
//CHECK-NEXT:         _r0 += 1 * numerical_diff::forward_central_difference(tanh, x, 0, 1, x);
//CHECK-NEXT:         *_d_x += _r0;
//CHECK-NEXT:     }
//CHECK-NEXT: }


int main(){
    auto df = clad::gradient(test_1);
    double x = 0.05, dx = 0;
    df.execute(x, &dx);
    printf("Result is:%f", dx);
    //CHECK-EXEC: Error Report for parameter at position 0:
    //CHECK-EXEC: Error due to the five-point central difference is: 0.0000000000
    //CHECK-EXEC: Error due to function evaluation is: 0.0000000000
    //CHECK-EXEC: Result is:0.997504
}
