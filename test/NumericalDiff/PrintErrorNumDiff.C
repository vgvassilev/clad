// RUN: %cladnumdiffclang -Xclang -plugin-arg-clad -Xclang -fprint-num-diff-errors %s -I%S/../../include -oPrintErrorNumDiff.out -Xclang -verify 2>&1 | FileCheck -check-prefix=CHECK %s
// RUN: ./PrintErrorNumDiff.out | %filecheck_exec %s
// RUN: %cladnumdiffclang -Xclang -plugin-arg-clad -Xclang -fprint-num-diff-errors -Xclang -plugin-arg-clad -Xclang -enable-tbr %s -I%S/../../include -oPrintErrorNumDiff.out -Xclang -verify
// RUN: ./PrintErrorNumDiff.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"

#include <cmath>

extern "C" int printf(const char* fmt, ...);

double test_1(double x){
  return std::tgamma(x); // expected-warning {{function 'tgamma' was not differentiated because}}
  // expected-note@13 {{falling back to numerical differentiation for 'tgamma}}
}

//CHECK: void test_1_grad(double x, double *_d_x) {
//CHECK-NEXT:     {
//CHECK-NEXT:         double _r0 = 0.;
//CHECK-NEXT:         _r0 += 1 * numerical_diff::forward_central_difference(std::tgamma, x, 0, 1, x);
//CHECK-NEXT:         *_d_x += _r0;
//CHECK-NEXT:     }
//CHECK-NEXT: }


int main(){
    auto df = clad::gradient(test_1);
    double x = 0.5, dx = 0;
    df.execute(x, &dx);
    printf("Result is:%f", dx);
    //CHECK-EXEC: Error Report for parameter at position 0:
    //CHECK-EXEC: Error due to the five-point central difference is: 0.0000000005
    //CHECK-EXEC: Error due to function evaluation is: 0.0000000000
    //CHECK-EXEC: Result is:-3.480231
}
