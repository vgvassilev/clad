// RUN: %cladnumdiffclang -Xclang -plugin-arg-clad -Xclang -fprint-num-diff-errors %s -I%S/../../include -oPrintErrorNumDiff.out 2>&1 | FileCheck -check-prefix=CHECK %s
// -Xclang -verify 2>&1 RUN: ./PrintErrorNumDiff.out | FileCheck -check-prefix=CHECK-EXEC %s

//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

#include <cmath>

extern "C" int printf(const char* fmt, ...);

double test_1(double x){
   return tanh(x);
}

//CHECK: warning: Falling back to numerical differentiation for 'tanh' since no suitable overload was found and clad could not derive it. To disable this feature, compile your programs with -DCLAD_NO_NUM_DIFF.
//CHECK: void test_1_grad(double x, clad::array_ref<double> _d_x) {
//CHECK-NEXT:     double _t0;
//CHECK-NEXT:     _t0 = x;
//CHECK-NEXT:     {
//CHECK-NEXT:         double _r0 = 1 * numerical_diff::forward_central_difference(tanh, _t0, 0, 1, _t0);
//CHECK-NEXT:         * _d_x += _r0;
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
