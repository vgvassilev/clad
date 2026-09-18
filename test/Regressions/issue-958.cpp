// RUN: %cladclang %s -I%S/../../include -o issue-958.out 2>&1 | %filecheck %s
// RUN: ./issue-958.out

#include "clad/Differentiator/Differentiator.h"

double fn(double x) {
    double __b = x;
    return __b;
}

// CHECK: double fn_darg0(double x) {
// CHECK-NEXT:    double _d_x = 1;
// CHECK-NEXT:    double _d___b = _d_x;
// CHECK-NEXT:    double __b = x;
// CHECK-NEXT:    return _d___b;
// CHECK-NEXT: }

int main() {
    auto df = clad::differentiate(fn, "x");
    return 0;
}
