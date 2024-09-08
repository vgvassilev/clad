// RUN: %cladclang %s -I%S/../../include -oconstexprTest.out | %filecheck %s
// RUN: ./constexprTest.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"

#include "../TestUtils.h"


constexpr double fn(double a, double b) {
    return (a+b)/2;
}

//CHECK: constexpr double fn_darg0(double a, double b) {
//CHECK-NEXT:    double _d_a = 1;
//CHECK-NEXT:    double _d_b = 0;
//CHECK-NEXT:    double _t0 = (a + b);
//CHECK-NEXT:    return ((_d_a + _d_b) * 2 - _t0 * 0) / (2 * 2);
//CHECK-NEXT:}

constexpr double mul(double a, double b, double c) {
     double val = 99.00;
     double result = val * a + 100 - b + c;
     return result;
}

//CHECK: constexpr double mul_darg0(double a, double b, double c) {
//CHECK-NEXT:    double _d_a = 1;
//CHECK-NEXT:    double _d_b = 0;
//CHECK-NEXT:    double _d_c = 0;
//CHECK-NEXT:    double _d_val = 0.;
//CHECK-NEXT:    double val = 99.;
//CHECK-NEXT:    double _d_result = _d_val * a + val * _d_a + 0 - _d_b + _d_c;
//CHECK-NEXT:    double result = val * a + 100 - b + c;
//CHECK-NEXT:    return _d_result;
//CHECK-NEXT:}

int main() {
    INIT_DIFFERENTIATE(fn,"a");
    INIT_DIFFERENTIATE(mul, "a");

    TEST_DIFFERENTIATE(fn, 4, 7); // CHECK-EXEC: {0.50}
    TEST_DIFFERENTIATE(mul, 5, 6, 10); // CHECK-EXEC: {99.00}
}
