// RUN: %cladclang %s -I%S/../../include -std=c++23 -oConstevalTest.out | %filecheck %s
// RUN: ./ConstevalTest.out | %filecheck_exec %s

#include <limits>

#include "clad/Differentiator/Differentiator.h"

#include "../TestUtils.h"


consteval double fn(double a, double b) {
    return (a+b)/2;
}

//CHECK: consteval double fn_darg0(double a, double b) {
//CHECK-NEXT:    double _d_a = 1;
//CHECK-NEXT:    double _d_b = 0;
//CHECK-NEXT:    double _t0 = (a + b);
//CHECK-NEXT:    return ((_d_a + _d_b) * 2 - _t0 * 0) / (2 * 2);
//CHECK-NEXT:}

consteval double mul(double a, double b, double c) {
     double val = 99.00;
     double result = val * a + 100 - b + c;
     return result;
}

//CHECK: consteval double mul_darg0(double a, double b, double c) {
//CHECK-NEXT:    double _d_a = 1;
//CHECK-NEXT:    double _d_b = 0;
//CHECK-NEXT:    double _d_c = 0;
//CHECK-NEXT:    double _d_val = 0.;
//CHECK-NEXT:    double val = 99.;
//CHECK-NEXT:    double _d_result = _d_val * a + val * _d_a + 0 - _d_b + _d_c;
//CHECK-NEXT:    double result = val * a + 100 - b + c;
//CHECK-NEXT:    return _d_result;
//CHECK-NEXT:}

consteval double fn_test() {
    auto dx = clad::differentiate(fn, "a");

    return dx.execute(4, 7);
}

consteval double mul_test() {
    auto dx = clad::differentiate(mul, "a");

    return dx.execute(5, 6, 10);
}

int main() {
    constexpr double fn_result = fn_test();
    printf("%.2f\n", fn_result); // CHECK-EXEC: 0.50

    constexpr double mul_result = mul_test();
    printf("%.2f\n", mul_result); // CHECK-EXEC: 99.00
}
