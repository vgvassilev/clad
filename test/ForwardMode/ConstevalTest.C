// RUN: %cladclang %s -I%S/../../include -std=c++23 -oConstevalTest.out | %filecheck %s
// RUN: ./ConstevalTest.out | %filecheck_exec %s
// UNSUPPORTED: clang-14, clang-15, clang-16

#include "clad/Differentiator/Differentiator.h"

// A run-time function that calls a consteval one: half() is folded to a
// constant, whose derivative is 0. Defined before the consteval functions
// below, which are differentiated at compile time.
consteval double half() { return 0.5; }
double scaleHalf(double x) { return x * half(); }

consteval double fn(double x, double y) {
    return (x+y)/2;
}

//CHECK: consteval double fn_darg0(double x, double y) {
//CHECK-NEXT:    double _d_x = 1;
//CHECK-NEXT:    double _d_y = 0;
//CHECK-NEXT:    return ((_d_x + _d_y) * 2) / 4.;
//CHECK-NEXT:    }

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
//CHECK-NEXT:    double _d_result = _d_val * a + val * _d_a - _d_b + _d_c;
//CHECK-NEXT:    double result = val * a + 100 - b + c;
//CHECK-NEXT:    return _d_result;
//CHECK-NEXT:    }

consteval double fn_test() {
    auto dx = clad::differentiate(fn, "x");

    return dx.execute(4, 7);
}

consteval double mul_test() {
    auto dx = clad::differentiate(mul, "a");

    return dx.execute(5, 6, 10);
}

//CHECK: double scaleHalf_darg0(double x) {
//CHECK-NEXT:    double _d_x = 1;
//CHECK-NEXT:    double _t0 = half();
//CHECK-NEXT:    return _d_x * _t0;
//CHECK-NEXT: }

int main() {
    constexpr double fn_result = fn_test();
    printf("%.2f\n", fn_result); // CHECK-EXEC: 0.50

    constexpr double mul_result = mul_test();
    printf("%.2f\n", mul_result); // CHECK-EXEC: 99.00

    auto d_scaleHalf = clad::differentiate(scaleHalf, "x");
    printf("%.2f\n", d_scaleHalf.execute(3)); // CHECK-EXEC: 0.50
}
