// RUN: %cladclang %s -I%S/../../include -oconstexprTest.out | FileCheck %s
// RUN: ./constexprTest.out | FileCheck -check-prefix=CHECK-EXEC %s

#include "clad/Differentiator/Differentiator.h"

#include "../TestUtils.h"

constexpr double mul (double a, double b, double c) {
    double result = a*b*c;
    return result;
}

//CHECK: constexpr void mul_grad(double a, double b, double c, clad::array_ref<double> _d_a, clad::array_ref<double> _d_b, clad::array_ref<double> _d_c) {
//CHECK-NEXT:    double _t0;
//CHECK-NEXT:    double _t1;
//CHECK-NEXT:    double _t2;
//CHECK-NEXT:    double _t3;
//CHECK-NEXT:    double _d_result = 0;
//CHECK-NEXT:    _t2 = a;
//CHECK-NEXT:    _t1 = b;
//CHECK-NEXT:    _t3 = _t2 * _t1;
//CHECK-NEXT:    _t0 = c;
//CHECK-NEXT:    double result = _t3 * _t0;
//CHECK-NEXT:    _d_result += 1;
//CHECK-NEXT:    {
//CHECK-NEXT:        double _r0 = _d_result * _t0;
//CHECK-NEXT:        double _r1 = _r0 * _t1;
//CHECK-NEXT:        * _d_a += _r1;
//CHECK-NEXT:        double _r2 = _t2 * _r0;
//CHECK-NEXT:        * _d_b += _r2;
//CHECK-NEXT:        double _r3 = _t3 * _d_result;
//CHECK-NEXT:        * _d_c += _r3;
//CHECK-NEXT:    }
//CHECK-NEXT:}

constexpr double fn( double a, double b, double c) {
    double val = 98.00;
    double result = a * b / c * (a+b) * 100 +c;
    return result;
}

//CHECK: constexpr void fn_grad(double a, double b, double c, clad::array_ref<double> _d_a, clad::array_ref<double> _d_b, clad::array_ref<double> _d_c) {
//CHECK-NEXT:    double _d_val = 0;
//CHECK-NEXT:    double _t0;
//CHECK-NEXT:    double _t1;
//CHECK-NEXT:    double _t2;
//CHECK-NEXT:    double _t3;
//CHECK-NEXT:    double _t4;
//CHECK-NEXT:    double _t5;
//CHECK-NEXT:    double _d_result = 0;
//CHECK-NEXT:    double val = 98.;
//CHECK-NEXT:    _t3 = a;
//CHECK-NEXT:    _t2 = b;
//CHECK-NEXT:    _t4 = _t3 * _t2;
//CHECK-NEXT:    _t1 = c;
//CHECK-NEXT:    _t5 = _t4 / _t1;
//CHECK-NEXT:    _t0 = (a + b);
//CHECK-NEXT:    double result = _t5 * _t0 * 100 + c;
//CHECK-NEXT:    _d_result += 1;
//CHECK-NEXT:    {
//CHECK-NEXT:        double _r0 = _d_result * 100;
//CHECK-NEXT:        double _r1 = _r0 * _t0;
//CHECK-NEXT:        double _r2 = _r1 / _t1;
//CHECK-NEXT:        double _r3 = _r2 * _t2;
//CHECK-NEXT:        * _d_a += _r3;
//CHECK-NEXT:        double _r4 = _t3 * _r2;
//CHECK-NEXT:        * _d_b += _r4;
//CHECK-NEXT:        double _r5 = _r1 * -_t4 / (_t1 * _t1);
//CHECK-NEXT:        * _d_c += _r5;
//CHECK-NEXT:        double _r6 = _t5 * _r0;
//CHECK-NEXT:        * _d_a += _r6;
//CHECK-NEXT:        * _d_b += _r6;
//CHECK-NEXT:        * _d_c += _d_result;
//CHECK-NEXT:    }
//CHECK-NEXT:}

double arr[3] = {};
double arr1[3] = {};
int main() {

    INIT_GRADIENT(mul);
    INIT_GRADIENT(fn);

    TEST_GRADIENT(mul, 3, 2, 3, 4, &arr[0], &arr[1], &arr[2]); // CHECK-EXEC: {12.00, 8.00, 6.00}
    TEST_GRADIENT(fn, 3, 4, 9, 10, &arr1[0], &arr1[1], &arr[2]); //CHECK-EXEC: {1530.00, 880.00, -467.00}

}
