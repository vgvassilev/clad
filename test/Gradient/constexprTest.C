// RUN: %cladclang %s -I%S/../../include -oconstexprTest.out | %filecheck %s
// RUN: ./constexprTest.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -enable-tbr %s -I%S/../../include -oconstexprTest.out
// RUN: ./constexprTest.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"

#include "../TestUtils.h"

constexpr double mul (double a, double b, double c) {
    double result = a*b*c;
    return result;
}

//CHECK: constexpr void mul_grad(double a, double b, double c, double *_d_a, double *_d_b, double *_d_c) {
//CHECK-NEXT:    double _d_result = 0;
//CHECK-NEXT:    double result = a * b * c;
//CHECK-NEXT:    _d_result += 1;
//CHECK-NEXT:    {
//CHECK-NEXT:        *_d_a += _d_result * c * b;
//CHECK-NEXT:        *_d_b += a * _d_result * c;
//CHECK-NEXT:        *_d_c += a * b * _d_result;
//CHECK-NEXT:    }
//CHECK-NEXT: }

constexpr double fn( double a, double b, double c) {
    double val = 98.00;
    double result = a * b / c * (a+b) * 100 +c;
    return result;
}

//CHECK: constexpr void fn_grad(double a, double b, double c, double *_d_a, double *_d_b, double *_d_c) {
//CHECK-NEXT:    double _d_val = 0;
//CHECK-NEXT:    double val = 98.;
//CHECK-NEXT:    double _d_result = 0;
//CHECK-NEXT:    double result = a * b / c * (a + b) * 100 + c;
//CHECK-NEXT:    _d_result += 1;
//CHECK-NEXT:    {
//CHECK-NEXT:        *_d_a += _d_result * 100 * (a + b) / c * b;
//CHECK-NEXT:        *_d_b += a * _d_result * 100 * (a + b) / c;
//CHECK-NEXT:        double _r0 = _d_result * 100 * (a + b) * -(a * b / (c * c));
//CHECK-NEXT:        *_d_c += _r0;
//CHECK-NEXT:        *_d_a += a * b / c * _d_result * 100;
//CHECK-NEXT:        *_d_b += a * b / c * _d_result * 100;
//CHECK-NEXT:        *_d_c += _d_result;
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
