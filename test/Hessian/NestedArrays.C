// RUN: %cladclang %s -I%S/../../include -o NestedArrays.out 2>&1 | %filecheck %s
// RUN: ./NestedArrays.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"
#include <stdio.h>
#include <math.h>

double f0_inner(double* params) {
    return params[0] * params[1] * params[2]
         + params[3] * params[3]
         + params[4] * params[0];
}

double f0_outer(double* params) {
    return f0_inner(params) + sin(params[2]) - exp(params[4]);
}

double f1_inner(double* p) {
    return p[0] * p[1];
}

double f1_outer(double x, double y) {
    double arr[2] = {x, y};
    return f1_inner(arr);
}

double f2_inner(double* p, double x) {
    return p[0] * x * x;
}

double f2_outer(double x) {
    double arr[2] = {2.0, 3.0};
    return f2_inner(arr, x);
}

// CHECK: clad::ValueAndPushforward<double, double> f0_inner_pushforward(double *params, double *_d_params) {
// CHECK-NEXT:     double {{_t[0-9]+}} = params[0] * params[1];
// CHECK-NEXT:     return {{{_t[0-9]+}} * params[2] + params[3] * params[3] + params[4] * params[0], (_d_params[0] * params[1] + params[0] * _d_params[1]) * params[2] + {{_t[0-9]+}} * _d_params[2] + _d_params[3] * params[3] + params[3] * _d_params[3] + _d_params[4] * params[0] + params[4] * _d_params[0]};
// CHECK-NEXT: }

// CHECK: clad::ValueAndPushforward<double, double> f1_inner_pushforward(double *p, double *_d_p) {
// CHECK-NEXT:     return {p[0] * p[1], _d_p[0] * p[1] + p[0] * _d_p[1]};
// CHECK-NEXT: }

// CHECK: clad::ValueAndPushforward<double, double> f2_inner_pushforward(double *p, double x, double *_d_p, double _d_x) {
// CHECK-NEXT:     double {{_t[0-9]+}} = p[0] * x;
// CHECK-NEXT:     return {{{_t[0-9]+}} * x, (_d_p[0] * x + p[0] * _d_x) * x + {{_t[0-9]+}} * _d_x};
// CHECK-NEXT: }

// CHECK: void f0_inner_pushforward_pullback(double *params, double *_d_params, clad::ValueAndPushforward<double, double> _d_y, double *_d_params0, double *_d_d_params) {
// CHECK-NEXT:     double {{_d_t[0-9]+}} = {{0\.|0\.0|0}};
// CHECK-NEXT:     double {{_t[0-9]+}} = params[0] * params[1];
// CHECK-NEXT:     {
// CHECK-NEXT:         {{_d_t[0-9]+}} += _d_y.value * params[2];
// CHECK-NEXT:         if (_d_params0)
// CHECK-NEXT:             _d_params0[2] += {{_t[0-9]+}} * _d_y.value;
// CHECK-NEXT:         if (_d_params0)
// CHECK-NEXT:             _d_params0[3] += _d_y.value * params[3];
// CHECK-NEXT:         if (_d_params0)
// CHECK-NEXT:             _d_params0[3] += params[3] * _d_y.value;
// CHECK-NEXT:         if (_d_params0)
// CHECK-NEXT:             _d_params0[4] += _d_y.value * params[0];
// CHECK-NEXT:         if (_d_params0)
// CHECK-NEXT:             _d_params0[0] += params[4] * _d_y.value;
// CHECK-NEXT:         if (_d_d_params)
// CHECK-NEXT:             _d_d_params[0] += _d_y.pushforward * params[2] * params[1];
// CHECK-NEXT:         if (_d_params0)
// CHECK-NEXT:             _d_params0[1] += _d_params[0] * _d_y.pushforward * params[2];
// CHECK-NEXT:         if (_d_params0)
// CHECK-NEXT:             _d_params0[0] += _d_y.pushforward * params[2] * _d_params[1];
// CHECK-NEXT:         if (_d_d_params)
// CHECK-NEXT:             _d_d_params[1] += params[0] * _d_y.pushforward * params[2];
// CHECK-NEXT:         if (_d_params0)
// CHECK-NEXT:             _d_params0[2] += (_d_params[0] * params[1] + params[0] * _d_params[1]) * _d_y.pushforward;
// CHECK-NEXT:         {{_d_t[0-9]+}} += _d_y.pushforward * _d_params[2];
// CHECK-NEXT:         if (_d_d_params)
// CHECK-NEXT:             _d_d_params[2] += {{_t[0-9]+}} * _d_y.pushforward;
// CHECK-NEXT:         if (_d_d_params)
// CHECK-NEXT:             _d_d_params[3] += _d_y.pushforward * params[3];
// CHECK-NEXT:         if (_d_params0)
// CHECK-NEXT:             _d_params0[3] += _d_params[3] * _d_y.pushforward;
// CHECK-NEXT:         if (_d_params0)
// CHECK-NEXT:             _d_params0[3] += _d_y.pushforward * _d_params[3];
// CHECK-NEXT:         if (_d_d_params)
// CHECK-NEXT:             _d_d_params[3] += params[3] * _d_y.pushforward;
// CHECK-NEXT:         if (_d_d_params)
// CHECK-NEXT:             _d_d_params[4] += _d_y.pushforward * params[0];
// CHECK-NEXT:         if (_d_params0)
// CHECK-NEXT:             _d_params0[0] += _d_params[4] * _d_y.pushforward;
// CHECK-NEXT:         if (_d_params0)
// CHECK-NEXT:             _d_params0[4] += _d_y.pushforward * _d_params[0];
// CHECK-NEXT:         if (_d_d_params)
// CHECK-NEXT:             _d_d_params[0] += params[4] * _d_y.pushforward;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         if (_d_params0)
// CHECK-NEXT:             _d_params0[0] += {{_d_t[0-9]+}} * params[1];
// CHECK-NEXT:         if (_d_params0)
// CHECK-NEXT:             _d_params0[1] += params[0] * {{_d_t[0-9]+}};
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: void f1_inner_pushforward_pullback(double *p, double *_d_p, clad::ValueAndPushforward<double, double> _d_y, double *_d_p0, double *_d_d_p) {
// CHECK-NEXT:     {
// CHECK-NEXT:         if (_d_p0)
// CHECK-NEXT:             _d_p0[0] += _d_y.value * p[1];
// CHECK-NEXT:         if (_d_p0)
// CHECK-NEXT:             _d_p0[1] += p[0] * _d_y.value;
// CHECK-NEXT:         if (_d_d_p)
// CHECK-NEXT:             _d_d_p[0] += _d_y.pushforward * p[1];
// CHECK-NEXT:         if (_d_p0)
// CHECK-NEXT:             _d_p0[1] += _d_p[0] * _d_y.pushforward;
// CHECK-NEXT:         if (_d_p0)
// CHECK-NEXT:             _d_p0[0] += _d_y.pushforward * _d_p[1];
// CHECK-NEXT:         if (_d_d_p)
// CHECK-NEXT:             _d_d_p[1] += p[0] * _d_y.pushforward;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: void f2_inner_pushforward_pullback(double *p, double x, double *_d_p, double _d_x, clad::ValueAndPushforward<double, double> _d_y, double *_d_p0, double *_d_x0, double *_d_d_p, double *_d_d_x) {
// CHECK-NEXT:     double {{_d_t[0-9]+}} = {{0\.|0\.0|0}};
// CHECK-NEXT:     double {{_t[0-9]+}} = p[0] * x;
// CHECK-NEXT:     {
// CHECK-NEXT:         {{_d_t[0-9]+}} += _d_y.value * x;
// CHECK-NEXT:         *_d_x0 += {{_t[0-9]+}} * _d_y.value;
// CHECK-NEXT:         if (_d_d_p)
// CHECK-NEXT:             _d_d_p[0] += _d_y.pushforward * x * x;
// CHECK-NEXT:         *_d_x0 += _d_p[0] * _d_y.pushforward * x;
// CHECK-NEXT:         if (_d_p0)
// CHECK-NEXT:             _d_p0[0] += _d_y.pushforward * x * _d_x;
// CHECK-NEXT:         *_d_d_x += p[0] * _d_y.pushforward * x;
// CHECK-NEXT:         *_d_x0 += (_d_p[0] * x + p[0] * _d_x) * _d_y.pushforward;
// CHECK-NEXT:         {{_d_t[0-9]+}} += _d_y.pushforward * _d_x;
// CHECK-NEXT:         *_d_d_x += {{_t[0-9]+}} * _d_y.pushforward;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         if (_d_p0)
// CHECK-NEXT:             _d_p0[0] += {{_d_t[0-9]+}} * x;
// CHECK-NEXT:         *_d_x0 += p[0] * {{_d_t[0-9]+}};
// CHECK-NEXT:     }
// CHECK-NEXT: }

int main() {
    double p5[5] = {1.5, 2.0, -0.5, 3.0, 1.0};
    
    auto h_0 = clad::hessian(f0_outer, "params[0:4]");
    auto h_1 = clad::hessian(f1_outer);
    auto h_2 = clad::hessian(f2_outer);

    double H_0[25] = {0};
    double H_1[4] = {0};
    double H_2[1] = {0};
    h_0.execute(p5, H_0);
    h_1.execute(2.0, 3.0, H_1);
    h_2.execute(5.0, H_2);

    printf("%.2f\n", H_0[0]);
    // CHECK-EXEC: 0.00
    
    printf("%.2f\n", H_0[1]);
    // CHECK-EXEC: -0.50
    
    printf("%.4f\n", H_0[12]);
    // CHECK-EXEC: 0.4794
    
    printf("%.4f\n", H_0[24]);
    // CHECK-EXEC: -2.7183

    printf("%.2f\n", H_1[0]);
    // CHECK-EXEC: 0.00

    printf("%.2f\n", H_1[1]);
    // CHECK-EXEC: 1.00

    printf("%.2f\n", H_1[2]);
    // CHECK-EXEC: 1.00

    printf("%.2f\n", H_1[3]);
    // CHECK-EXEC: 0.00

    printf("%.2f\n", H_2[0]);
    // CHECK-EXEC: 4.00

    return 0;
}