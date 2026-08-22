// RUN: %cladclang %s -I%S/../../include -oconstexprTest.out 2>&1 | %filecheck %s
// RUN: ./constexprTest.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr %s -I%S/../../include -std=c++14 -oconstexprTest.out
// RUN: ./constexprTest.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"
#include <iostream>
#include "../TestUtils.h"

double mat_ref[4];
clad::array_ref<double> mat_ref_f(mat_ref, 4);

double j[2] = {3, 4};
double mat_ref1[9];
clad::array_ref<double>mat_ref_f1(mat_ref1, 9);

constexpr double fn(double x, double y) { return x * y; }

//CHECK:inline constexpr void fn_hessian(double x, double y, double *hessianMatrix) {
//CHECK-NEXT:    clad::ValueAndPushforward<double, double> _d_y{0., 0.};
//CHECK-NEXT:    _d_y.pushforward = 1.;
//CHECK-NEXT:    double _d_x(0.);
//CHECK-NEXT:    double _d_y0(0.);
//CHECK-NEXT:    _d_x = 1.;
//CHECK-NEXT:    fn_pushforward_pullback(x, y, _d_x, _d_y0, _d_y, hessianMatrix + {{0U|0UL|0ULL}}, hessianMatrix + {{1U|1UL|1ULL}});
//CHECK-NEXT:    _d_x = 0.;
//CHECK-NEXT:    _d_y0 = 1.;
//CHECK-NEXT:    fn_pushforward_pullback(x, y, _d_x, _d_y0, _d_y, hessianMatrix + {{2U|2UL|2ULL}}, hessianMatrix + {{3U|3UL|3ULL}});
//CHECK-NEXT:    _d_y0 = 0.;
//CHECK-NEXT:}

constexpr double g(double i, double j[2]) { return i * (j[0] + j[1]); }

//CHECK:inline constexpr void g_hessian(double i, double j[2], double *hessianMatrix) {
//CHECK-NEXT:    clad::ValueAndPushforward<double, double> _d_y{0., 0.};
//CHECK-NEXT:    _d_y.pushforward = 1.;
//CHECK-NEXT:    double _d_i(0.);
//CHECK-NEXT:    double _d_j[2]{0};
//CHECK-NEXT:    _d_i = 1.;
//CHECK-NEXT:    g_pushforward_pullback(i, j, _d_i, _d_j, _d_y, hessianMatrix + {{0U|0UL|0ULL}}, hessianMatrix + {{1U|1UL|1ULL}});
//CHECK-NEXT:    _d_i = 0.;
//CHECK-NEXT:    _d_j[{{0U|0UL|0ULL}}] = 1.;
//CHECK-NEXT:    g_pushforward_pullback(i, j, _d_i, _d_j, _d_y, hessianMatrix + {{3U|3UL|3ULL}}, hessianMatrix + {{4U|4UL|4ULL}});
//CHECK-NEXT:    _d_j[{{0U|0UL|0ULL}}] = 0.;
//CHECK-NEXT:    _d_j[{{1U|1UL|1ULL}}] = 1.;
//CHECK-NEXT:    g_pushforward_pullback(i, j, _d_i, _d_j, _d_y, hessianMatrix + {{6U|6UL|6ULL}}, hessianMatrix + {{7U|7UL|7ULL}});
//CHECK-NEXT:    _d_j[{{1U|1UL|1ULL}}] = 0.;
//CHECK-NEXT:}

int main() {
    
    INIT_HESSIAN(fn);
    INIT_HESSIAN(g, "i, j[0:1]");

    TEST_HESSIAN(fn, 2, 2, 3, mat_ref_f); // CHECK-EXEC: {0.00, 1.00, 1.00, 0.00}
    TEST_HESSIAN(g, 2, 2, j, mat_ref_f1); // CHECK-EXEC: {0.00, 1.00, 1.00, 1.00, 0.00, 0.00, 1.00, 0.00, 0.00}
}
