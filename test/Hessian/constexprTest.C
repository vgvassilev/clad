// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr %s -I%S/../../include -std=c++14 -oconstexprTest.out 2>&1 | %filecheck %s
// RUN: ./constexprTest.out | %filecheck_exec %s
// RUN: %cladclang %s -I%S/../../include -std=c++14 -oconstexprTest.out
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

//CHECK: constexpr void fn_hessian(double x, double y, double *hessianMatrix) {
//CHECK-NEXT:    fn_darg0_grad(x, y, hessianMatrix + {{0U|0UL|0ULL}}, hessianMatrix + {{1U|1UL|1ULL}});
//CHECK-NEXT:    fn_darg1_grad(x, y, hessianMatrix + {{2U|2UL|2ULL}}, hessianMatrix + {{3U|3UL|3ULL}});
//CHECK-NEXT:}

constexpr double g(double i, double j[2]) { return i * (j[0] + j[1]); }

//CHECK: constexpr void g_hessian(double i, double j[2], double *hessianMatrix) {
//CHECK-NEXT:    g_darg0_grad(i, j, hessianMatrix + {{0U|0UL|0ULL}}, hessianMatrix + {{1U|1UL|1ULL}});
//CHECK-NEXT:    g_darg1_0_grad(i, j, hessianMatrix + {{3U|3UL|3ULL}}, hessianMatrix + {{4U|4UL|4ULL}});
//CHECK-NEXT:    g_darg1_1_grad(i, j, hessianMatrix + {{6U|6UL|6ULL}}, hessianMatrix + {{7U|7UL|7ULL}});
//CHECK-NEXT:}

int main() {
    
    INIT_HESSIAN(fn);
    INIT_HESSIAN(g, "i, j[0:1]");

    TEST_HESSIAN(fn, 2, 2, 3, mat_ref_f); // CHECK-EXEC: {0.00, 1.00, 1.00, 0.00}
    TEST_HESSIAN(g, 2, 2, j, mat_ref_f1); // CHECK-EXEC: {0.00, 1.00, 1.00, 1.00, 0.00, 0.00, 1.00, 0.00, 0.00}
}
