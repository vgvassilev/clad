// RUN: %cladclang %s -I%S/../../include -otesthessUtility.out 2>&1 | %filecheck %s
// RUN: ./testhessUtility.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -enable-tbr %s -I%S/../../include -otesthessUtility.out
// RUN: ./testhessUtility.out | %filecheck_exec %s
// CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

#include "../TestUtils.h"

double mat_ref[4];
clad::array_ref<double> mat_ref_f(mat_ref, 4);

double j[2] = {3, 4};
double mat_ref1[9];
clad::array_ref<double>mat_ref_f1(mat_ref1, 9);

double fn(double x, double y) { return x * y; }

//CHECK: void fn_hessian(double x, double y, double *hessianMatrix) {
//CHECK-NEXT:    fn_darg0_grad(x, y, hessianMatrix + {{0U|0UL}}, hessianMatrix + {{1U|1UL}});
//CHECK-NEXT:    fn_darg1_grad(x, y, hessianMatrix + {{2U|2UL}}, hessianMatrix + {{3U|3UL}});
//CHECK-NEXT:}

double g(double i, double j[2]) { return i * (j[0] + j[1]); }

//CHECK: void g_hessian(double i, double j[2], double *hessianMatrix) {
//CHECK-NEXT:   g_darg0_grad(i, j, hessianMatrix + {{0U|0UL}}, hessianMatrix + {{1U|1UL}});
//CHECK-NEXT:   g_darg1_0_grad(i, j, hessianMatrix + {{3U|3UL}}, hessianMatrix + {{4U|4UL}});
//CHECK-NEXT:   g_darg1_1_grad(i, j, hessianMatrix + {{6U|6UL}}, hessianMatrix + {{7U|7UL}});
//CHECK-NEXT: }

int main() {

    INIT_HESSIAN(fn);
    INIT_HESSIAN(g, "i, j[0:1]");

    TEST_HESSIAN(fn, 2, 2, 3, mat_ref_f); // CHECK-EXEC: {0.00, 1.00, 1.00, 0.00}
    TEST_HESSIAN(g, 2, 2, j, mat_ref_f1); // CHECK-EXEC: {0.00, 1.00, 1.00, 1.00, 0.00, 0.00, 1.00, 0.00, 0.00}

}
