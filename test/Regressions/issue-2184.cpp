// RUN: %cladclang -std=c++17 -I%S/../../include %s -o %t 2>&1
// RUN: %t | %filecheck_exec %s
// CHECK-EXEC: J1 = {{\[\[}}6.0, 1.0], [0.0, 5.0]]
// CHECK-EXEC: J2 row0 = {{\[}}10.0, 20.0, 0.0, 0.0, 0.0]
// CHECK-EXEC: J2 row1 = {{\[}}0.0, 0.0, 20.0, 30.0, 1.0]
// CHECK-EXEC: J3 row0 = {{\[}}10.0, 20.0, 0.0, 0.0, 0.0]
// CHECK-EXEC: J3 row1 = {{\[}}0.0, 0.0, 20.0, 30.0, 1.0]
// CHECK-EXEC: ALL TESTS PASSED!

// Issue #2184: clad::jacobian with an argument list still needs a full-width matrix.
// 1. Differentiating an argument subset should size the Jacobian matrix columns
//    strictly for the requested independent variables, not all parameters.
// 2. Differentiating an array parameter (const or non-const) should treat it as an
//    independent input variable rather than an output parameter receiving a derivative.

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>
#include <cassert>
#include <cmath>

void f_scalar_subset(double a, double b, double c, double out[2]) {
    out[0] = a * a + b;
    out[1] = b * c;
}

void residuals(const double p[5], double x, double y, double z, double out[2]) {
    out[0] = p[0] * x + p[1] * y;
    out[1] = p[2] * y + p[3] * z + p[4];
}

void residuals_nonconst(double p[5], double x, double y, double z, double out[2]) {
    out[0] = p[0] * x + p[1] * y;
    out[1] = p[2] * y + p[3] * z + p[4];
}

int main() {
    // 1. Scalar subset test: f_scalar_subset with "a, b" -> Jacobian width should be 2, not 3
    {
        auto jac1 = clad::jacobian(f_scalar_subset, "a, b");
        clad::matrix<double> J1(2, 2);
        double out1[2] = {0, 0};
        jac1.execute(3.0, 4.0, 5.0, out1, &J1);
        // out[0] = a^2 + b -> d(out[0])/da = 2*a = 6, d(out[0])/db = 1
        // out[1] = b*c -> d(out[1])/da = 0, d(out[1])/db = c = 5
        printf("J1 = [[%.1f, %.1f], [%.1f, %.1f]]\n", J1[0][0], J1[0][1], J1[1][0], J1[1][1]);
        assert(std::abs(J1[0][0] - 6.0) < 1e-6);
        assert(std::abs(J1[0][1] - 1.0) < 1e-6);
        assert(std::abs(J1[1][0] - 0.0) < 1e-6);
        assert(std::abs(J1[1][1] - 5.0) < 1e-6);
    }

    // 2. Const array parameter test: residuals with "p"
    {
        auto jac2 = clad::jacobian(residuals, "p");
        clad::matrix<double> J2(2, 5);
        const double p[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
        double out2[2] = {0, 0};
        jac2.execute(p, 10.0, 20.0, 30.0, out2, &J2);
        // out[0] = p[0]*x + p[1]*y -> d/dp = [x, y, 0, 0, 0] = [10, 20, 0, 0, 0]
        // out[1] = p[2]*y + p[3]*z + p[4] -> d/dp = [0, 0, y, z, 1] = [0, 0, 20, 30, 1]
        printf("J2 row0 = [%.1f, %.1f, %.1f, %.1f, %.1f]\n", J2[0][0], J2[0][1], J2[0][2], J2[0][3], J2[0][4]);
        printf("J2 row1 = [%.1f, %.1f, %.1f, %.1f, %.1f]\n", J2[1][0], J2[1][1], J2[1][2], J2[1][3], J2[1][4]);
        assert(std::abs(J2[0][0] - 10.0) < 1e-6);
        assert(std::abs(J2[0][1] - 20.0) < 1e-6);
        assert(std::abs(J2[0][2] - 0.0) < 1e-6);
        assert(std::abs(J2[0][3] - 0.0) < 1e-6);
        assert(std::abs(J2[0][4] - 0.0) < 1e-6);

        assert(std::abs(J2[1][0] - 0.0) < 1e-6);
        assert(std::abs(J2[1][1] - 0.0) < 1e-6);
        assert(std::abs(J2[1][2] - 20.0) < 1e-6);
        assert(std::abs(J2[1][3] - 30.0) < 1e-6);
        assert(std::abs(J2[1][4] - 1.0) < 1e-6);
    }

    // 3. Non-const array parameter test: residuals_nonconst with "p"
    {
        auto jac3 = clad::jacobian(residuals_nonconst, "p");
        clad::matrix<double> J3(2, 5);
        double p[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
        double out3[2] = {0, 0};
        jac3.execute(p, 10.0, 20.0, 30.0, out3, &J3);
        printf("J3 row0 = [%.1f, %.1f, %.1f, %.1f, %.1f]\n", J3[0][0], J3[0][1], J3[0][2], J3[0][3], J3[0][4]);
        printf("J3 row1 = [%.1f, %.1f, %.1f, %.1f, %.1f]\n", J3[1][0], J3[1][1], J3[1][2], J3[1][3], J3[1][4]);
        assert(std::abs(J3[0][0] - 10.0) < 1e-6);
        assert(std::abs(J3[0][1] - 20.0) < 1e-6);
        assert(std::abs(J3[1][2] - 20.0) < 1e-6);
        assert(std::abs(J3[1][3] - 30.0) < 1e-6);
        assert(std::abs(J3[1][4] - 1.0) < 1e-6);
    }

    printf("ALL TESTS PASSED!\n");
    return 0;
}
