// RUN: %cladclang %s -I%S/../../include -oDotProductIdentity.out
// RUN: ./DotProductIdentity.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -enable-ua %s \
// RUN:     -I%S/../../include -oDotProductIdentity_ua.out
// RUN: ./DotProductIdentity_ua.out | %filecheck_exec %s
// CHECK-EXEC-NOT: FAIL

#include "clad/Differentiator/Differentiator.h"
#include "../TestUtils.h"
#include <cmath>
#include <cstdio>

double h3(double a, double b, double c) {
    return a * b + std::sin(c) * a;
}

double q(const double* x, int n) {
    double r = 0;
    for (int i = 0; i < n; ++i) r += x[i] * x[i] * x[i];
    return r;
}

int main() {
    double a = 0.7, b = -1.3, c = 2.1;
    double da = 0, db = 0, dc = 0;
    auto g3 = clad::gradient(h3);
    g3.execute(a, b, c, &da, &db, &dc);

    double pa = clad::differentiate(h3, "a").execute(a, b, c);
    double pb = clad::differentiate(h3, "b").execute(a, b, c);
    double pc = clad::differentiate(h3, "c").execute(a, b, c);

    double xdot[3] = {0.31, -0.52, 0.87};
    double Xb[3]   = {da, db, dc};
    double Yd      = xdot[0]*pa + xdot[1]*pb + xdot[2]*pc;
    printf("%d\n", test_utils::almost_equal(test_utils::dot(xdot, Xb, 3), Yd) ? 1 : 0);
    // CHECK-EXEC: 1

    const int n = 5;
    double x[n], dx[n], fdg[n];
    for (int i = 0; i < n; ++i) { x[i] = 0.5 + 0.3*i; dx[i] = 0; }
    auto gq = clad::gradient(q, "x");
    gq.execute(x, n, dx);
    test_utils::fd_gradient(q, x, n, fdg);
    printf("%d\n", test_utils::max_abs_diff(dx, fdg, n) < 1e-4 ? 1 : 0);
    // CHECK-EXEC: 1

    double bad[n];
    for (int i = 0; i < n; ++i) bad[i] = dx[i] + 1.0;
    printf("%d\n", test_utils::max_abs_diff(bad, fdg, n) < 1e-4 ? 1 : 0);
    // CHECK-EXEC: 0

    return 0;
}
