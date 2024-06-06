// RUN: %cladnumdiffclang %s  -I%S/../../include -oInterfaceCompatibility.out 2>&1 | %filecheck %s
// RUN: ./InterfaceCompatibility.out | %filecheck_exec %s
// RUN: %cladnumdiffclang -Xclang -plugin-arg-clad -Xclang -enable-tbr %s  -I%S/../../include -oInterfaceCompatibility.out
// RUN: ./InterfaceCompatibility.out | %filecheck_exec %s

//CHECK-NOT: {{.*error|warning|note:.*}}
#include "clad/Differentiator/Differentiator.h"
#include <cmath>

double f1(double* x, double y) {
    y = x[1];
    return y;
}
double f2(double x[2], int* y) {
    return *y * x[0];
}

int main() {
    double x[2] = {2, 5}, dx[2] = {0}, dy = 0;

    auto df1 = clad::gradient(f1);
    df1.execute(x, 5, dx, &dy);
    printf("{%.2f, %.2f, %.2f}\n", dx[0], dx[1], dy);  // CHECK-EXEC: {0.00, 1.00, 0.00}

    dx[0] = dx[1] = 0;
    int y[] = {9}, dy2[] = {0};
    auto df2 = clad::gradient(f2);
    df2.execute(x, y, dx, dy2);
    printf("{%.2f, %.2f, %.2f}\n", dx[0], dx[1], (double)*dy2);  // CHECK-EXEC: {9.00, 0.00, 2.00}
}