// RUN: %cladnumdiffclang %s  -I%S/../../include -oInterfaceCompatibility.out 2>&1 | FileCheck %s
// RUN: ./InterfaceCompatibility.out | FileCheck -check-prefix=CHECK-EXEC %s
// RUN: %cladnumdiffclang -Xclang -plugin-arg-clad -Xclang -enable-tbr %s  -I%S/../../include -oInterfaceCompatibility.out
// RUN: ./InterfaceCompatibility.out | FileCheck -check-prefix=CHECK-EXEC %s

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
    clad::array_ref<double> dx_ref(dx, 2);
    clad::array_ref<double> dy_ref(&dy, 1);

    auto df1 = clad::gradient(f1);
    df1.execute(x, 5, dx_ref, dy_ref);
    printf("{%.2f, %.2f, %.2f}\n", dx_ref[0], dx_ref[1], *dy_ref);  // CHECK-EXEC: {0.00, 1.00, 0.00}

    dx_ref[0] = dx_ref[1] = 0;
    int y[] = {9}, dy2[] = {0};
    clad::array_ref<int> dy2_ref(dy2, 1);
    auto df2 = clad::gradient(f2);
    df2.execute(x, y, dx_ref, dy2_ref);
    printf("{%.2f, %.2f, %.2f}\n", dx_ref[0], dx_ref[1], (double)*dy2_ref);  // CHECK-EXEC: {9.00, 0.00, 2.00}
}