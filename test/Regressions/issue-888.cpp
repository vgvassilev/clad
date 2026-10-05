
// RUN: %clang++ -fplugin=%clad_plugin_path -fplugin-arg-clad-print-derivatives %s -c -o /dev/null 2>&1 | FileCheck %s
// RUN: %cladclang %s -I%S/../../include -o issue-888.out
// RUN: %filecheck_exec %s --exec=issue-888.out

#include "clad/Differentiator/Differentiator.h"
#include <stdio.h>

double f1(double x) {
    double y = -x;
    return y;
}

double f2(double x) {
    x = -x;
    return x;
}

int main() {
    auto f1_grad = clad::gradient(f1);
    auto f2_grad = clad::gradient(f2);
    
    double dx1 = 3.0;
    f1_grad.execute(2.0, &dx1);
    printf("f1_grad: %f\n", dx1);
    // CHECK-EXEC: f1_grad: 2.000000

    double dx2 = 3.0;
    f2_grad.execute(2.0, &dx2);
    printf("f2_grad: %f\n", dx2);
    // CHECK-EXEC: f2_grad: 2.000000

    // CHECK: void f1_grad(double x, double *_d_x) {
    // CHECK: void f2_grad(double x, double *_d_x) {
    // CHECK: double _local_d_x = 0.;
    // CHECK: *_d_x += _local_d_x;
}
