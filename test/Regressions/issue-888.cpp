
// RUN: %clang++ -fplugin=%clad_plugin_path -fplugin-arg-clad-print-derivatives %s -c -o /dev/null 2>&1 | FileCheck %s

#include "clad/Differentiator/Differentiator.h"

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
    // CHECK: void f1_grad(double x, double *_d_x) {
    // CHECK: void f2_grad(double x, double *_d_x) {
    // CHECK: double _local_d_x = 0.;
    // CHECK: *_d_x += _local_d_x;
}
