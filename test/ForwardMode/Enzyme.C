// RUN: %cladclang %s -lstdc++ -I%S/../../include -oEnzyme.out 2>&1 | FileCheck %s
// RUN: ./Enzyme.out
// CHECK-NOT: {{.*error|warning|note:.*}}
// XFAIL:*

#include "clad/Differentiator/Differentiator.h"

double f(double x, double y) { return x * y; }

// CHECK:     void f_diff_enzyme() {
// CHECK-NEXT:}

int main(){
     auto f_dx = clad::differentiate<clad::opts::use_enzyme>(f, "x");
}