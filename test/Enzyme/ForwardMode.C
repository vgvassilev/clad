// RUN: %cladclang %s -I%S/../../include -oEnzyme.out 2>&1 | %filecheck %s
// RUN: ./Enzyme.out
// CHECK-NOT: {{.*error|warning|note:.*}}
// REQUIRES: Enzyme
// XFAIL:*
// Forward mode is not implemented yet

#include "clad/Differentiator/Differentiator.h"

double f(double x, double y) { return x * y; }

// CHECK:     double f_darg0_enzyme(double x, double y) {
// CHECK-NEXT:}

int main(){
     auto f_dx = clad::differentiate<clad::opts::use_enzyme>(f, "x");
}