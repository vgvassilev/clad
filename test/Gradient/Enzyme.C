
// RUN: %cladclang %s -lstdc++ -I%S/../../include -oEnzyme.out 2>&1 | FileCheck %s
// RUN: ./Enzyme.out
// CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

double f(double x, double y) { return x * y; }

// CHECK:  void f_grad_enzyme(double x, double y, clad::array_ref<double> _d_x, clad::array_ref<double> _d_y) {
// CHECK-NEXT:}

int main(){
     auto f_dx = clad::gradient<clad::opts::use_enzyme>(f);
}