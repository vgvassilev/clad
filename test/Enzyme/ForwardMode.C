// RUN: %cladclang %s -lstdc++ -I%S/../../include -oEnzyme.out 2>&1 | FileCheck %s
// RUN: ./Enzyme.out | FileCheck -check-prefix=CHECK-EXEC %s
// CHECK-NOT: {{.*error|warning|note:.*}}
// REQUIRES: Enzyme
// Forward mode is not implemented yet

#include "clad/Differentiator/Differentiator.h"

double f(double x, double y) {
      return x * y; 
}

// CHECK: double f_darg0_enzyme(double x, double y) {
// CHECK-NEXT:     double diff = __enzyme_fwddiff_f_x(f, x, 1., y, 0.);
// CHECK-NEXT:     return diff;
// CHECK-NEXT: }

int main(){
     auto f_dx = clad::differentiate<clad::opts::use_enzyme>(f, "x");
     double ans = f_dx.execute(3,4);
     printf("Ans = %.2f\n",ans); // CHECK-EXEC: Ans = 4.00
}