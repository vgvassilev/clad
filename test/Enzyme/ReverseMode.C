// RUN: %cladclang %s -I%S/../../include -oReverseMode.out | FileCheck %s
// RUN: ./ReverseMode.out | FileCheck -check-prefix=CHECK-EXEC %s
// CHECK-NOT: {{.*error|warning|note:.*}}
// REQUIRES: Enzyme

#include "clad/Differentiator/Differentiator.h"

double f(double* arr) { return arr[0] * arr[1]; }

// CHECK: void f_grad_enzyme(double *arr, clad::array_ref<double> _d_arr) {
// CHECK-NEXT:    double *d_arr = _d_arr.ptr();
// CHECK-NEXT:    __enzyme_autodiff_f(f, arr, d_arr);
// CHECK-NEXT:}

int main() {
  auto f_grad = clad::gradient<clad::opts::use_enzyme>(f);
  double v[2] = {3, 4};
  double g[2] = {0};
  f_grad.execute(v, g);
  printf("d_x = %.2f, d_y = %.2f", g[0], g[1]); // CHECK-EXEC: d_x = 4.00, d_y = 3.00
}
