// RUN: %cladclang %s -I%S/../../include -fopenmp -oOpenMP.out 2>&1 | %filecheck %s

#include "clad/Differentiator/Differentiator.h"

double sum_of_squares(const double *x, int n) {
  double total = 0.0;

  #pragma omp parallel for reduction(+:total)
  for (int i = 0; i < n; i++) {
    total += x[i] * x[i];
  }

  return total;
}

// CHECK: double sum_of_squares_darg0_1(const double *x, int n) {
// CHECK-NEXT:   int _d_n = 0;
// CHECK-NEXT:   double _d_total = 0.;
// CHECK-NEXT:   double total = 0.;
// CHECK-NEXT:   #pragma omp parallel for reduction(+: _d_total,total)
// CHECK-NEXT:       for (int i = 0; i < n; i++) {
// CHECK-NEXT:           _d_total += (i == 1.) * x[i] + x[i] * (i == 1.);
// CHECK-NEXT:           total += x[i] * x[i];
// CHECK-NEXT:       }
// CHECK-NEXT:   return _d_total;
// CHECK-NEXT: }

int main() {
  double x[5] = {1, 2, 3, 4, 5};
  auto d_fn_arr = clad::differentiate(sum_of_squares, "x[1]");
  printf("Result is = %.2f\n", d_fn_arr.execute(x, 5)); // CHECK-EXEC: Result is = 4.00
}
