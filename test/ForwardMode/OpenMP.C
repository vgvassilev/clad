// RUN: %cladclang %s -I%S/../../include %ompldflags -oOpenMP.out 2>&1 | %filecheck %s
// RUN: ./OpenMP.out | %filecheck_exec %s
//
// REQUIRES: openmp

#include "clad/Differentiator/Differentiator.h"

double sum_of_squares(const double *x, int n) {
  double total = 0.0;

  #pragma omp parallel for reduction(+:total)
  for (int i = 0; i < n; i++) {
    total += x[i] * x[i];
  }

  return total;
}

int main() {
    double x[5] = {1, 2, 3, 4, 5};
    auto d_fn_arr = clad::differentiate(sum_of_squares, "x[1]");
    printf("Result is = %.2f\n", d_fn_arr.execute(x, 5)); // CHECK-EXEC: Result is = 4.00
}
