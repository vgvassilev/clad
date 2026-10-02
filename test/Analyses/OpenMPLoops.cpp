// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -enable-va %s -I%S/../../include -fopenmp -oOpenMPLoops.out
// RUN: ./OpenMPLoops.out | %filecheck_exec %s
// REQUIRES: OpenMP

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

// A function with an OpenMP parallel for loop.
// We differentiate with respect to `x`, but the OpenMP loop only uses and modifies `y` and `sum`.
// The Activity Analyzer should mark the OpenMP loop and its variables as inactive.
// Thus, it should avoid generating unnecessary derivative code for the OpenMP loop.
double test_inactive_omp_loop(double x, double y) {
  double sum = 0.0;
  
  #pragma omp parallel for reduction(+:sum)
  for (int i = 0; i < 10; ++i) {
    sum += y * i;
  }

  return (x * x) + sum;
}

// A function where the OpenMP loop is active.
// The Activity Analyzer should correctly mark it as active and
// derivative code should be generated appropriately.
double test_active_omp_loop(double x) {
  double sum = 0.0;
  
  #pragma omp parallel for reduction(+:sum)
  for (int i = 0; i < 10; ++i) {
    sum += x * i;
  }

  return sum;
}

int main() {
  auto df_inactive = clad::gradient(test_inactive_omp_loop, "x");
  double dx1 = 0.0, dy1 = 0.0;
  df_inactive.execute(3.0, 2.0, &dx1, &dy1);
  // df/dx = 2*x = 6.00
  printf("dx1 = %.2f\n", dx1); // CHECK-EXEC: dx1 = 6.00
  
  auto df_active = clad::gradient(test_active_omp_loop, "x");
  double dx2 = 0.0;
  df_active.execute(3.0, &dx2);
  // df/dx = sum(i=0 to 9) = 0+1+2+3+4+5+6+7+8+9 = 45.0
  printf("dx2 = %.2f\n", dx2); // CHECK-EXEC: dx2 = 45.00
  
  return 0;
}
