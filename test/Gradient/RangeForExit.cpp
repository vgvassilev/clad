// RUN: %cladclang %s -I%S/../../include -oRangeForExit.out
// RUN: ./RangeForExit.out | %filecheck_exec %s

// An iteration of a range-for left by break or return still steps the adjoint
// iterator back in the reverse sweep, so it has to have been advanced.

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

double breaks(double x) {
  double s = 0;
  double a[3] = {x, 2 * x, 3 * x};
  for (double v : a) {
    if (v > 4)
      break;
    s += v;
  }
  return s;
}

double returns(double x) {
  double s = 0;
  double a[3] = {x, 2 * x, 3 * x};
  for (double v : a) {
    if (v > 4)
      return s;
    s += v;
  }
  return s;
}

int main() {
  double dx = 0;
  clad::gradient(breaks).execute(2, &dx);
  printf("%.2f\n", dx); // CHECK-EXEC: 3.00
  dx = 0;
  clad::gradient(returns).execute(2, &dx);
  printf("%.2f\n", dx); // CHECK-EXEC: 3.00
}
