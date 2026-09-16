// RUN: %cladclang %s -I%S/../../../include -o%t 2>&1 | %filecheck_nodiag %s
// RUN: %t | %filecheck_prints %s

// docs-begin-pointer-array
#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

double sum_sq(const double* x, int n) {
  double s = 0;
  for (int i = 0; i < n; ++i)
    s += x[i] * x[i];
  return s;
}

int main() {
  auto g = clad::gradient(sum_sq, "x");
  double y[4] = {1, 2, 3, 4}, dy[4] = {};
  clad::array_ref<double> dy_ref(dy, 4);
  g.execute(y, 4, dy_ref);
  printf("dy = {%g, %g, %g, %g}\n", dy[0], dy[1], dy[2],
         dy[3]); // prints: dy = {2, 4, 6, 8}
}
// docs-end-pointer-array
