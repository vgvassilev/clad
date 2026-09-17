// RUN: %cladclang %s -I%S/../../../include -o%t 2>&1 | %filecheck_nodiag %s
// RUN: %t | %filecheck_prints %s

// docs-begin-fixed-size-array
#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

double weighted(double x[3]) { return x[0] * x[1] + x[2]; }

int main() {
  auto g = clad::gradient(weighted);
  double x[3] = {2, 3, 4}, dx[3] = {};
  g.execute(x, dx);
  printf("dx = {%g, %g, %g}\n", dx[0], dx[1], dx[2]); // prints: dx = {3, 2, 1}
}
// docs-end-fixed-size-array
