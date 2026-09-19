// RUN: %cladclang %s -I%S/../../../include -o%t 2>&1 | %filecheck_nodiag %s
// RUN: %t | %filecheck_prints %s

// docs-begin-vector-mode
#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

double prod(double x, double y, double z) { return x * y * z; }

int main() {
  auto grad = clad::differentiate<clad::opts::vector_mode>(prod, "x,y");
  double x = 3.0, y = 4.0, z = 5.0;
  double dx = 0.0, dy = 0.0;
  grad.execute(x, y, z, &dx, &dy);
  printf("d_x = %.2f, d_y = %.2f\n", dx, dy); // prints: d_x = 20.00, d_y = 15.00
}
// docs-end-vector-mode
