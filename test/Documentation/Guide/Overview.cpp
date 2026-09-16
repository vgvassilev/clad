// RUN: %cladclang %s -I%S/../../../include -o%t 2>&1 | %filecheck_nodiag %s
// RUN: %t | %filecheck_prints %s
// docs-readme: overview

// docs-begin-overview
#include "clad/Differentiator/Differentiator.h"
#include <iostream>

double f(double x, double y) { return x * y; }

int main() {
  auto f_dx = clad::differentiate(f, "x");
  // Computes the derivative of 'f' at (x, y) = (3, 4) and prints it.
  std::cout << f_dx.execute(3, 4) << std::endl; // prints: 4
  f_dx.dump();
  // prints: double f_darg0(double x, double y) {
  // prints:     double _d_x = 1;
  // prints:     double _d_y = 0;
  // prints:     return _d_x * y + x * _d_y;
  // prints: }
}
// docs-end-overview
