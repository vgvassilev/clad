// RUN: %cladclang %s -I%S/../../../include -o%t 2>&1 | %filecheck_nodiag %s
// RUN: %t | %filecheck_prints %s

// docs-begin-forward-mode
#include "clad/Differentiator/Differentiator.h"
#include <iostream>

double fn(double x, double y) { return x * x + y * y; }

int main() {
  // Differentiate 'fn' with respect to 'x'.
  auto d_fn_1 = clad::differentiate(fn, "x");

  // Computes the derivative of 'fn' with respect to 'x' at (x, y) = (3, 4).
  std::cout << d_fn_1.execute(3, 4) << "\n"; // prints: 6
}
// docs-end-forward-mode
