// RUN: %cladclang %s -I%S/../../../include -o%t 2>&1 | %filecheck_nodiag %s
// RUN: %t | %filecheck_prints %s

// docs-begin-reverse-mode
#include "clad/Differentiator/Differentiator.h"
#include <iostream>

double f(double x, double y, double z) { return x * y * z; }

int main() {
  auto d_f = clad::gradient(f, "x, y");
  // One adjoint per differentiated parameter, and clad accumulates into them,
  // so they start at zero.
  double dx = 0, dy = 0;
  d_f.execute(/*x=*/2, /*y=*/3, /*z=*/4, &dx, &dy);
  std::cout << "dx: " << dx << ", dy: " << dy << "\n"; // prints: dx: 12, dy: 8
}
// docs-end-reverse-mode
