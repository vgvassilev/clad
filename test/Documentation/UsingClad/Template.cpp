// RUN: %cladclang %s -I%S/../../../include -o%t 2>&1 | %filecheck_nodiag %s
// RUN: %t | %filecheck_prints %s

// docs-begin-template
#include "clad/Differentiator/Differentiator.h"
#include <iostream>

template <typename T> T poly(T x, T y) { return x * x * y; }

int main() {
  auto d_poly = clad::gradient(poly<double>);
  double d_x = 0, d_y = 0;
  d_poly.execute(3, 5, &d_x, &d_y);
  std::cout << d_x << " " << d_y << "\n"; // prints: 30 9
}
// docs-end-template
