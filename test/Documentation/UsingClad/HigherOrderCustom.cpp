// RUN: %cladclang %s -I%S/../../../include -o%t 2>&1 | %filecheck_nodiag %s
// RUN: %t | %filecheck_prints %s

// docs-begin-higher-order-custom
#include "clad/Differentiator/Differentiator.h"
#include <cmath>
#include <iostream>

double mysin(double x) { return std::sin(x); }

int main() {
  auto d_sin_2 = clad::differentiate<2>(mysin);
  std::cout << d_sin_2.execute(3) << "\n"; // prints: -0.14112
}
// docs-end-higher-order-custom
