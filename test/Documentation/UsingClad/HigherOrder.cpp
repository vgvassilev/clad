// RUN: %cladclang %s -I%S/../../../include -o%t 2>&1 | %filecheck_nodiag %s
// RUN: %t | %filecheck_prints %s

// docs-begin-higher-order
#include "clad/Differentiator/Differentiator.h"
#include <iostream>

double fn(double i) { return i * i * i * i; }

int main() {
  // Differentiate the 3rd order derivative of 'fn' with respect to 'i'.
  auto d_fn_3 = clad::differentiate<3>(fn, "i");
  std::cout << d_fn_3.execute(3) << "\n"; // prints: 72
}
// docs-end-higher-order
