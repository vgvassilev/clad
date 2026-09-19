// RUN: %cladclang %s -I%S/../../../include -o%t 2>&1 | %filecheck_nodiag %s
// RUN: %t | %filecheck_prints %s

// docs-begin-error-estimation
#include "clad/Differentiator/Differentiator.h"
#include <iomanip>
#include <iostream>

double func(double x, double y) { return x * y; }

int main() {
  auto dfunc_error = clad::estimate_error(func);

  // The gradient's arguments, plus a double& that receives the error estimate.
  double x = 3, y = 5, d_x = 0, d_y = 0, final_error = 0;
  dfunc_error.execute(x, y, &d_x, &d_y, final_error);

  std::cout << std::setprecision(3) << final_error << "\n"; // prints: 5.36e-06
}
// docs-end-error-estimation
