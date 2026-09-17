// RUN: %cladclang %s -I%S/../../../include -o%t 2>&1 | %filecheck_nodiag %s
// RUN: %t | %filecheck_prints %s
// docs-readme: error-estimation

// docs-begin-error-estimation
#include "clad/Differentiator/Differentiator.h"
#include <iostream>

double f(double x, double y) {
  double z;
  z = x + y;
  return z;
}

int main() {
  // Generate the floating point error estimation code for 'f'.
  auto df = clad::estimate_error(f);
  // Print the generated code to standard output.
  df.dump();
  // Declare the necessary variables.
  double x = 3, y = 5, d_x = 0, d_y = 0, final_error = 0;
  // Finally call execute on the generated code.
  df.execute(x, y, &d_x, &d_y, final_error);
  // After this, 'final_error' holds the floating point error in 'f'.
  std::cout << final_error << "\n"; // prints: 1.90735e-06
}
// docs-end-error-estimation
