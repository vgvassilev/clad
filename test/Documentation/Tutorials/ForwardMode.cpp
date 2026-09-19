// RUN: %cladclang %s -I%S/../../../include -o%t 2>&1 | %filecheck_nodiag %s
// RUN: %t | %filecheck_prints %s
//
// The region between the docs- markers is included verbatim by
// docs/userDocs/source/user/tutorials.rst. Keep it readable: it is
// documentation that happens to be executed, not a test that happens to be
// quoted. Everything outside the markers is the harness that keeps it honest.

// docs-begin-forward-mode
#include "clad/Differentiator/Differentiator.h"
#include <iostream>

double func(double x) { return x * x; }

int main() {
  // Ask clad for the derivative of func with respect to x.
  auto d_func = clad::differentiate(func, "x");
  // Call it the way func itself would be called.
  std::cout << d_func.execute(/*x=*/3) << "\n"; // prints: 6
  // And print the code clad generated for it.
  d_func.dump();
}
// docs-end-forward-mode
