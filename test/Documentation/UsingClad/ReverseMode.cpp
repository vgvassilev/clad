// RUN: %cladclang %s -I%S/../../../include -o%t 2>&1 | %filecheck_nodiag %s
// RUN: %t | %filecheck_prints %s

// docs-begin-reverse-mode
#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

double fn(double x, double y) { return x * x + y * y; }

int main() {
  // Differentiate 'fn' with respect to every parameter.
  auto fn_grad = clad::gradient(fn);

  // The derivative is accumulated into these, so they start at zero.
  double dx = 0, dy = 0;

  // The arguments of 'fn' first, then one pointer per differentiated
  // parameter, in the same order.
  fn_grad.execute(3, 4, &dx, &dy);

  printf("dfn/dx = %g, dfn/dy = %g\n", dx, dy);
  // prints: dfn/dx = 6, dfn/dy = 8
}
// docs-end-reverse-mode
