// RUN: %cladclang -I%S/../../include %s -o %t
// RUN: %t | %filecheck_exec %s
// UNSUPPORTED: clang-11, clang-12, clang-13, clang-14, clang-15, clang-16

// Reverse mode copies a lambda's body to build the primal, and the copy of a
// declaration was written for variables alone: anything else came back null
// and left a declaration statement with nothing in it, which crashed clang the
// next time the body was printed.

#include "clad/Differentiator/Differentiator.h"

#include <cstdio>

double square(double x) {
  auto f = [](double v) {
    using real = double;
    typedef real elem;
    elem w = v * v;
    return w;
  };
  return f(x) * x;
}

int main() {
  auto g = clad::gradient(square);
  double dx = 0;
  g.execute(3, &dx);
  printf("square: %.2f\n", dx);
  // CHECK-EXEC: square: 27.00
}
