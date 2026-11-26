// RUN: %cladclang -fsyntax-only -I%S/../../include -DON -DOFF %s
// Regression test for issue #2320: compiling with -DON or -DOFF crashed #pragma clad

#include "clad/Differentiator/Differentiator.h"

double f(double x) { return x * x; }

int main() {
  auto d = clad::differentiate(f, "x");
  (void)d;
}
