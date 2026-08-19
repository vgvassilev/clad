// RUN: %cladclang -O1 -Xclang -fdebug-pass-manager %s -I%S/../../include \
// RUN:   -oPipelineOptOut.out 2>&1 | %filecheck %s
// RUN: ./PipelineOptOut.out | %filecheck_exec %s
// REQUIRES: Enzyme

// The other half of PipelineOptIn.C: an ordinary clad gradient in a build
// that has the Enzyme backend compiled in. Nothing here names that backend,
// so its passes are asked for and then declined -- the decision is made per
// module, when the pass would run, rather than once per process.

#include "clad/Differentiator/Differentiator.h"

#include <cstdio>

double f(double x, double y) { return x * x * y; }

// CHECK: Skipping pass: EnzymeNewPM
// CHECK-NOT: Running pass: EnzymeNewPM

int main() {
  auto g = clad::gradient(f);
  double dx = 0, dy = 0;
  g.execute(3, 4, &dx, &dy);
  printf("{%.2f, %.2f}\n", dx, dy);
  // CHECK-EXEC: {24.00, 9.00}
}
