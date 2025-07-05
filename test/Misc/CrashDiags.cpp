// RUN: env CLAD_FORCE_CRASH= not %cladclang %s -I%S/../../include 2>&1 | FileCheck %s
// RUN: env CLAD_FORCE_CRASH= not %cladclang -DREVERSE %s -I%S/../../include 2>&1 | FileCheck %s
// REQUIRES: asserts

#include "clad/Differentiator/Differentiator.h"

double fn1(double x) {
  return x * x + 3 * x + 5;
}

int main() {
#ifdef REVERSE
  auto grad = clad::gradient(fn1);
#else
  auto dx = clad::differentiate(fn1);
#endif
}

// CHECK: Building code for '<double fn1(double x)>[name=fn1, order=1, mode={{.*}}, args='', tbr]'
// CHECK-NEXT: While visiting <CompoundStmt> [ '
// CHECK: --- Begin Stmt Dump ---
// CHECK return x * x + 3 * x + 5;
// CHECK: --- End Stmt Dump ---
