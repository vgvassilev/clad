// What the forced std::terminate leaves behind depends on the clang running
// it: through 22 the driver exits non-zero, while 23 lets the signal through,
// so neither `not` nor `not --crash` is right across the versions clad
// supports. The trace is what this test is about, so hand it to FileCheck and
// let that decide. `;` gives lit only the right-hand command's status.
// RUN: env CLAD_FORCE_CRASH= %cladclang %s -I%S/../../include > %t.fwd 2>&1 ; FileCheck %s < %t.fwd
// RUN: env CLAD_FORCE_CRASH= %cladclang -DREVERSE %s -I%S/../../include > %t.rev 2>&1 ; FileCheck %s < %t.rev
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

// CHECK: Building code for '<double fn1(double x)>[name=fn1, order=1, mode={{.*}}, args=''
// CHECK-NEXT: While visiting <CompoundStmt> [ '
// CHECK: --- Begin Stmt Dump ---
// CHECK return x * x + 3 * x + 5;
// CHECK: --- End Stmt Dump ---
