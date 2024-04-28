// RUN: %cladclang %s -I%S/../../include -oTestExecOnly.out 2>&1 | %filecheck %s
// RUN: ./TestExecOnly.out | %filecheck_exec %s
// CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

double sq(double x) { return x*x; }

// CHECK: This check must not trigger

extern "C" int printf(const char*,...);
int main() {
  auto dsq = clad::differentiate(sq, "x");
  printf("dsq(1.)=%f\n", dsq.execute(1.));
}

// CHECK-EXEC: dsq(1.)=1.0
