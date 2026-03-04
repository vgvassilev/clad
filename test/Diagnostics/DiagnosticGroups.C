// Test diagnostic groups allow controlling warnings with -W flags
// RUN: %cladclang %s -I%S/../../include 2>&1 | %filecheck %s --check-prefix=CHECK-DEFAULT
// RUN: %cladclang %s -I%S/../../include -Wno-clad 2>&1 | %filecheck %s --check-prefix=CHECK-NO-CLAD

#include "clad/Differentiator/Differentiator.h"

extern int (*indirect_func)(int);

int test_unsupported(int x) {
  // CHECK-DEFAULT: indirect calls
  // CHECK-NO-CLAD-NOT: indirect calls
  return indirect_func(x);
}

double test_valid(double x) {
  return x * x + 2 * x + 1;
}

int main() {
  auto grad = clad::gradient(test_valid);
  return 0;
}
