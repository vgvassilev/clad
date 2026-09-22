// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr %s -I%S/../../include -o %t 2>&1 | %filecheck --check-prefix=WARN %s
// RUN: %t | %filecheck_exec %s

// WARN: warning: gradient uses a global variable
// WARN-NOT: error:

#include "clad/Differentiator/Differentiator.h"

#include <cstdio>

double carry = 0;

double f_loop_carry(double x) {
  double r = 0;
  for (int i = 0; i < 2; ++i) {
    r = r + carry;
    carry = x;
  }
  return r;
}

int main() {
  auto grad = clad::gradient<clad::opts::enable_va>(f_loop_carry);
  double dx = 0;
  carry = 0;
  grad.execute(3, &dx);
  printf("{%.2f}\n", dx); // CHECK-EXEC: {1.00}
  return 0;
}
