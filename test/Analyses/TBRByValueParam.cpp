// RUN: %cladclang %s -I%S/../../include -oTBRByValueParam.out 2>&1 \
// RUN:   | %filecheck %s
// RUN: ./TBRByValueParam.out | %filecheck_exec %s

// A write to a number passed by value stays in the callee, so a call to it
// needs no restore_tracker.

#include "clad/Differentiator/Differentiator.h"

#include <cstdio>

struct M {
  double m[2];
  double& at(int i) {
    --i;
    return m[i];
  }
};

double f(double x) {
  M a{{x, 0}};
  a.at(2) = a.at(1) * x;
  return a.at(2);
}

// CHECK: at_reverse_forw(int i, M *_d_this, int _d_i) {
// CHECK: void f_grad(
// CHECK-NOT: _tracker

int main() {
  double dx = 0;
  clad::gradient(f).execute(3, &dx);
  printf("%.2f\n", dx); // CHECK-EXEC: 6.00
}
