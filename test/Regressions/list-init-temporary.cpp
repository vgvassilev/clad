// RUN: %cladclang -std=c++17 -I%S/../../include %s -o %t 2>&1 | %filecheck %s
// RUN: %t | %filecheck_exec %s

// A list-initialized temporary with several arguments, here one from a
// defaulted parameter, crashed Sema. Reduced from ImpactX's beamline elements.

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

struct Drift {
  double ds;
  Drift(double ds, int nslice = 1) : ds(ds) {}
};

double fn(double x) {
  auto d = Drift{x};
  return d.ds * x;
}

// CHECK: double fn_darg0(double x) {
// CHECK:     Drift _d_d = Drift{_d_x, 0};

int main() {
  auto d = clad::differentiate(fn, "x");
  printf("%.2f\n", d.execute(3)); // CHECK-EXEC: 6.00
}
