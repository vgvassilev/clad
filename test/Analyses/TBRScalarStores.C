// What to-be-recorded analysis changes about a plain assignment: the copy
// clad would otherwise take of the value the assignment is about to destroy.
//
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -fdisable-analysis=all %s \
// RUN:   -I%S/../../include -oTBRScalarStores.out 2>&1 \
// RUN:   | %filecheck --check-prefix=CHECK-OFF %s
// RUN: ./TBRScalarStores.out | %filecheck_exec %s
//
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -fenable-analysis=tbr %s \
// RUN:   -I%S/../../include -oTBRScalarStores.out 2>&1 \
// RUN:   | %filecheck --check-prefix=CHECK-TBR %s
// RUN: ./TBRScalarStores.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"

#include <cstdio>

// Two assignments overwrite t. The first is squaring, whose pullback reads the
// value being overwritten, so that copy has to be taken either way. The second
// discards t outright, and nothing in the reverse sweep asks what it held, so
// only that copy and its restore go away. Keeping both cases in one function
// is the point: it shows the analysis deciding, not deleting.
double f(double x) {
  double t = x * x;
  t = t * t;
  t = x + 1;
  return t * x;
}

// CHECK-OFF-LABEL: void f_grad(double x, double *_d_x) {
// CHECK-OFF: double _t0 = t;
// CHECK-OFF: t = t * t;
// CHECK-OFF: double _t1 = t;
// CHECK-OFF: t = x + 1;
// CHECK-OFF: t = _t1;
// CHECK-OFF: t = _t0;

// CHECK-TBR-LABEL: void f_grad(double x, double *_d_x) {
// CHECK-TBR: double _t0 = t;
// CHECK-TBR: t = t * t;
// CHECK-TBR-NOT: double _t1 = t;
// CHECK-TBR-NOT: t = _t1;
// CHECK-TBR: t = _t0;

int main() {
  auto g = clad::gradient(f);
  double d = 0;
  g.execute(3, &d);
  printf("%.2f\n", d); // CHECK-EXEC: 7.00
}
