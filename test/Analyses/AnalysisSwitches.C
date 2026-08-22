// The switches that select an analysis, and the value they must not change.
//
// With the analyses off, clad stores t before each assignment that overwrites
// it: nothing has proven the overwritten value is never read again.
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -fdisable-analysis=all %s \
// RUN:   -I%S/../../include -oAnalysisSwitches.out 2>&1 \
// RUN:   | %filecheck --check-prefix=CHECK-OFF %s
// RUN: ./AnalysisSwitches.out | %filecheck_exec %s
//
// Naming the one analysis that changes this function does the same.
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -fdisable-analysis=tbr %s \
// RUN:   -I%S/../../include -oAnalysisSwitches.out 2>&1 \
// RUN:   | %filecheck --check-prefix=CHECK-OFF %s
// RUN: ./AnalysisSwitches.out | %filecheck_exec %s
//
// With to-be-recorded on, one store and its restore are gone.
// RUN: %cladclang %s -I%S/../../include -oAnalysisSwitches.out 2>&1 \
// RUN:   | %filecheck --check-prefix=CHECK-TBR %s
// RUN: ./AnalysisSwitches.out | %filecheck_exec %s
//
// The last switch decides, in either order.
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -fdisable-analysis=all \
// RUN:   -Xclang -plugin-arg-clad -Xclang -fenable-analysis=tbr %s \
// RUN:   -I%S/../../include -oAnalysisSwitches.out 2>&1 \
// RUN:   | %filecheck --check-prefix=CHECK-TBR %s
// RUN: ./AnalysisSwitches.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -fenable-analysis=tbr \
// RUN:   -Xclang -plugin-arg-clad -Xclang -fdisable-analysis=all %s \
// RUN:   -I%S/../../include -oAnalysisSwitches.out 2>&1 \
// RUN:   | %filecheck --check-prefix=CHECK-OFF %s
// RUN: ./AnalysisSwitches.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"

#include <cstdio>

double f(double x) {
  double t = x * x;
  t = t * t;
  t = x + 1;
  return t * x;
}

// The first assignment's store survives either way -- `t = t * t` reads the
// value it overwrites. The second's does not: `t = x + 1` discards t, and the
// reverse sweep needs no value from before it.

// CHECK-OFF: void f_grad(double x, double *_d_x) {
// CHECK-OFF: double _t0 = t;
// CHECK-OFF: t = t * t;
// CHECK-OFF: double _t1 = t;
// CHECK-OFF: t = x + 1;
// CHECK-OFF: t = _t1;

// CHECK-TBR: void f_grad(double x, double *_d_x) {
// CHECK-TBR: double _t0 = t;
// CHECK-TBR: t = t * t;
// CHECK-TBR-NOT: double _t1 = t;
// CHECK-TBR-NOT: t = _t1;

int main() {
  auto g = clad::gradient(f);
  double d = 0;
  g.execute(3, &d);
  printf("%.2f\n", d); // CHECK-EXEC: 7.00
}
