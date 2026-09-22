// The switch that selects the loop analysis, what it costs to turn it off,
// and the report of what the analysis could not prove.
//
// On, and asked to report: only the loop whose trip count cannot be proven is
// named, and so is the callee whose writes cannot be attributed to a
// parameter.
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -Rclad-analysis=loop %s \
// RUN:   -I%S/../../include -oLoopAnalysisSwitch.out 2>&1 \
// RUN:   | %filecheck --check-prefix=CHECK-ON %s
// RUN: ./LoopAnalysisSwitch.out | %filecheck_exec %s
//
// Off: every loop is counted at run time, and the report says why -- not that
// the analysis failed, which would be false.
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -fdisable-analysis=loop \
// RUN:   -Xclang -plugin-arg-clad -Xclang -Rclad-analysis=loop %s \
// RUN:   -I%S/../../include -oLoopAnalysisSwitch.out 2>&1 \
// RUN:   | %filecheck --check-prefix=CHECK-OFF %s
// RUN: ./LoopAnalysisSwitch.out | %filecheck_exec %s
//
// What the switch changes in the generated code: the reverse sweep works the
// trip count out for itself, and sums a broadcast adjoint in a register.
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -fdump-derived-fn %s \
// RUN:   -I%S/../../include -oLoopAnalysisSwitch.out 2>&1 \
// RUN:   | %filecheck --check-prefix=CHECK-CODE %s
// RUN: ./LoopAnalysisSwitch.out | %filecheck_exec %s
//
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -fdisable-analysis=loop \
// RUN:   -Xclang -plugin-arg-clad -Xclang -fdump-derived-fn %s \
// RUN:   -I%S/../../include -oLoopAnalysisSwitch.out 2>&1 \
// RUN:   | %filecheck --check-prefix=CHECK-CONSERVATIVE %s
// RUN: ./LoopAnalysisSwitch.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"

#include <cstdio>

// The bounds are a parameter the body never writes, so the reverse sweep can
// work each count out; and `x[i]`, read on every iteration of the inner loop
// without moving, is a reduction over it.
double sum(const double* x, const double* w, int n) {
  double s = 0;
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      s += x[i] * w[j];
  return s;
}

// The bound is a variable the body writes, so nothing about the loop's own
// bounds says how many times it ran.
double walk(const double* x, int n) {
  double s = 0;
  int k = n;
  for (int i = 0; i < k; i++) {
    s += x[i];
    k = k - 1;
  }
  return s;
}

// The increment steps by two, so the count is not the distance between the
// bounds.
double stride(const double* x, int n) {
  double s = 0;
  for (int i = 0; i < n; i += 2)
    s += x[i];
  return s;
}

double f(const double* x, const double* w, int n) {
  return sum(x, w, n) + walk(x, n) + stride(x, n);
}

// Nothing is said about the two loops of sum: their counts are proven, so
// there is nothing left in the generated code to report about them.
// CHECK-ON-NOT: LoopAnalysisSwitch.C:[[# @LINE - 33]]:3:
// CHECK-ON-NOT: LoopAnalysisSwitch.C:[[# @LINE - 33]]:5:

// Each report reads cost, cause, fix: what clad had to emit, what in the code
// forced it, and what to write instead. The caret of the cause is on the token
// that missed -- the bound here, the increment below -- not on the `for`,
// which is where the cost lands but not the cause.
// The inner loop's own index moves w's read, so nothing is said about it. In
// the outer loop it stands still, and what stops it there is said instead.
// CHECK-ON: LoopAnalysisSwitch.C:[[# @LINE - 40]]:19: remark: clad adds to this adjoint in memory on every iteration
// CHECK-ON: note: the index reads a variable the loop writes
// CHECK-ON: note: to avoid this, make it a broadcast read (CLAD1002)
// CHECK-ON: LoopAnalysisSwitch.C:[[# @LINE - 34]]:3: remark: clad adds a counter to this loop and increments it every iteration
// CHECK-ON: LoopAnalysisSwitch.C:[[# @LINE - 35]]:23: note: the bound is written elsewhere in the function
// CHECK-ON: note: to avoid this, make it a counted loop (CLAD1001)
// CHECK-ON: LoopAnalysisSwitch.C:[[# @LINE - 26]]:3: remark: clad adds a counter to this loop and increments it every iteration
// CHECK-ON: LoopAnalysisSwitch.C:[[# @LINE - 27]]:26: note: the increment does not step the index by one
// CHECK-ON: note: to avoid this, make it a counted loop (CLAD1001)

// Off, every loop is reported, and the note names the switch that turned the
// analysis off rather than blaming code that nothing looked at.
// CHECK-OFF: LoopAnalysisSwitch.C:[[# @LINE - 54]]:3: remark: clad adds a counter to this loop and increments it every iteration
// CHECK-OFF: note: the loop analysis is off (-fdisable-analysis=loop)
// CHECK-OFF: LoopAnalysisSwitch.C:[[# @LINE - 55]]:5: remark: clad adds a counter to this loop and increments it every iteration
// CHECK-OFF: note: the loop analysis is off (-fdisable-analysis=loop)
// CHECK-OFF: LoopAnalysisSwitch.C:[[# @LINE - 47]]:3: remark: clad adds a counter to this loop and increments it every iteration
// CHECK-OFF: note: the loop analysis is off (-fdisable-analysis=loop)
// CHECK-OFF: LoopAnalysisSwitch.C:[[# @LINE - 38]]:3: remark: clad adds a counter to this loop and increments it every iteration
// CHECK-OFF: note: the loop analysis is off (-fdisable-analysis=loop)

// CHECK-CODE: void sum_pullback(const double *x, const double *w, int n, double _d_y, double *_d_x, int *_d_n) {
// CHECK-CODE: for (_t0 = n > 0 ? (unsigned {{int|long|long long}})n : 0{{U|UL|ULL}}; _t0; _t0--) {
// CHECK-CODE: double _acc0 = 0.;
// CHECK-CODE: _acc0 += _r_d0 * w[j];
// CHECK-CODE: _d_x[i] += _acc0;

// With the analysis off the forward sweep counts the iterations instead --
// onto a tape, for the inner loop -- and the adjoint is accumulated where it
// stands. The values are the same either way: an analysis may change the code
// clad generates, never what that code computes.
// CHECK-CONSERVATIVE: void sum_pullback(const double *x, const double *w, int n, double _d_y, double *_d_x, int *_d_n) {
// CHECK-CONSERVATIVE: clad::tape<unsigned {{int|long|long long}}> _t1 = {};
// CHECK-CONSERVATIVE: for (; _t0; _t0--) {
// CHECK-CONSERVATIVE-NOT: _acc
// CHECK-CONSERVATIVE: _d_x[i] += _r_d0 * w[j];

int main() {
  double x[4] = {1, 2, 3, 4};
  double w[4] = {1, 1, 1, 1};
  double dx[4] = {0, 0, 0, 0};
  auto g = clad::gradient(f, "x");
  g.execute(x, w, 4, dx);
  // sum is (x[0] + ... + x[3]) * (w[0] + ... + w[3]), so d/dx[i] is 4 for
  // every i; walk runs over the first two elements and adds (1, 1, 0, 0);
  // stride runs over the even ones and adds (1, 0, 1, 0).
  printf("%.2f %.2f %.2f %.2f\n", dx[0], dx[1], dx[2], dx[3]);
  // CHECK-EXEC: 6.00 5.00 5.00 4.00
}
