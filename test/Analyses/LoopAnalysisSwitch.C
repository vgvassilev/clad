// The switch that selects the loop analysis, and what it costs to turn it off.
//
// What the switch changes in the generated code: the reverse sweep works the
// trip count out for itself, and sums a broadcast adjoint in a register.
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -fdump-derived-fn %s \
// RUN:   -I%S/../../include -oLoopAnalysisSwitch.out 2>&1 \
// RUN:   | %filecheck --check-prefix=CHECK-CODE %s
// RUN: ./LoopAnalysisSwitch.out | %filecheck_exec %s
//
// With it off the forward sweep counts the iterations instead -- onto a tape,
// for the inner loop -- and the adjoint is accumulated where it stands. The
// values are the same either way: an analysis may change the code clad
// generates, never what that code computes.
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

double f(const double* x, const double* w, int n) {
  return sum(x, w, n) + walk(x, n);
}

// Off, every loop is reported, and for the other reason.

// CHECK-CODE: void sum_pullback(const double *x, const double *w, int n, double _d_y, double *_d_x, int *_d_n) {
// CHECK-CODE: for (_t0 = n > 0 ? (unsigned {{int|long|long long}})n : 0{{U|UL|ULL}}; _t0; _t0--) {
// CHECK-CODE: double _acc0 = 0.;
// CHECK-CODE: _acc0 += _r_d0 * w[j];
// CHECK-CODE: _d_x[i] += _acc0;

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
  // every i; walk runs over the first two elements only and adds (1, 1, 0, 0).
  printf("%.2f %.2f %.2f %.2f\n", dx[0], dx[1], dx[2], dx[3]);
  // CHECK-EXEC: 5.00 5.00 4.00 4.00
}
