// An analysis that can be switched for the whole translation unit can be
// switched for one request. The loop analysis had no clad::opts pair until
// both came from Analyses.td, so it could only be turned off for everything.
//
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -fdump-derived-fn %s \
// RUN:   -I%S/../../include -oLoopAnalysisRequestSwitch.out 2>&1 \
// RUN:   | %filecheck %s
// RUN: ./LoopAnalysisRequestSwitch.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"

#include <cstdio>

// The bound is a parameter the body never writes, so the reverse sweep could
// work the trip count out for itself -- which is what the analysis proves,
// and what this request asks it not to do.
double sum(const double* x, int n) {
  double s = 0;
  for (int i = 0; i < n; i++)
    s += x[i] * x[i];
  return s;
}

// Asked for the conservative derivative, the forward sweep counts the
// iterations and the reverse sweep spends them, rather than recomputing the
// count from the bound.
// CHECK: void sum_grad(const double *x, int n, double *_d_x, int *_d_n) {
// CHECK: for (; _t0; _t0--)

int main() {
  double x[3] = {1, 2, 3};
  double dx[3] = {0, 0, 0};
  int dn = 0;

  auto g = clad::gradient<clad::opts::disable_loop>(sum);
  g.execute(x, 3, dx, &dn);
  // An analysis may change the code clad generates, never what that code
  // computes.
  printf("%.1f %.1f %.1f\n", dx[0], dx[1], dx[2]);
  // CHECK-EXEC: 2.0 4.0 6.0
}
