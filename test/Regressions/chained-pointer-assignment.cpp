// RUN: %cladclang -std=c++17 -I%S/../../include %s -o %t 2>&1 | %filecheck %s
// RUN: %t | %filecheck_exec %s

// In `q = p = x + 1` on pointers the forward sweep updates both adjoint
// pointers, and the inner one is both the left-hand side of its own update and
// the right-hand side of the outer one. Using the same node for both made the
// derivative a DAG, which clad reports as an internal error.

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

double chained(double* params, double const* x) {
  double const* p = x;
  double const* q = x;
  q = p = x + 1;
  return params[0] * params[0] * q[0];
}

double caller(double* params, double const* x) { return chained(params, x); }

// CHECK: double chained_reverse_forw(double *params, const double *x, double *_d_params, double *_d_x, clad::restore_tracker &_tracker0) {
// CHECK:     _d_p = _d_x + 1;
// CHECK-NEXT:     _d_q = _d_p;
// CHECK-NEXT:     q = p = x + 1;

int main() {
  double params[1] = {2}, x[2] = {1, 3}, d[1] = {0};
  auto g = clad::gradient(caller, "params");
  g.execute(params, x, d);
  printf("%.2f\n", d[0]); // CHECK-EXEC: 12.00
}
