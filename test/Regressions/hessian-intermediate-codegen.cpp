// RUN: %cladclang -std=c++17 -I%S/../../include %s -o %t 2>&1 | %filecheck %s
// RUN: %t | %filecheck_exec %s

// Derivatives are generated into every translation unit that needs them, so
// clad declares them inline. That also makes them discardable: CodeGen defers
// each until a DeclRefExpr asks for it. A hessian is assembled from one
// pushforward of the function and one pullback of that pushforward, and only
// the pullback is called, so the pushforward itself never reaches the backend.
//
// Derivatives are cached per request, so the forward-mode derivative below
// reuses that same declaration rather than building its own. Its call to
// fn_pushforward is the DeclRefExpr that makes CodeGen emit it after all; were
// the intermediate undiscardable instead (internal linkage, or held back from
// CodeGen), the symbol would stay undefined and this would not link.

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

double fn(double x, double y) { return x * x * y; }
double caller(double x, double y) { return fn(x, y) + y; }

// CHECK: {{^}}inline clad::ValueAndPushforward<double, double> fn_pushforward(double x, double y, double _d_x, double _d_y) {
// CHECK: {{^}}inline void fn_hessian(double x, double y, double *hessianMatrix) {
// CHECK: {{^}}inline double caller_darg0(double x, double y) {

int main() {
  auto h = clad::hessian(fn);
  auto d = clad::differentiate(caller, "x");

  double matrix[4] = {0, 0, 0, 0};
  h.execute(3, 5, matrix);
  printf("%.2f %.2f %.2f %.2f\n", matrix[0], matrix[1], matrix[2], matrix[3]);
  // CHECK-EXEC: 10.00 6.00 6.00 0.00

  printf("%.2f\n", d.execute(3, 5));
  // CHECK-EXEC: 30.00
}
