// RUN: %cladclang -std=c++17 -I%S/../../include %s -o %t 2>&1 | %filecheck %s
// RUN: %t | %filecheck_exec %s

// Derivatives are generated into every translation unit that needs them, so
// clad declares them inline. That also makes them discardable: CodeGen defers
// each until a DeclRefExpr asks for it. Hessian mode derives fn once in forward
// mode per direction and only reverse-differentiates the result, so nothing
// references fn_darg0 and it never reaches the backend.
//
// Derivatives are cached per request, so the clad::differentiate below reuses
// that same declaration rather than building its own. Its call is the
// DeclRefExpr that makes CodeGen emit fn_darg0 after all; were the intermediate
// undiscardable instead (internal linkage, or held back from CodeGen), the
// symbol would stay undefined and this would not link.

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

double fn(double x, double y) { return x * x * y; }

// CHECK: {{^}}inline double fn_darg0(double x, double y) {
// CHECK: {{^}}inline void fn_hessian(double x, double y, double *hessianMatrix) {

int main() {
  auto h = clad::hessian(fn);
  auto d = clad::differentiate(fn, "x");

  double matrix[4] = {0, 0, 0, 0};
  h.execute(3, 5, matrix);
  printf("%.2f %.2f %.2f %.2f\n", matrix[0], matrix[1], matrix[2], matrix[3]);
  // CHECK-EXEC: 10.00 6.00 6.00 0.00

  printf("%.2f\n", d.execute(3, 5));
  // CHECK-EXEC: 30.00
}
