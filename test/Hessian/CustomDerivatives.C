// RUN: %cladclang %s -I%S/../../include -oCustomDerivatives.out 2>&1 | %filecheck %s
// RUN: ./CustomDerivatives.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"

// Each direction of a hessian resolves its own custom forward derivative. The
// planner reuses one forward request across directions, so a custom derivative
// registered for the first direction must not be carried into the second.

double f(double x, double y) { return x * x * y; }

namespace clad {
namespace custom_derivatives {
// Only the first direction has a custom derivative, and it is deliberately
// scaled by 1000 so that using it for the second direction too is visible.
double f_darg0(double x, double y) { return 1000 * 2 * x * y; }
} // namespace custom_derivatives
} // namespace clad

// CHECK: void f_hessian(double x, double y, double *hessianMatrix) {
// CHECK-NEXT:     clad::custom_derivatives::f_darg0_grad(x, y, hessianMatrix + {{0U|0UL|0ULL}}, hessianMatrix + {{1U|1UL|1ULL}});
// CHECK-NEXT:     f_darg1_grad(x, y, hessianMatrix + {{2U|2UL|2ULL}}, hessianMatrix + {{3U|3UL|3ULL}});
// CHECK-NEXT: }

int main() {
  double m[4] = {};
  clad::hessian(f).execute(3., 1., m);
  // Row 0 differentiates the custom 2000*x*y; row 1 the real df/dy = x*x.
  printf("f [%.0f, %.0f, %.0f, %.0f]\n", m[0], m[1], m[2], m[3]);
  // CHECK-EXEC: f [2000, 6000, 6, 0]
}
