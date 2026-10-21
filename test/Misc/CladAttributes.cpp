// RUN: %cladclang %s -I%S/../../include -oCladAttributes.out 2>&1 | %filecheck %s
// RUN: ./CladAttributes.out | %filecheck_exec %s

// Clad's attributes in their [[clad::...]] spelling. The elidable reverse_forw
// is only declared: linking succeeds only if clad omits the call to it.

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

[[clad::non_differentiable]] double scale(double x) { return 3 * x; }

struct S {
  double v;
  S(double x, double* y) : v(x) { *y = x; }
};

namespace clad {
namespace custom_derivatives {
namespace class_functions {
[[clad::elidable_reverse_forw]] clad::ValueAndAdjoint<S, S>
constructor_reverse_forw(clad::Tag<S>, double x, double* y, double d_x,
                         double* d_y);
} // namespace class_functions
} // namespace custom_derivatives
} // namespace clad

double f(double x, double y) {
  S s(x, &y);
  return scale(x) * x;
}

// CHECK: void f_grad(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NOT: constructor_reverse_forw

int main() {
  double dx = 0, dy = 0;
  clad::gradient(f).execute(2, 5, &dx, &dy);
  printf("%.1f\n", dx); // CHECK-EXEC: 6.0
}
