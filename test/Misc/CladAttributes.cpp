// RUN: %cladclang %s -I%S/../../include -oCladAttributes.out 2>&1 | %filecheck %s
// RUN: ./CladAttributes.out | %filecheck_exec %s

// Clad's attributes in their [[clad::...]] spelling. The elidable reverse_forw
// and the no-op pullback are only declared: linking succeeds only if clad
// omits the calls to them.

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

double log_value(double x, double y);
namespace clad {
namespace custom_derivatives {
[[clad::noop_derivative]] void log_value_pullback(double x, double y,
                                                  double d_out, double* d_x,
                                                  double* d_y);
} // namespace custom_derivatives
} // namespace clad
double log_value(double x, double y) { return x + y; }

double f(double x, double y) {
  S s(x, &y);
  return scale(x) * x + log_value(x, y);
}

// CHECK: void f_grad(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NOT: constructor_reverse_forw
// CHECK-NOT: log_value_pullback

int main() {
  double dx = 0, dy = 0;
  clad::gradient(f).execute(2, 5, &dx, &dy);
  printf("%.1f\n", dx); // CHECK-EXEC: 6.0
}
