// RUN: %cladclang %s -I%S/../../include -oNonActiveCallBase.out | %filecheck %s
// RUN: ./NonActiveCallBase.out | %filecheck_exec %s

// Regression test: forward mode over a member/operator call whose base object
// has no tangent, i.e. does not depend on the differentiation variable. Such a
// base has no pushforward to call; clad used to take the address of its absent
// tangent and fail with "cannot take the address of an rvalue of type 'void'".

#include "clad/Differentiator/Differentiator.h"

#include <cstdio>

struct Arr {
  double d[3];
  double operator()(int i) const { return d[i]; }
};

namespace clad {
namespace custom_derivatives {
namespace class_functions {
clad::ValueAndPushforward<double, double>
operator_call_pushforward(const Arr* a, int i, const Arr* d_a, int /*d_i*/) {
  return {(*a)(i), (*d_a)(i)};
}
} // namespace class_functions
} // namespace custom_derivatives
} // namespace clad

// g is read but never differentiated with respect to, so it has no tangent.
static Arr g{{2, 3, 4}};

double reads_nonactive(double x) {
  double s = 0;
  for (int i = 0; i < 3; ++i)
    s += x * g(i);
  return s;
}

// CHECK: double reads_nonactive_darg0(double x) {

// A base without a tangent must not zero out the whole call when an argument
// is active: the pushforward still has to run, with a zero placeholder
// filling the `_d_this` slot (issue #1960).
struct Sq {
  double operator()(double v) const { return v * v; }
};

static Sq sq;

double active_arg(double x) { return sq(x); }

// CHECK: double active_arg_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     Sq _d_base = {};
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t0 = sq.operator_call_pushforward(x, &_d_base, _d_x);
// CHECK-NEXT:     return _t0.pushforward;
// CHECK-NEXT: }

// The same shape reached through a cast address, as JIT-generated wrappers
// produce when referring to an object living in the host process.
double cast_base(double x) {
  return reinterpret_cast<Sq const*>(&sq)->operator()(x);
}

// CHECK: double cast_base_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     Sq _d_base = {};
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t0 = reinterpret_cast<const Sq *>(&sq)->operator_call_pushforward(x, &_d_base, _d_x);
// CHECK-NEXT:     return _t0.pushforward;
// CHECK-NEXT: }

// When neither the base nor any argument is varied, the call really does
// contribute nothing: a zero derivative is returned and no pushforward is
// emitted.
double inactive_arg(double x) { return x + sq(2.0); }

// CHECK: double inactive_arg_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     return _d_x + 0.;
// CHECK-NEXT: }

int main() {
  auto dx = clad::differentiate(reads_nonactive, "x");
  // d/dx sum_i x * g(i) = sum_i g(i) = 9
  printf("%.2f\n", dx.execute(1.5)); // CHECK-EXEC: 9.00

  auto dsq = clad::differentiate(active_arg, "x");
  printf("%.2f\n", dsq.execute(3.)); // CHECK-EXEC: 6.00

  auto dcast = clad::differentiate(cast_base, "x");
  printf("%.2f\n", dcast.execute(3.)); // CHECK-EXEC: 6.00

  auto dinactive = clad::differentiate(inactive_arg, "x");
  printf("%.2f\n", dinactive.execute(3.)); // CHECK-EXEC: 1.00
}
