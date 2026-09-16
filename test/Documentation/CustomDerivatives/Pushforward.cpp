// RUN: %cladclang %s -I%S/../../../include -o%t 2>&1 | %filecheck_nodiag %s
// RUN: %t | %filecheck_prints %s
//
// The regions between the docs- markers are included verbatim by
// docs/userDocs/source/user/CustomDerivatives.rst. Keep them readable: they
// are documentation that happens to be executed, not a test that happens to
// be quoted. Everything outside the markers is the harness that keeps it
// honest -- in particular the dumped derivative, which is what shows clad
// called the custom derivative instead of differentiating fn's body.

#include "clad/Differentiator/Differentiator.h"
#include <iostream>

// docs-begin-pushforward-fn
double fn(double u, double v) {
  return u * v;
}
// docs-end-pushforward-fn

// docs-begin-pushforward-custom
namespace clad {
namespace custom_derivatives {

clad::ValueAndPushforward<double, double>
fn_pushforward(double u, double v, double du, double dv) {
  double y = fn(u, v); // compute the primal value
  double dy = v * du + u * dv; // compute the output derivative
  return {y, dy};
}

} // namespace custom_derivatives
} // namespace clad
// docs-end-pushforward-custom

// The code the section walks through: u = x, v = 2 * x, y = fn(u, v).
double f(double x) {
  double u = x;
  double v = 2 * x;
  return fn(u, v);
}

int main() {
  auto d_f = clad::differentiate(f, "x");
  // f(x) is 2 * x * x, so the derivative at 3 is 4 * 3.
  std::cout << d_f.execute(3) << "\n"; // prints: 12
  d_f.dump();
  // prints: clad::ValueAndPushforward<double, double> _t0 = clad::custom_derivatives::fn_pushforward(u, v, _d_u, _d_v);
}
