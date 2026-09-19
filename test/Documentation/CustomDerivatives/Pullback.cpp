// RUN: %cladclang %s -I%S/../../../include -o%t 2>&1 | %filecheck_nodiag %s
// RUN: %t | %filecheck_prints %s
//
// The regions between the docs- markers are included verbatim by
// docs/userDocs/source/user/CustomDerivatives.rst. Keep them readable: they
// are documentation that happens to be executed, not a test that happens to
// be quoted. Everything outside the markers is the harness that keeps it
// honest -- in particular the dumped gradient, which is what shows clad
// called the custom derivative instead of differentiating fn's body.

#include "clad/Differentiator/Differentiator.h"
#include <iostream>

// docs-begin-pullback-fn
double fn(double u, double v) {
  return u * v;
}
// docs-end-pullback-fn

// docs-begin-pullback-custom
namespace clad {
namespace custom_derivatives {

void fn_pullback(double u, double v, double dr, double *du, double *dv) {
  *du += v * dr;
  *dv += u * dr;
}

} // namespace custom_derivatives
} // namespace clad
// docs-end-pullback-custom

// The code the section walks through: r = fn(u, v), y = r, return y.
double f(double u, double v) {
  double r = fn(u, v);
  double y = r;
  return y;
}

int main() {
  auto f_grad = clad::gradient(f);
  double du = 0, dv = 0;
  f_grad.execute(3, 5, &du, &dv);
  // f(u, v) is u * v, so the two partial derivatives at (3, 5) are v and u.
  std::cout << du << " " << dv << "\n"; // prints: 5 3
  f_grad.dump();
  // prints: clad::custom_derivatives::fn_pullback(u, v, _d_r, &_r0, &_r1);
}
