// RUN: %cladclang %s -I%S/../../../include -o%t 2>&1 | %filecheck_nodiag %s
// RUN: %t | %filecheck_prints %s
//
// The region between the docs- markers is included verbatim by
// docs/userDocs/source/user/CustomDerivatives.rst. Keep it readable: it is
// documentation that happens to be executed, not a test that happens to be
// quoted. Everything outside the markers is the harness that keeps it honest
// -- in particular the dumped derivatives, which are what show clad called
// the custom derivatives instead of differentiating A::fn's body.

#include "clad/Differentiator/Differentiator.h"
#include <iostream>

// docs-begin-member-functions
class A {
public:
  double val1, val2;

  double fn(double u, double v) {
    return u * val1 + v * val2;
  }
};

namespace clad {
namespace custom_derivatives {
namespace class_functions {
  // pushforward custom derivative
  clad::ValueAndPushforward<double, double>
  fn_pushforward(A *a, double u, double v, A *da, double du, double dv) {
    double y = a->fn(u, v); // compute the primal value
    // compute the derivative
    double dy = u * da->val1 + du * a->val1 + v * da->val2 + dv * a->val2;
    return {y, dy};
  }

  // pullback custom derivative
  void fn_pullback(A *a, double u, double v, double dr, A *da, double *du, double *dv) {
    *du += dr * a->val1;
    da->val1 += dr * u;
    *dv += dr * a->val2;
    da->val2 += dr * v;
  }
} // namespace class_functions
} // namespace custom_derivatives
} // namespace clad
// docs-end-member-functions

double host(double u, double v) {
  A a{2, 3};
  return a.fn(u, v);
}

int main() {
  // host(u, v) is 2 * u + 3 * v, so its derivative with respect to u is val1.
  auto host_du = clad::differentiate(host, "u");
  std::cout << host_du.execute(4, 5) << "\n"; // prints: 2
  host_du.dump();
  // prints: clad::ValueAndPushforward<double, double> _t0 = clad::custom_derivatives::class_functions::fn_pushforward(&a, u, v, &_d_a, _d_u, _d_v);

  auto host_grad = clad::gradient(host);
  double du = 0, dv = 0;
  host_grad.execute(4, 5, &du, &dv);
  std::cout << du << " " << dv << "\n"; // prints: 2 3
  host_grad.dump();
  // prints: clad::custom_derivatives::class_functions::fn_pullback(&a, u, v, 1, &_d_a, &_r0, &_r1);
}
