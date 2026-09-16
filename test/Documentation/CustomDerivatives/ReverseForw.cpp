// RUN: %cladclang %s -I%S/../../../include -o%t 2>&1 | %filecheck_nodiag %s
// RUN: %t | %filecheck_prints %s
//
// The regions between the docs- markers are included verbatim by
// docs/userDocs/source/user/CustomDerivatives.rst. Keep them readable: they
// are documentation that happens to be executed, not a test that happens to
// be quoted. Everything outside the markers is the harness that keeps it
// honest.
//
// The reverse-forward custom derivative belongs to g, the function returning
// the reference, not to fn, which merely calls it. Naming it fn_reverse_forw
// compiles and runs, and clad never looks at it: the dumped gradient below is
// what tells the two apart.

#include "clad/Differentiator/Differentiator.h"
#include <iostream>

// docs-begin-reverse-forw-primal
double &g(double &u, double &v) {
  if (u > v)
    return u;
  return v;
}

double fn(double u, double v) {
  double &r = g(u, v);
  return r;
}
// docs-end-reverse-forw-primal

// docs-begin-reverse-forw-custom
namespace clad {
namespace custom_derivatives {

clad::ValueAndAdjoint<double &, double &>
g_reverse_forw(double &u, double &v, double &du, double &dv) {
  if (u > v) {
    return {u, du}; // primal value and adjoint
  }
  return {v, dv}; // primal value and adjoint
}

} // namespace custom_derivatives
} // namespace clad
// docs-end-reverse-forw-custom

int main() {
  auto fn_grad = clad::gradient(fn);
  double du = 0, dv = 0;

  // v is the larger one, so g returns v and only dv picks the derivative up.
  fn_grad.execute(3, 5, &du, &dv);
  std::cout << du << " " << dv << "\n"; // prints: 0 1

  // Swap the two and the adjoint moves with the branch taken at run time.
  du = dv = 0;
  fn_grad.execute(5, 3, &du, &dv);
  std::cout << du << " " << dv << "\n"; // prints: 1 0

  fn_grad.dump();
  // prints: clad::ValueAndAdjoint<double &, double &> _t0 = clad::custom_derivatives::g_reverse_forw(u, v, *_d_u, *_d_v);
}
