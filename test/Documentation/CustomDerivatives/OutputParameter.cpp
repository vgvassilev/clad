// RUN: %cladclang %s -I%S/../../../include -o%t 2>&1 | %filecheck_nodiag %s
// RUN: %t | %filecheck_prints %s
//
// The region between the docs- markers is included verbatim by
// docs/userDocs/source/user/CustomDerivatives.rst. Keep it readable: it is
// documentation that happens to be executed, not a test that happens to be
// quoted. Everything outside the markers is the harness that keeps it honest
// -- in particular the dumped derivative, which is what shows clad called the
// custom derivative instead of differentiating f's body.

#include "clad/Differentiator/Differentiator.h"
#include <iostream>

// docs-begin-output-parameter
void f(double x, double& out) {
  out = x * x;
}

namespace clad {
namespace custom_derivatives {

// `double`, not `void`: the derivative of a function that writes its result
// through a single floating point reference returns that parameter's tangent,
// so that is what a custom derivative for it has to return.
double f_darg0(double x, double& out) {
  out = x * x;  // the primal value, as f computes it
  return 2 * x; // and the tangent of out, which the caller reads
}

} // namespace custom_derivatives
} // namespace clad
// docs-end-output-parameter

int main() {
  auto d_f = clad::differentiate(f, "x");
  double out = 0;
  std::cout << d_f.execute(3, out) << "\n"; // prints: 6
  std::cout << out << "\n";                 // prints: 9
  d_f.dump();
  // The body is the one written above, not one clad derived: a signature that
  // did not match would be reported rather than quietly replaced.
  // prints: return 2 * x;
}
