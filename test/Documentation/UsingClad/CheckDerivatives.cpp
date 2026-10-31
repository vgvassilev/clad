// RUN: %cladnumdiffclang %s -I%S/../../../include -o%t 2>&1 | %filecheck_nodiag %s
// RUN: %t | %filecheck_prints %s
//
// The region between the docs- markers is included verbatim by
// docs/userDocs/source/user/UsingClad.rst. Keep it readable: it is
// documentation that happens to be executed, not a test that happens to be
// quoted.
//
// %cladnumdiffclang, not %cladclang: the check compares against the numerical
// differentiation that -DCLAD_NO_NUM_DIFF compiles out.

// docs-begin-check-derivatives
#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

double f(double x, double y) { return x * x * y; }

int main() {
  auto df = clad::gradient(f);

  // Compare what clad generated against central differences of f itself.
  clad::check_report report = clad::check_gradient(f, df, 3.0, 2.0);
  if (!report.ok())
    report.print(); // which input disagrees, both derivatives, both errors

  printf("%zu derivatives checked, %d agree\n", report.entries().size(),
         (int)report.ok()); // prints: 2 derivatives checked, 1 agree
}
// docs-end-check-derivatives
