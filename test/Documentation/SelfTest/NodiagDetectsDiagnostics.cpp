// RUN: %cladclang %s -I%S/../../../include -o%t 2>%t.log
// RUN: ! %filecheck_nodiag %s < %t.log
//
// %filecheck_nodiag guards every documentation example against compiling with
// a diagnostic, which is how an example rots: clad stops recognising something,
// says so, falls back, and the example still runs and still prints a plausible
// number. A guard nobody tests is a guard that can be switched off by accident,
// which is what happened to the output check in the exec-only view. So provoke
// a diagnostic here and require that the guard catches it.

#warning this diagnostic is the point of the test

#include "clad/Differentiator/Differentiator.h"

double f(double x) { return x * x; }

int main() {
  auto d_f = clad::differentiate(f, "x");
  return d_f.execute(3) == 6 ? 0 : 1;
}
