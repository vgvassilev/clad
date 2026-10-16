// RUN: %cladclang %s -DMODE=0 -I%S/../../include -fsyntax-only -Xclang -verify=expected,scalar
// RUN: %cladclang %s -DMODE=1 -I%S/../../include -fsyntax-only -Xclang -verify
#include "clad/Differentiator/Differentiator.h"
struct P { double v; };
CLAD_NONDIFFERENTIABLE double opaque(double x) { return x; }

double record_temporary(double x, bool enabled) {
  P p{x};
  P&& ref = enabled ? (opaque(x), static_cast<P&&>(P{x * x}))
                    : static_cast<P&&>(p); // scalar-error@-1 {{materialized non-scalar conditional results are not supported}}
  return ref.v;
}

double subobject_temporary(double x, bool enabled) {
  double r = x;
  double&& ref = enabled ? (opaque(x), static_cast<P&&>(P{x * x}).v)
                         : static_cast<double&&>(r); // expected-error@-1 {{materialized non-scalar conditional results are not supported}}
  return ref + r;
}

int main() {
#if MODE
  clad::differentiate<clad::opts::vector_mode>(subobject_temporary, "x");
#else
  clad::differentiate(record_temporary, "x");
  clad::differentiate(subobject_temporary, "x");
#endif
}
