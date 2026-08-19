// RUN: %cladclang %s -I%S/../../include -fsyntax-only -Xclang -verify

// Differentiating a subset with the Enzyme backend describes each argument
// with an activity marker. A parameter that is active but neither a real nor
// a pointer has no shadow to pass and no slot in the returned struct, so the
// call cannot express it. Refuse rather than emit a call whose arguments no
// longer line up, which would leave the adjoint untouched and the gradient
// silently short.

#include "clad/Differentiator/Differentiator.h"

struct Pair {
  double a, b;
};

double byRecord(Pair p, double y) { return p.a * y; }

double byField(Pair p, double y) { return p.a * y; }

int main() {
  // expected-error@+1 {{cannot differentiate 'p' with the Enzyme backend: only real and pointer parameters are supported}}
  clad::gradient<clad::opts::use_enzyme>(byRecord, "p");
  // expected-error@+1 {{cannot differentiate 'p' with the Enzyme backend: only real and pointer parameters are supported}}
  clad::gradient<clad::opts::use_enzyme>(byField, "p.a");
}
