// A call clad cannot differentiate and cannot numerically differentiate
// either is said to be so, and counts as zero.
// RUN: %cladnumdiffclang %s -I%S/../../include -fsyntax-only -Xclang -verify

#include "clad/Differentiator/Differentiator.h"

// numerical_diff calls the function with as many arguments as it has
// parameters, which would drop the rest of a variadic one's: the function
// falls back, but a call that passes more than its parameters does not.
double vs(double a, ...);

// expected-warning@+3 {{attempted differentiation of function 'vs'}}
// expected-note@+2 {{falling back to numerical differentiation for 'vs'}}
// expected-warning@+1 {{numerical differentiation of 'vs' is not viable for this call: it passes arguments past the parameters; considering its derivative as 0}}
double f(double x, double y) { return vs(x, y); }

// Called with its parameters alone, nothing is dropped.
// expected-warning@+2 2{{attempted differentiation of function 'vs'}}
// expected-note@+1 2{{falling back to numerical differentiation for 'vs'}}
double g(double x) { return vs(x); }

void request() {
  clad::gradient(f);
  clad::differentiate(g, "x");
  clad::gradient(g);
}
