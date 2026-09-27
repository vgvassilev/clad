// RUN: %cladclang %s -I%S/../../include -fsyntax-only -Xclang -verify

#include "clad/Differentiator/Differentiator.h"

// An adjoint whose initializer has no derivative would be left uninitialized
// for the reverse sweep to read, which answers with a wrong gradient rather
// than a missing one, so the declaration is refused instead.
//
// A lambda yielding a pointer to a pointer is one way to reach that: the
// adjoint of `r` needs the derivative of the call, which clad does not
// build. This is a limitation of its own, unrelated to which declarations the
// activity analysis calls varied -- the same happens with the analysis off.
double f(double x) {
  double y = 0;
  double* p = &y;
  // expected-warning@+2 {{direct lambda calls are not supported, ignored}}
  // expected-error@+1 {{derivative of the initializer of 'r' is not available; the computed gradient would be incorrect}}
  double** r = [&] { return &p; }();
  **r = x * x;
  return y;
}

int main() { auto g = clad::gradient(f, "x"); }
