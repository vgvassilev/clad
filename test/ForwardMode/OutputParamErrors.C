// RUN: %cladclang %s -I%S/../../include -fsyntax-only -Xclang -verify

#include "clad/Differentiator/Differentiator.h"

// A void-returning function's forward derivative returns the tangent of the
// parameter it writes through. Writing through more than one leaves no single
// tangent to return, and choosing among them would be a guess, so this is
// refused rather than answered with a derivative whose result is unreachable.

// expected-error@+1 {{attempted to differentiate 'two_outputs', which returns void and writes through 2 parameters; forward mode returns a single tangent, so differentiate a wrapper returning the output you need}}
void two_outputs(double i, double& a, double& b) {
  a = i;
  b = i * i;
}

// The wrapper the diagnostic points at is how one output is asked for.
double wrapper(double i) {
  double a = 0, b = 0;
  two_outputs(i, a, b);
  return b;
}

int main() {
  auto d = clad::differentiate(two_outputs, "i");
  auto ok = clad::differentiate(wrapper, "i");
  (void)ok;
}
