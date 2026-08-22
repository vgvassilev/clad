// RUN: %cladclang -fopenmp %s -I%S/../../include -fsyntax-only -Xclang -verify
// REQUIRES: OpenMP

// The adjoint of a private variable is accumulated per thread, so each copy
// has to start at the identity -- a reduction, which needs a type '+' accepts.
// For anything else the adjoint stays private and uninitialised, and clad says
// so rather than emitting a clause that cannot compile in the user's file.

#include "clad/Differentiator/Differentiator.h"

struct Pair {
  double a, b;
};

double fn(const double* x, int n) {
  double out = 0;
  Pair p{0, 0};
#pragma omp parallel for private(p) reduction(+ : out)
  // expected-warning@-1 {{adjoints of private variables are accumulated per thread}}
  for (int i = 0; i < n; i++) {
    p.a = x[i];
    p.b = x[i];
    out += p.a + p.b;
  }
  return out;
}

int main() { clad::gradient(fn); }
