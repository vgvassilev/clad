// RUN: %cladclang -fsyntax-only -Xclang -verify %s -I%S/../../include

// Differentiating a call to a declaration-only operator used to schedule its
// derivative and call FunctionDecl::getName() on it, which asserts because an
// operator's name is not a simple identifier. Report it like any other
// undefined function instead. This is the non-Kokkos reduction of a crash
// hit while differentiating Kokkos parallel dispatch (a KOKKOS_LAMBDA's
// operator()).

#include "clad/Differentiator/Differentiator.h"

struct S {
  double operator()(double y) const; // declared, never defined
};

double f(double x) {
  S s;
  // expected-warning@+2 {{attempted differentiation of function 'operator()' without definition and no suitable overload was found in namespace 'custom_derivatives'}}
  // expected-note@+1 {{numerical differentiation is not viable for 'operator()'; considering 'operator()' as 0}}
  return s(x);
}

int main() {
  auto d = clad::differentiate(f, "x");
  (void)d;
}
