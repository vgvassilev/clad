// RUN: %cladclang %s -I%S/../../include -fsyntax-only -Xclang -verify

#include "clad/Differentiator/Differentiator.h"

double product(double x, double y) { return x * y; }

namespace clad::custom_derivatives {
void product_pullback(double x, double, double seed, clad::no_adjoint,
                      double* dy) { // expected-note@-1 {{candidate 'product_pullback'}}
  *dy += x * seed;
}
} // namespace clad::custom_derivatives

double loss(double x, double y) {
  return product(x, y); // expected-error {{user-defined derivative for 'product' was provided but not used}}
}

void check() {
  // Missing support for dx is not a declaration that dx is zero.
  clad::gradient(loss);
}

double compact(double x, double y) { return x * y; }

namespace clad::custom_derivatives {
void compact_pullback(double x, double, double seed,
                      double* dy) { // expected-note@-1 {{candidate 'compact_pullback'}}
  *dy += x * seed;
}
} // namespace clad::custom_derivatives

double compact_loss(double x, double y) {
  return compact(x, y); // expected-error {{user-defined derivative for 'compact' was provided but not used; expected signature 'void (double, double, double, double *, double *)'}}
}

void check_compact() {
  // A single output with no marker cannot identify whether it is dx or dy.
  clad::gradient<clad::opts::assume_unselected_inactive>(compact_loss, "x");
}
