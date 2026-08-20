// RUN: %cladclang %s -I%S/../../include -fsyntax-only -Xclang -verify 2>&1

#include "clad/Differentiator/Differentiator.h"

double f(double a, const double *b) {
  return b[0] * a + b[1] * a + b[2] * a;
}

// A non-const pointer parameter outside the requested set has no tangent to
// pass and no const pointee to read a null one as zero through.
double g(double a, double *b) { // expected-error 2 {{dependent non-const pointer and array parameters are not supported; differentiate w.r.t. 'b' or mark it const}}
  return b[0] * a;
}

int main() {
    clad::hessian(f, 1); // expected-error {{hessian mode differentiation w.r.t. array or pointer parameters needs explicit declaration of the indices of the array using the args parameter; did you mean 'clad::hessian}}
    clad::hessian(f, "a");
    clad::hessian(f, "a, b"); // expected-error {{hessian mode differentiation w.r.t. array or pointer parameters needs explicit declaration of the indices of the array using the args parameter; did you mean 'clad::hessian}}
    clad::hessian(f, "a, b[0:2]");
    clad::hessian(g, "a");
};