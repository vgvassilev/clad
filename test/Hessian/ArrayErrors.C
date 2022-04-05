// RUN: %cladclang %s -I%S/../../include -fsyntax-only -Xclang -verify 2>&1

#include "clad/Differentiator/Differentiator.h"

//CHECK-NOT: {{.*error|warning|note:.*}}
//XFAIL:*
double f(int a, double *b) {
  return b[0] * a + b[1] * a + b[2] * a;
}

int main() {
    clad::hessian(f, 1); // expected-error {{Hessian mode differentiation w.r.t. array or pointer parameters needs explicit declaration of the indices of the array using the args parameter; did you mean 'clad::hessian(f, "b[0:<last index of b>]")'}}
    clad::hessian(f, "a");
    clad::hessian(f, "a, b"); // expected-error {{Hessian mode differentiation w.r.t. array or pointer parameters needs explicit declaration of the indices of the array using the args parameter; did you mean 'clad::hessian(f, "a, b[0:<last index of b>]")'}}
    clad::hessian(f, "a, b[0:2]");
};