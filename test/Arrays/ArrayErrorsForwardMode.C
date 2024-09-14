// RUN: %cladclang %s -I%S/../../include -fsyntax-only -Xclang -verify 2>&1

#include "clad/Differentiator/Differentiator.h"

double addArr(double *arr) {
  return arr[0] + arr[1] + arr[2] + arr[3];
}

struct Double {
  double n;
};

double addDoubleArr(Double *arr) { // expected-error {{attempted differentiation w.r.t. a parameter ('arr') which is not an array or pointer of a real type}}
  return arr[0].n + arr[1].n + arr[2].n + arr[3].n;
}

int main() {
  clad::differentiate(addArr, "arr[1:2]"); // expected-error {{Forward mode differentiation w.r.t. several parameters at once is not supported, call 'clad::differentiate' for each parameter separately}}

  clad::differentiate(addArr, "arr[2:1]"); // expected-error {{Range specified in 'arr[2:1]' is in incorrect format}}

  clad::differentiate(addDoubleArr, "arr[1]");
}
