// RUN: %cladclang %s -I%S/../../include -fsyntax-only -Xclang -verify

#include "clad/Differentiator/Differentiator.h"

struct NonAssignable {
  double x;
  NonAssignable operator=(const NonAssignable&) = delete;
};

double used(double x) {
  double r = 0;
  for (int i = 0; i < 2; ++i) {
    NonAssignable u{x}; // expected-error {{cannot differentiate local variable 'u' of type 'NonAssignable': reverse-mode promotion requires an unavailable assignment operator}}
    r += u.x;
  }
  return r;
}

int main() { clad::gradient(used); }