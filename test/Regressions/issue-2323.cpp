// RUN: %cladclang -fsyntax-only -Xclang -verify -std=c++17 -I%S/../../include %s

// Issue #2323: Vector mode crashes on a callee that returns a struct.
// Vector mode does not support differentiating functions returning a struct or
// class type because record returns have no vector tangent form.

#include "clad/Differentiator/Differentiator.h"

struct S { double v; };
S mk(double x) { return S{x * x}; } // expected-note 2 {{'mk' declared here with return type 'S'}}

double f(double x, double y) {
  return mk(x).v * y; // expected-error {{cannot differentiate call to 'mk' in vector mode: functions returning a struct or class are not supported yet}}
}

int main() {
  auto d = clad::differentiate<clad::opts::vector_mode>(f);
  (void)d;
  auto d2 = clad::differentiate<clad::opts::vector_mode>(mk); // expected-error {{vector mode currently does not support function 'mk' returning a struct or class type}}
  (void)d2;
}

