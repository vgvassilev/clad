// RUN: %cladclang %s -I%S/../../include -fsyntax-only -Xclang -verify
// UNSUPPORTED: clang-11, clang-12, clang-13, clang-14, clang-15, clang-16

#include "clad/Differentiator/Differentiator.h"

double mutable_forward(double x) {
  double c = 2.0;
  auto g = [c](double t) mutable { c += 1; return t * c; }; // expected-error {{differentiation of a mutable lambda is not supported: its captures change between calls}}
  return g(x);
}

double mutable_reverse(double x) {
  double c = 2.0;
  auto g = [c](double t) mutable { c += 1; return t * c; }; // expected-error {{differentiation of a mutable lambda is not supported: its captures change between calls}}
  return g(x);
}

double byref_write_forward(double x) {
  double c = 2.0;
  auto g = [&c](double t) { c = c * t; return c; }; // expected-error {{differentiation of a lambda that assigns to a by-reference capture is not supported}}
  return g(x);
}

double byref_write_reverse(double x) {
  double c = 2.0;
  auto g = [&c](double t) { c = c * t; return c; }; // expected-error {{differentiation of a lambda that assigns to a by-reference capture is not supported}}
  return g(x);
}

double byref_incr_reverse(double x) {
  double c = 2.0;
  auto g = [&c](double t) { c++; return t * c; }; // expected-error {{differentiation of a lambda that assigns to a by-reference capture is not supported}}
  return g(x);
}

int main() {
  clad::differentiate(mutable_forward, 0);
  clad::gradient(mutable_reverse);
  clad::differentiate(byref_write_forward, 0);
  clad::gradient(byref_write_reverse);
  clad::gradient(byref_incr_reverse);
}
