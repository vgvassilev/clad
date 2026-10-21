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

// The closure would be rebuilt every iteration, with that iteration's captures.
double in_loop_forward(double x) {
  double s = 0;
  for (int i = 0; i < 3; ++i) {
    auto g = [x, i](double t) { return x * t * i; }; // expected-error {{differentiation of a lambda with captures created inside a loop is not supported yet}}
    s += g(x);
  }
  return s;
}

double in_loop_reverse(double x) {
  double s = 0;
  for (int i = 0; i < 3; ++i) {
    auto g = [x, i](double t) { return x * t * i; }; // expected-error {{differentiation of a lambda with captures created inside a loop is not supported yet}}
    s += g(x);
  }
  return s;
}

struct Scaled {
  double k = 3;
  double forward(double x) {
    auto f = [this](double t) { return k * t; }; // expected-error {{differentiation of a lambda that captures 'this' is not supported yet}}
    return f(x);
  }
  double reverse(double x) {
    auto f = [this](double t) { return k * t; }; // expected-error {{differentiation of a lambda that captures 'this' is not supported yet}}
    return f(x);
  }
};

double vector_mode(double x, double y) {
  auto f = [y](double t) { return t * t * y; }; // expected-error {{vector-mode differentiation of a lambda with captures is not supported yet}}
  return f(x);
}

int main() {
  clad::differentiate(&Scaled::forward, "x");
  clad::gradient(&Scaled::reverse);
  clad::differentiate<clad::opts::vector_mode>(vector_mode, "x,y");
  clad::differentiate(mutable_forward, 0);
  clad::differentiate(in_loop_forward, 0);
  clad::gradient(in_loop_reverse);
  clad::gradient(mutable_reverse);
  clad::differentiate(byref_write_forward, 0);
  clad::gradient(byref_write_reverse);
  clad::gradient(byref_incr_reverse);
}
