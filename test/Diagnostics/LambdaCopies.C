// RUN: %cladclang %s -I%S/../../include -fsyntax-only -Xclang -verify
// UNSUPPORTED: clang-11, clang-12, clang-13, clang-14, clang-15, clang-16

#include "clad/Differentiator/Differentiator.h"
#include <utility>

double copy_closure(double x) {
  auto inner = [x] { return x * x; };
  auto copy = inner; // expected-error {{differentiation of copied or moved closures is not supported; use a reference to the lambda instead}}
  return copy();
}

double move_closure(double x) {
  auto inner = [x] { return x * x; };
  auto moved = std::move(inner); // expected-error {{differentiation of copied or moved closures is not supported; use a reference to the lambda instead}}
  return moved();
}

double capture_closure(double x) {
  auto inner = [x] { return x * x; };
  auto outer = [inner] { return inner(); }; // expected-error {{differentiation of copied or moved closures is not supported; use a reference to the lambda instead}}
  return outer();
}

double temporary_reference(double x) {
  const auto& inner = [x] { return x * x; }; // expected-error {{differentiation of a closure reference requires a named lambda initializer}}
  return inner();
}

struct ThisCapture {
  double value;
  double evaluate(double x) {
    auto inner = [this](double y) { return value * y; }; // expected-error {{reverse-mode differentiation requires ordinary variable captures}}
    return inner(x);
  }

  double parenthesized(double x) {
    auto inner = ([this](double y) { return value * y; }); // expected-error {{reverse-mode differentiation requires ordinary variable captures}}
    return inner(x);
  }

  double braced(double x) {
    auto inner{[this](double y) { return value * y; }}; // expected-error {{reverse-mode differentiation requires ordinary variable captures}}
    return inner(x);
  }
};

void request_derivatives() {
  clad::gradient(copy_closure);
  clad::gradient(move_closure);
  clad::gradient(capture_closure);
  clad::gradient(temporary_reference);
  clad::gradient(&ThisCapture::evaluate);
  clad::gradient(&ThisCapture::parenthesized);
  clad::gradient(&ThisCapture::braced);
}
