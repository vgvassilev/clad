// RUN: %cladclang %s -I%S/../../include -fsyntax-only -Xclang -verify

#include "clad/Differentiator/Differentiator.h"

double from_pointer(double* p) { return *p; }
namespace clad {
namespace custom_derivatives {
void from_pointer_pullback(const double*, double, double*) {}
} // namespace custom_derivatives
} // namespace clad

void request_custom() {
  clad::pullback(from_pointer);
  // expected-error@* {{cannot initialize a parameter of type}}
  // expected-note@* {{passing argument to parameter 'derivedFn' here}}
}

double passive(double x, double y __attribute__((annotate("non_differentiable")))) {
  return x * y;
}
template <int N> double times(double x) { return x * N; }

void in_place(double* p) {}
namespace clad {
namespace custom_derivatives {
void in_place_reverse_forw(double*, double*, clad::pullback_state<double>&) {}
void in_place_pullback(double*, double*, clad::pullback_state<double>) {}
} // namespace custom_derivatives
} // namespace clad

void request() {
  clad::pullback(passive); // expected-error {{cannot expose the pullback of 'passive'}}
  clad::pullback(times<3>); // expected-error {{cannot expose the pullback of 'times}}
  clad::pullback(in_place); // expected-error {{this custom pullback requires state from its forward pass}}
}

using Unary = double (*)(double);
double square(double x) { return x * x; }
double triple(double x) { return 3 * x; }

void indirect(Unary target, bool choose) {
  clad::pullback(target); // expected-error {{requires a function address known at compile time}}
  Unary local = square;
  local = triple;
  clad::pullback(local); // expected-error {{requires a function address known at compile time}}
  clad::pullback(choose ? square : triple); // expected-error {{requires a function address known at compile time}}
  clad::pullback(static_cast<Unary>(nullptr)); // expected-error {{requires a function address known at compile time}}
  clad::pullback(+[](double x) { return x * x; }); // expected-error {{cannot differentiate a lambda converted to a function pointer}}
}
