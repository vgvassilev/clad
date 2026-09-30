// RUN: %cladclang %s -I%S/../../include -fsyntax-only -Xclang -verify

#include "clad/Differentiator/Differentiator.h"

double selection(double x, double y) { return x * y; }
double passive_selection(double x,
                         double y __attribute__((annotate("non_differentiable")))) {
  return x * y;
}
double all_passive(double x __attribute__((annotate("non_differentiable")))) {
  return x * x;
}

// Simulate the libc++ internal type on the libstdc++ coverage runner. The
// declaration alone is enough: the public boundary must reject it before
// requesting a definition or trying to generate a malformed derivative.
#ifndef _LIBCPP_VERSION
namespace std { struct __nat; }
double reserved_tag(std::__nat tag, double x);
#endif

double* stateful(double* p) { return p; }
namespace clad {
namespace custom_derivatives {
clad::ValueAndAdjoint<double*, double*>
stateful_reverse_forw(double* p, double* d_p,
                      clad::pullback_state<double>& state) {
  return {p, d_p};
}
void stateful_pullback(double* p, double* d_p,
                       clad::pullback_state<double> state) {}
} // namespace custom_derivatives
} // namespace clad

int main() {
  clad::pullback(selection, "999999999999999999999999999999999999999"); // expected-error {{could not parse argument index}}
  clad::pullback(selection, "0junk"); // expected-error {{could not parse argument index}}
  clad::pullback(selection, ",x"); // expected-error {{empty parameter name in differentiation argument list}}
  clad::pullback(selection, "x,,y"); // expected-error {{empty parameter name in differentiation argument list}}
  clad::pullback(selection, "x,"); // expected-error {{empty parameter name in differentiation argument list}}
  clad::pullback(selection, "0,0"); // expected-error {{requested parameter 'x' was specified multiple times}}
  clad::pullback(selection, "0,x"); // expected-error {{requested parameter 'x' was specified multiple times}}
  clad::pullback(passive_selection, "y"); // expected-error {{is marked non-differentiable and cannot be selected for pullback}}
  clad::pullback(passive_selection, 1); // expected-error {{is marked non-differentiable and cannot be selected for pullback}}
  clad::pullback(all_passive); // expected-error {{pullback requires at least one differentiable parameter}}

#ifndef _LIBCPP_VERSION
  clad::pullback(reserved_tag); // expected-error {{pullback does not support functions with std::__nat parameters}}
#endif
  clad::pullback(stateful); // expected-error {{the public pullback interface cannot expose a custom pullback_state parameter}}

  // The parser is shared with gradient: malformed input must fail without an
  // exception or an assertion there too, and later requests remain usable.
  clad::gradient(selection, "999999999999999999999999999999999999999"); // expected-error {{could not parse argument index}}
  clad::gradient(selection, ",x"); // expected-error {{empty parameter name in differentiation argument list}}
  auto good = clad::pullback(selection, 1);
  double dy = 0;
  good.execute(2., 3., 1., nullptr, &dy);
}
