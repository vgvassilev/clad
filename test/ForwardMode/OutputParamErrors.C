// RUN: %cladclang %s -I%S/../../include -fsyntax-only -Xclang -verify

#include "clad/Differentiator/Differentiator.h"

// A void-returning function's forward derivative returns the tangent of the
// parameter it writes through. Writing through more than one leaves no single
// tangent to return, and choosing among them would be a guess, so this is
// refused rather than answered with a derivative whose result is unreachable.
void two_outputs(double i, double& a, double& b) {
  // expected-note@-1 {{'a' is written through here}}
  // expected-note@-2 {{'b' is written through here}}
  a = i;
  b = i * i;
}

// The wrapper the diagnostic points at is how one output is asked for.
double wrapper(double i) {
  double a = 0, b = 0;
  two_outputs(i, a, b);
  return b;
}

// An output parameter is a non-const reference to a floating point type. An
// `int&` a function counts through is not one, so this has a single output
// and differentiates; counting it would refuse a function that has exactly
// one real output.
void with_counter(double x, double& out, int& iterations) {
  out = x * x;
  iterations = 1;
}

// An enum reference is not one either. Clad cannot build a tangent for an
// enum, so treating it as an output would leave the derivative looking for a
// tangent that was never created.
enum Status { Ok, Bad };
void with_status(double x, double& out, Status& s) {
  out = x * x;
  // The enum assignment itself is nothing this change touches; clad has
  // always said it cannot carry a derivative through one.
  // expected-warning@+1 {{derivative of an assignment attempts to assign to unassignable expr, assignment ignored}}
  s = Ok;
}

// A custom derivative is matched on its whole signature, return type
// included, so a `_darg0` written when the derivative of a void primal
// returned void no longer matches the one clad now asks for. It is not
// silently dropped for clad's own: clad reports that one was provided and
// names the signature it expected.
void custom_out(double x, double& out);

namespace clad {
namespace custom_derivatives {
// expected-note@+1 {{candidate 'custom_out_darg0' has different return type ('double' expected but has 'void')}}
void custom_out_darg0(double x, double& out) { out = 42; }
} // namespace custom_derivatives
} // namespace clad

void custom_out(double x, double& out) { out = x * x; }

int main() {
  // expected-error@+1 {{attempted to differentiate 'two_outputs', which returns void and writes through 2 parameters; forward mode returns a single tangent, so differentiate a wrapper returning the output you need}}
  auto d = clad::differentiate(two_outputs, "i");
  auto ok = clad::differentiate(wrapper, "i");
  auto counted = clad::differentiate(with_counter, "x");
  auto status = clad::differentiate(with_status, "x");
  // expected-error@+1 {{user-defined derivative for 'custom_out' was provided but not used; expected signature 'double (double, double &)' does not match}}
  auto custom = clad::differentiate(custom_out, "x");
  (void)ok;
  (void)counted;
  (void)status;
  (void)custom;
}
