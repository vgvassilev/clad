// RUN: %cladclang %s -I%S/../../include -fsyntax-only -Xclang -verify

// Reverse mode differentiates a real function of one argument through its
// pushforward, whose callees then need a pushforward too.

#include "clad/Differentiator/Differentiator.h"

double two(double a, double b) { return a * b; }

double through_two(double x) {
  return two(1.0, x); // expected-warning {{no pushforward of 'two' was derived; its contribution to the derivative is considered 0}}
}
double f1(double x) { return through_two(x); }

double through_lambda(double x) {
  auto mul = [](double a, double b) { return a * b; };
  return mul(1.0, x); // expected-warning {{no pushforward of 'operator()' was derived; its contribution to the derivative is considered 0}}
}
double f2(double x) { return through_lambda(x); }

int main() {
  clad::gradient(f1);
  clad::gradient(f2);
}
