// A continue inside a switch continues the enclosing loop. Reverse mode
// cannot record that yet, so it says so rather than build a derivative that
// pops a tape the iteration never pushed.
// RUN: %cladclang %s -I%S/../../include -fsyntax-only -Xclang -verify

#include "clad/Differentiator/Differentiator.h"

double continued(double x) {
  double s = 0;
  for (int i = 0; i < 4; ++i) {
    switch (i) {
    case 0:
      s += x;
      continue; // expected-error {{reverse-mode differentiation of a 'continue' inside a 'switch' is not supported yet}}
    default:
      break;
    }
    s *= x;
  }
  return s;
}

// The return counts the continue among the loop's own.
double continuedThenReturned(double x) {
  double s = 0;
  for (int i = 0; i < 4; ++i) {
    switch (i) {
    case 0:
      s += x;
      continue; // expected-error {{reverse-mode differentiation of a 'continue' inside a 'switch' is not supported yet}}
    default:
      break;
    }
    if (i == 2)
      return s;
    s *= x;
  }
  return s;
}

void request() {
  clad::gradient(continued);
  clad::gradient(continuedThenReturned);
}
