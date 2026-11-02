// A derivative that calls an operator, explained with remarks: the operator
// has no identifier, so it is not a push onto a tape, and nothing asks it for
// a name.
//
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -Rclad-analysis=tbr \
// RUN:   -fsyntax-only %s -I%S/../../include

#include "clad/Differentiator/Differentiator.h"

struct V {
  double x;
};
V operator*(V a, double s) { return {a.x * s}; }

double f(double x) {
  V v{x};
  V w = v * x;
  return w.x * x;
}

int main() { clad::gradient(f); }
