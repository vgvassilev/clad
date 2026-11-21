// RUN: %cladclang %s -I%S/../../include -fsyntax-only -Xclang -verify

#include "clad/Differentiator/Differentiator.h"

// Reverse mode moves a local declared in a nested scope to function scope.
// When its type cannot be assigned, clad constructs each new value in the
// storage of the moved variable, which it cannot do without running a
// destructor the original function runs once per scope (#2431).

struct Guard {
  double x;
  Guard& operator=(const Guard&) = delete;
  ~Guard() {}
};

double loop(double x) {
  double r = 0;
  for (int i = 0; i < 2; ++i) {
    Guard g{x}; // expected-error {{reverse mode cannot move local variable 'g' of type 'Guard' out of its scope: the type cannot be assigned, and it is not trivially destructible}}
    r += g.x;
  }
  return r;
}

// The reverse sweep restores the variable from a stored copy.
struct Pinned {
  double x;
  Pinned() = default;
  explicit Pinned(double v) : x(v) {}
  Pinned(const Pinned&) = delete;
  Pinned& operator=(const Pinned&) = delete;
};

double pinned(double x) {
  double r = 0;
  if (x > 0) {
    Pinned p(x); // expected-error {{reverse mode cannot move local variable 'p' of type 'Pinned' out of its scope: the type cannot be assigned, and it cannot be copied or moved}}
    r = p.x * p.x;
  }
  return r;
}

int main() {
  clad::gradient(loop);
  clad::gradient(pinned);
}
