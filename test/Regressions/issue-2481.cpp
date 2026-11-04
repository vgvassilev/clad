// The adjoint of a copy starts at zero. Its own gradient reaches the object
// it was copied from through the constructor's pullback, so starting from
// that object's adjoint adds whatever it already holds a second time.
// RUN: %cladclang %s -I%S/../../include -oissue-2481.out -Xclang -verify 2>&1
// RUN: ./issue-2481.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"

extern "C" int printf(const char* fmt, ...);

struct A { double x; };

// By value: the parameter is a copy, and its adjoint is the one the pullback
// accumulates into. Called twice, so the second call sees an adjoint the
// first one has already written to.
double g1(A a) { return a.x * a.x; }
double f1(double x) { A a{x}; return g1(a) + g1(a); }

// The same through a copy declared in the callee.
double g2(const A& a) { A u = a; return u.x * u.x; }
double f2(double x) { A a{x}; return g2(a) + g2(a); }

// Declared in a loop, where initializing from the source's adjoint also put
// the copy's adjoint above the declaration it named.
double f3(double x) {
  A a{x};
  double s = 0;
  for (int i = 0; i < 2; ++i) {
    A u = a;
    s += u.x * x;
  }
  return s;
}

int main() { // expected-no-diagnostics
  double dx = 0;
  clad::gradient(f1).execute(3, &dx);
  printf("by value: %.2f\n", dx); // CHECK-EXEC: by value: 12.00

  dx = 0;
  clad::gradient(f2).execute(3, &dx);
  printf("by copy: %.2f\n", dx); // CHECK-EXEC: by copy: 12.00

  dx = 0;
  clad::gradient(f3).execute(3, &dx);
  printf("in a loop: %.2f\n", dx); // CHECK-EXEC: in a loop: 12.00
}
