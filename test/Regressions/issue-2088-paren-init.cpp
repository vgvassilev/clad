// RUN: %cladclang -std=c++20 %s -I%S/../../include -fsyntax-only -Xclang -verify
// UNSUPPORTED: clang-12, clang-13, clang-14, clang-15

#include "clad/Differentiator/Differentiator.h"

// The fourth kind from issue #2088, kept apart because
// CXXParenListInitExpr -- parenthesised aggregate initialisation, P0960R3 --
// arrived in clang 16, so older trees have nothing to clone and nothing to
// diagnose.

struct P {
  double a, b;
};

// expected-warning@+1 {{statement kind 'CXXParenListInitExpr' is not supported}}
double paren_init(double x) { P p(x, 2.0); return p.a; }

// A union puts the member it initialises in the same slot an array filler
// would occupy, and a clone that drops it leaves CodeGen looking at a union
// with no member chosen.
union U {
  double a;
  int b;
};

// expected-warning@+1 {{statement kind 'CXXParenListInitExpr' is not supported}}
double union_init(double x) { U u(x); return u.a; }

// An array initialised this way is a gap rather than a fix: clad's array
// handling knows InitListExpr and not this node, so the clone is refused and
// said to be refused. The second diagnostic is the cascade from carrying on
// after an error, not a separate problem; both are here so that a later change
// to either has to say so.
// The cascade carries no source location of its own, hence the `@*`.
// expected-error@* {{cannot initialize an array element of type 'double' with an rvalue of type 'double[3]'}}
// expected-warning@+2 {{statement kind 'CXXParenListInitExpr' is not supported}}
// expected-error@+1 {{clad cannot differentiate this function: no clone exists for statement kind 'CXXParenListInitExpr'}}
double array_init(double x) { double arr[3](x, 2.0); return arr[0]; }

int main() {
  clad::differentiate(paren_init, "x");
  clad::differentiate(union_init, "x");
  clad::differentiate(array_init, "x");
}
