// RUN: %cladclang -fsyntax-only -Xclang -verify -std=c++20 -I%S/../../include %s

// Regression test for https://github.com/vgvassilev/clad/issues/2088.
//
// The VisitStmt fallback diagnoses a statement clad cannot differentiate and
// then clones it, so clonable-but-unsupported nodes such as a try/catch block
// (FirstDerivative/DiffInterface.C, ForwardMode/VectorModeInterface.C) still
// reach the derivative. A few kinds have no StmtClone case at all: cloning one
// asserts in StmtClone::VisitStmt or, with asserts off, builds a malformed node
// that ReferencesUpdater crashes walking. Those are diagnosed and dropped
// instead, and the compile finishes with only the diagnostic.
//
// This file covers three of the four families from the issue. C++20
// parenthesized aggregate initialization produces a kind that only exists from
// Clang 16 on, so it lives in issue-2088-paren-init.cpp.

#include "clad/Differentiator/Differentiator.h"

// The GNU binary conditional operator `x ?: y`, from the issue. Forward mode
// drops the unsupported initializer and still differentiates the rest.
double fwd(double x) {
  double y = x ?: 1.0; // expected-warning {{statement kind 'BinaryConditionalOperator' is not supported}} // expected-warning {{statement kind 'BinaryConditionalOperator' is not supported}}
  return y;
}

// Reverse mode with the unsupported node as the returned value, the form from
// the issue.
double rev(double x) {
  return x ?: 1.0; // expected-warning {{statement kind 'BinaryConditionalOperator' is not supported}}
}

// `[[likely]]` on a statement wraps that statement in an AttributedStmt, which
// StmtClone has no case for.
// expected-warning@* 1+ {{statement kind 'AttributedStmt' is not supported}}
double f_likely(double x) {
  double r = x;
  if (r > 0) [[likely]] {
    r *= 2;
  }
  return r;
}

// The base initializer of an inheriting constructor is a
// CXXInheritedCtorInitExpr, which also has no StmtClone case.
// expected-warning@* 1+ {{statement kind 'CXXInheritedCtorInitExpr' is not supported}}
struct Base {
  double b;
  Base() : b(0) {}
  Base(double x) : b(x) {}
};

struct Derived : Base {
  using Base::Base;
};

double f_inherited(double x) {
  Derived d(x);
  return d.b;
}

int main() {
  clad::gradient(fwd, "x");
  clad::differentiate(fwd, "x");
  clad::gradient(rev, "x");
  clad::differentiate(f_likely, "x");
  clad::gradient(f_inherited, "x");
}
