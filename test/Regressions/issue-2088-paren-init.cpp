// RUN: %cladclang -fsyntax-only -Xclang -verify -std=c++20 -I%S/../../include %s
// UNSUPPORTED: clang-12, clang-13, clang-14, clang-15

// Regression test for https://github.com/vgvassilev/clad/issues/2088.
//
// C++20 parenthesized aggregate initialization (P0960R3) is represented by
// CXXParenListInitExpr, which StmtClone has no case for. The VisitStmt fallback
// diagnoses the initializer and drops it instead of cloning it into a malformed
// node that ReferencesUpdater would crash walking. Covered in both directions
// below. The expression only exists from Clang 16, where P0960R3 landed, hence
// the version gate.

#include "clad/Differentiator/Differentiator.h"

struct Aggregate {
  double a;
  double b;
};

double f_aggregate(double x) {
  // Forward and reverse each visit the initializer once.
  Aggregate p(x, 2.0); // expected-warning 1+ {{statement kind 'CXXParenListInitExpr' is not supported}}
  return x;
}

int main() {
  clad::differentiate(f_aggregate, "x");
  clad::gradient(f_aggregate, "x");
}
