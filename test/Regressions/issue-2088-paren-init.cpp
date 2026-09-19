// RUN: %cladclang -std=c++20 %s -I%S/../../include -fsyntax-only -Xclang -verify
// UNSUPPORTED: clang-11, clang-12, clang-13, clang-14, clang-15, clang-16

#include "clad/Differentiator/Differentiator.h"

// The fourth kind from issue #2088, kept apart because
// CXXParenListInitExpr -- parenthesised aggregate initialisation -- is a
// clang 17 node, so older trees have nothing to clone and nothing to diagnose.

struct P {
  double a, b;
};

// expected-warning@+1 {{statement kind 'CXXParenListInitExpr' is not supported}}
double paren_init(double x) { P p(x, 2.0); return p.a; }

int main() { clad::differentiate(paren_init, "x"); }
