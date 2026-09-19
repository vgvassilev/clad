// RUN: %cladclang -std=c++17 -fsyntax-only -I%S/../../include -Xclang -verify %s

#include "clad/Differentiator/Differentiator.h"

struct Ref {
  double& a; // expected-error {{reference data members are not supported}}
};

double f(double x) {
  Ref r{x};
  return r.a * 2.0;
}

void test_call() {
  auto g = clad::gradient(f);
}