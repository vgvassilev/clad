// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -Rclad-analysis=liveness \
// RUN:   -fsyntax-only %s -I%S/../../include 2>&1 | %filecheck %s
// UNSUPPORTED: clang-14, clang-15, clang-16

#include "clad/Differentiator/Differentiator.h"

struct S {
  double a, b;
};

void step(S& s, double k) {
  s.a = s.a * s.b;
  s.b = s.b * k;
}

// A lambda that captures s by reference reads it without naming it.
double captured(double x) {
  S s{x, 2};
  auto peek = [&s]() { return s.a; };
  double r = s.a * x;
  step(s, x);
  // CHECK: :[[@LINE-1]]:3: remark: clad runs this call in the forward sweep, although nothing reads what it computes there
  // CHECK: :[[@LINE-2]]:3: note: the function takes the address of something this call writes, so a read through it cannot be ruled out
  // CHECK: note: to avoid this, make it a call whose results nothing reads (CLAD1006)
  return r + peek();
}

void gradients() { clad::gradient(captured); }
