// RUN: %cladclang %s -I%S/../../include -oRestoreTrackerOverlap.out
// RUN: ./RestoreTrackerOverlap.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr %s \
// RUN:   -I%S/../../include -oRestoreTrackerOverlap.out
// RUN: ./RestoreTrackerOverlap.out | %filecheck_exec %s

// g's tracker holds s.b and then all of s, taken after s.b changed. Restoring
// must leave the older s.b, which y's adjoint reads. The opposite order, all
// of s and then s.b after s changed, is checked on the tracker directly: the
// derivatives clad generates today restore s from their own copy right after,
// which hides what the tracker put back.

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

struct S {
  double a, b;
};

void h(S& s, double x) {
  s.a *= x;
  s.b *= x;
}

void g(S& s, double x) {
  s.b = x * s.a + s.b;
  h(s, x);
}

double f(double x) {
  S s{1, 2};
  double y = s.b * x;
  g(s, x);
  return s.a * s.b + y;
}

int main() {
  double dx = 0;
  clad::gradient(f).execute(3, &dx);
  // f = x^3 + 2x^2 + 2x => df/dx = 3x^2 + 4x + 2
  printf("dx=%.2f\n", dx); // CHECK-EXEC: dx=41.00

  S p{1, 2};
  clad::restore_tracker t;
  t.store(p);
  p = {3, 4};
  t.store(p.b);
  p.b = 5;
  t.restore();
  printf("%.0f %.0f\n", p.a, p.b); // CHECK-EXEC-NEXT: 1 2
}
