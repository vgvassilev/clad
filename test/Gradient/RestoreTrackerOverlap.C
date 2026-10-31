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
//
// g2's tracker holds s.a and then all of s, which shares s.a's address. The
// record of s must not be dropped as a repeated store of s.a: k changes s.b
// and its pullback leaves it changed, so only that record puts back the s.b
// that y's adjoint reads.

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

// s.b += 1 has nothing to propagate: the pullback is empty, and the forward
// sweep calls k itself.
void k(S& s) { s.b += 1; }
namespace clad {
namespace custom_derivatives {
void k_pullback(S& s, S* d_s) {}
} // namespace custom_derivatives
} // namespace clad

void g2(S& s, double x) {
  s.a = x * s.a;
  k(s);
}

double f2(double x) {
  S s{1, 2};
  double y = s.b * x;
  g2(s, x);
  return s.a * s.b + y;
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

  dx = 0;
  clad::gradient(f2).execute(3, &dx);
  // f2 = x * (2 + 1) + 2x => df2/dx = 5
  printf("dx=%.2f\n", dx); // CHECK-EXEC-NEXT: dx=5.00

  S p{1, 2};
  clad::restore_tracker t;
  t.store(p);
  p = {3, 4};
  t.store(p.b);
  p.b = 5;
  t.restore();
  printf("%.0f %.0f\n", p.a, p.b); // CHECK-EXEC-NEXT: 1 2

  // q.a and then all of q, which shares q.a's address, on the tracker
  // directly; and the same again after 128 records of other values.
  S q{1, 2};
  clad::restore_tracker t2;
  t2.store(q.a);
  q.a = 3;
  t2.store(q);
  q = {5, 6};
  t2.restore();
  printf("%.0f %.0f\n", q.a, q.b); // CHECK-EXEC-NEXT: 1 2

  static double pad[256];
  S r{1, 2};
  clad::restore_tracker t3;
  for (int i = 0; i < 256; i += 2)
    t3.store(pad[i]);
  t3.store(r.a);
  r.a = 3;
  t3.store(r);
  r = {5, 6};
  t3.restore();
  printf("%.0f %.0f\n", r.a, r.b); // CHECK-EXEC-NEXT: 1 2
}
