// RUN: %cladclang %s -I%S/../../include -oRestoreTrackerOverlap.out 2>&1 \
// RUN:   | %filecheck %s
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

// sq2 writes o through a pointer, and its custom pullback reads only x: it
// does not replay sq2. norm2's pullback reads o as each iteration left it,
// which the caller has to put back itself.
void sq2(const double* x, double* out) {
  out[0] = x[0] * x[0];
  out[1] = x[1] * x[1];
}
namespace clad {
namespace custom_derivatives {
void sq2_pullback(const double* x, double* out, double* d_x, double* d_out) {
  for (int i = 0; i < 2; i++) {
    d_x[i] += 2 * x[i] * d_out[i];
    d_out[i] = 0;
  }
}
} // namespace custom_derivatives
} // namespace clad

double norm2(const double* v) { return v[0] * v[0] + v[1] * v[1]; }

double f3(double x) {
  double o[2];
  double r = 0;
  for (int k = 1; k <= 3; k++) {
    double y[2] = {x * k, 2};
    sq2(y, o);
    r += norm2(o);
  }
  return r;
}

// fill writes p[0..n): handed a pointer into buf, not buf itself, the
// caller puts back as much as the extent says.
void fill(double* p, int n, double x) {
  for (int i = 0; i < n; i++)
    p[i] = x * (i + 1);
}
namespace clad {
namespace custom_derivatives {
void fill_pullback(double* p, int n, double x, double* d_p, int* d_n,
                   double* d_x) {
  for (int i = 0; i < n; i++) {
    *d_x += (i + 1) * d_p[i];
    d_p[i] = 0;
  }
}
} // namespace custom_derivatives
} // namespace clad

double f4(double x) {
  double buf[3] = {0, 0, 0};
  double r = 0;
  for (int m = 1; m <= 3; m++) {
    fill(&buf[1], 2, x * m);
    r += buf[1] * buf[2];
  }
  return r;
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

  dx = 0;
  clad::gradient(f3).execute(1.5, &dx);
  // f3 = (1 + 16 + 81) x^4 + 48 => df3/dx = 392 x^3
  printf("dx=%.2f\n", dx); // CHECK-EXEC-NEXT: dx=1323.00

  dx = 0;
  clad::gradient(f4).execute(1.5, &dx);
  // buf[1] = x m, buf[2] = 2 x m => f4 = 28 x^2 => df4/dx = 56 x
  printf("dx=%.2f\n", dx); // CHECK-EXEC-NEXT: dx=84.00

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

// The call writes all of the local array o, and its pullback does not replay
// it: the array is put back before the pullback and after it.
// CHECK: void f3_grad(double x, double *_d_x) {
// CHECK:          clad::record_range(_rec0, o, 2UL);
// CHECK-NEXT:     sq2(y, o);
// CHECK:              clad::peek_range(_rec0, o, 2UL);
// CHECK-NEXT:         clad::custom_derivatives::sq2_pullback(y, o, _d_y, _d_o);
// CHECK-NEXT:         clad::peek_range(_rec0, o, 2UL);
// CHECK-NEXT:         clad::drop_range(_rec0, 2UL);

// Of buf, the two elements fill writes.
// CHECK: void f4_grad(double x, double *_d_x) {
// CHECK:          clad::record_range(_rec0, &buf[1], 2UL);
// CHECK-NEXT:     fill(&buf[1], 2, x * m);
// CHECK:              clad::peek_range(_rec0, &buf[1], 2UL);
// CHECK:              clad::custom_derivatives::fill_pullback(&buf[1], 2, x * m, &_d_buf[1], &_r0, &_r1);
// CHECK-NEXT:         clad::peek_range(_rec0, &buf[1], 2UL);
// CHECK-NEXT:         clad::drop_range(_rec0, 2UL);
