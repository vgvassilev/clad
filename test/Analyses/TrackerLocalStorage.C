// RUN: %cladclang %s -I%S/../../include -oTrackerLocalStorage.out 2>&1 \
// RUN:   | %filecheck %s
// RUN: ./TrackerLocalStorage.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr %s \
// RUN:   -I%S/../../include -oTrackerLocalStorage.out
// RUN: ./TrackerLocalStorage.out | %filecheck_exec %s

// A reverse_forw that owns the storage a nested call mutates must not record
// that storage in the tracker it received from its caller: those addresses
// die with the reverse_forw, and the caller's restore would write into freed
// memory. The shapes below reach that storage in every way the ownership
// check has to see through -- address-of over a user operator[], an accessor
// call, and a member array. Reaching a local through a pointer variable is
// not recognized (see designatesLocallyOwnedStorage), so that shape is not
// exercised here.

#include "clad/Differentiator/Differentiator.h"

#include <cstdio>

// The nested call may not record into the tracker the caller passed in.
// CHECK: void innerAddrOf_reverse_forw(
// CHECK: clad::restore_tracker _tracker_unused0 = {};

void twice(int n, const double* x, double* out) {
  for (int i = 0; i < n; i++)
    out[i] = 2 * x[i];
}

double squaresum(int n, const double* v) {
  double t = 0;
  for (int i = 0; i < n; i++)
    t += v[i] * v[i];
  return t;
}

struct Buf {
  double vals[2];
  double* data() { return vals; }
  double& operator[](int i) { return vals[i]; }
};

struct Holder {
  Buf b;
};

// Reached with `&buf[0]` -- address-of over a user operator[] on a local.
void innerAddrOf(const double* x, double s, double* err) {
  Buf buf;
  twice(2, x, &buf[0]);
  *err = s * squaresum(2, &buf[0]);
}

// Reached with a member accessor call on a local.
void innerAccessor(const double* x, double s, double* err) {
  Buf buf;
  twice(2, x, buf.data());
  *err = s * squaresum(2, buf.data());
}

// Reached through a member of a local, then a subscript.
void innerMember(const double* x, double s, double* err) {
  Holder h;
  twice(2, x, &h.b.vals[0]);
  *err = s * squaresum(2, &h.b.vals[0]);
}

#define OUTER(NAME, INNER)                                                     \
  double NAME(const double* x) {                                               \
    double t = 0;                                                              \
    for (int k = 1; k <= 2; k++) {                                             \
      double e = 0;                                                            \
      INNER(x, k, &e);                                                         \
      t += e;                                                                  \
    }                                                                          \
    return t;                                                                  \
  }

OUTER(outerAddrOf, innerAddrOf)
OUTER(outerAccessor, innerAccessor)
OUTER(outerMember, innerMember)

int main() {
  double x[2] = {1, 2};
  double dx[2];

#define TEST(fn)                                                               \
  {                                                                            \
    dx[0] = dx[1] = 0;                                                         \
    auto grad = clad::gradient(fn, "x");                                       \
    grad.execute(x, dx);                                                       \
    printf(#fn ": dx={%.2f, %.2f}\n", dx[0], dx[1]);                           \
  }

  // f = sum_{k=1,2} k * |2x|^2 = 12 |x|^2 => df/dx_i = 24 x_i
  TEST(outerAddrOf);   // CHECK-EXEC: outerAddrOf: dx={24.00, 48.00}
  TEST(outerAccessor); // CHECK-EXEC: outerAccessor: dx={24.00, 48.00}
  TEST(outerMember);   // CHECK-EXEC: outerMember: dx={24.00, 48.00}
}
