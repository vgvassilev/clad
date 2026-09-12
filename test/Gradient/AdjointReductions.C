// RUN: %cladclang %s -I%S/../../include -oAdjointReductions.out 2>&1 | %filecheck %s
// RUN: ./AdjointReductions.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr %s \
// RUN:   -I%S/../../include -oAdjointReductions.out
// RUN: ./AdjointReductions.out | %filecheck_exec %s

// `a[i]` read on every iteration of the j loop is a broadcast, so its adjoint
// is a reduction over that loop -- and `_d_a[i] +=` accumulated in place is a
// store to a loop-invariant address, the one thing that keeps LLVM from
// vectorising the loop around it. Where the loop analysis can vouch for it,
// the sum lives in a register across the loop and reaches memory once after
// it. The negative half is every shape that must keep the store where it was,
// because moving it would move it past something that reads or resets that
// adjoint.

#include "clad/Differentiator/Differentiator.h"

#include <cmath>
#include <cstdio>

// -- Reduced ---------------------------------------------------------------

// The outer index is invariant in the inner loop.
double outerIndex(const double* a, const double* b, int n) {
  double s = 0;
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      s += a[i] * b[j];
  return s;
}
// CHECK: void outerIndex_grad_0_1(const double *a, const double *b, int n, double *_d_a, double *_d_b) {
// CHECK: double _acc0 = 0.;
// CHECK-NEXT: for (j = n > 0 ? n : 0 , _t1 = n > 0 ? (unsigned {{int|long|long long}})n : 0{{U|UL|ULL}}; _t1; _t1--) {
// CHECK: _acc0 += _r_d0 * b[j];
// CHECK: _d_a[i] += _acc0;

// A constant index is invariant in any loop.
double constantIndex(const double* a, int n) {
  double s = 0;
  for (int j = 0; j < n; j++)
    s += a[0] * a[j];
  return s;
}
// CHECK: void constantIndex_grad_0(const double *a, int n, double *_d_a) {
// CHECK-NOT: _acc
// CHECK: for (_t0 = n > 0 ? (unsigned {{int|long|long long}})n : 0{{U|UL|ULL}}; _t0; _t0--) {

// The array is declared in the enclosing loop, so it lives across the inner
// one: the sum is carried across the inner loop only, and reaches _d_t before
// the declaration's own adjoint hands it on and resets it.
double declaredOutside(const double* a, int n) {
  double s = 0;
  for (int i = 0; i < n; i++) {
    double t[1] = {a[i]};
    for (int j = 0; j < n; j++)
      s += t[0] * a[j];
  }
  return s;
}
// CHECK: void declaredOutside_grad_0(const double *a, int n, double *_d_a) {
// CHECK: double _acc0 = 0.;
// CHECK: _acc0 += _r_d0 * a[j];
// CHECK: _d_t[0] += _acc0;
// CHECK: _d_a[i] += _d_t[0];
// CHECK-NEXT: clad::zero_init(_d_t);

// -- Kept in place ---------------------------------------------------------
// (constantIndex above is one too: a[0] and a[j] are two different subscripts
// of a, so nothing about a[0] can be moved.)

// The array is declared in the body of the very loop the sum would cross: a
// fresh object each iteration, whose adjoint the reverse sweep resets each
// time round.
double declaredInside(const double* a, int n) {
  double s = 0;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      double t[1] = {a[i]};
      s += t[0] * a[j];
    }
  }
  return s;
}
// CHECK: void declaredInside_grad_0(const double *a, int n, double *_d_a) {
// CHECK-NOT: _acc

// The array is handed to a call in the body; the call's pullback may read
// or reset its adjoint.
double scale(const double* p, int k) { return p[k] * 2; }
double passedToCall(const double* a, int n) {
  double s = 0;
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      s += a[i] * scale(a, j);
  return s;
}
// CHECK: void passedToCall_grad_0(const double *a, int n, double *_d_a) {
// CHECK-NOT: _acc

// A pointer copied from the array before the loop aliases it inside: the
// writes through p reset _d_q's elements, and the sum for _d_q[i] must not
// cross those.
double aliasedOutside(double* q, int n) {
  double s = 0;
  double* p = q;
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++) {
      s += q[i] * q[j];
      p[j] = p[j] * 0.5;
    }
  return s;
}
// CHECK: void aliasedOutside_grad_0(double *q, int n, double *_d_q) {
// CHECK-NOT: _acc

// The index reads a variable the body writes, so it names a different
// element on different iterations.
double movingIndex(const double* a, int n) {
  double s = 0;
  int k = 0;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      s += a[k] * a[j];
      k = (k + 1) % n;
    }
  }
  return s;
}
// CHECK: void movingIndex_grad_0(const double *a, int n, double *_d_a) {
// CHECK-NOT: _acc

#define CHECK_GRAD(NAME, ARG, ...)                                                  \
  do {                                                                         \
    auto g = clad::gradient(NAME, ARG);                                        \
    double da[4] = {0, 0, 0, 0};                                               \
    g.execute(__VA_ARGS__, da);                                                \
    bool ok = true;                                                            \
    for (int k = 0; k < 4; k++) {                                              \
      double ap[4], am[4];                                                     \
      for (int q = 0; q < 4; q++) {                                            \
        ap[q] = a[q];                                                          \
        am[q] = a[q];                                                          \
      }                                                                        \
      ap[k] += h;                                                              \
      am[k] -= h;                                                              \
      double fd = (NAME##_fd(ap) - NAME##_fd(am)) / (2 * h);                   \
      ok = ok && std::abs(da[k] - fd) <= 1e-5 * std::max(1.0, std::abs(fd));   \
    }                                                                          \
    printf("%s: %s\n", #NAME, ok ? "ok" : "MISMATCH");                         \
  } while (0)

// Each primal as a function of `a` alone, for the finite-difference check.
static const double B[4] = {0.5, -1.5, 2.0, 1.0};
double outerIndex_fd(const double* a) { return outerIndex(a, B, 4); }
double constantIndex_fd(const double* a) { return constantIndex(a, 4); }
double declaredOutside_fd(const double* a) { return declaredOutside(a, 4); }
double declaredInside_fd(const double* a) { return declaredInside(a, 4); }
double passedToCall_fd(const double* a) { return passedToCall(a, 4); }
double movingIndex_fd(const double* a) { return movingIndex(a, 4); }
double aliasedOutside_fd(const double* a) {
  double c[4] = {a[0], a[1], a[2], a[3]};
  return aliasedOutside(c, 4);
}

int main() {
  const double h = 1e-5;
  double a[4] = {0.5, 1.5, 2.5, 3.5};
  clad::gradient(outerIndex, "a,b");
  CHECK_GRAD(constantIndex, "a", a, 4);
  CHECK_GRAD(declaredOutside, "a", a, 4);
  CHECK_GRAD(declaredInside, "a", a, 4);
  CHECK_GRAD(passedToCall, "a", a, 4);
  {
    double c[4] = {a[0], a[1], a[2], a[3]};
    CHECK_GRAD(aliasedOutside, "q", c, 4);
  }
  CHECK_GRAD(movingIndex, "a", a, 4);
  // CHECK-EXEC: constantIndex: ok
  // CHECK-EXEC: declaredOutside: ok
  // CHECK-EXEC: declaredInside: ok
  // CHECK-EXEC: passedToCall: ok
  // CHECK-EXEC: aliasedOutside: ok
  // CHECK-EXEC: movingIndex: ok
  {
    auto g = clad::gradient(outerIndex, "a,b");
    double da[4] = {0, 0, 0, 0}, db[4] = {0, 0, 0, 0};
    g.execute(a, B, 4, da, db);
    // d/da[i] = sum_j b[j] = 2.0 for every i; d/db[j] = sum_i a[i] = 8.0.
    printf("outerIndex: %.2f %.2f %.2f %.2f\n", da[0], da[3], db[0], db[3]);
    // CHECK-EXEC: outerIndex: 2.00 2.00 8.00 8.00
  }
  return 0;
}
