// RUN: %cladclang %s -I%S/../../include -oEarlyReturns.out 2>&1 | %filecheck %s
// RUN: ./EarlyReturns.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr %s -I%S/../../include -oEarlyReturns.out 2>&1 | %filecheck --check-prefix=CHECK-NOTBR %s
// RUN: ./EarlyReturns.out | %filecheck_exec %s
//
// The gradients must stay Memcheck-clean on every return path: the valgrind CI
// row runs both executables, and the early path reads hoisted locals the
// forward sweep never assigned, so their declarations may not leave
// indeterminate bits behind.

#include "clad/Differentiator/Differentiator.h"
#include "../TestUtils.h"

// A body whose only return is in tail position keeps the plain shape: control
// falls through to the reverse sweep, so no lambda is materialised.
double noEarly(double x, double y) { return x * y; }

// CHECK-LABEL: void noEarly_grad(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NOT: _rev

// A non-tail return cannot fall through, so the reverse sweep becomes a named
// lambda that both the early-return site and the tail return call.
double singleEarly(double x, double y) {
  if (x > y)
    return x * x;
  return x * y;
}

// CHECK-LABEL: void singleEarly_grad(double x, double y, double *_d_x, double *_d_y) {
// CHECK: auto _rev0 = [&

// Several early returns share one lambda; each site gets its own call
// statement rather than reusing a single node.
double multiEarly(double x, double y) {
  if (x > 10)
    return x;
  if (y > 10)
    return y * y;
  return x * y;
}

// CHECK-LABEL: void multiEarly_grad(double x, double y, double *_d_x, double *_d_y) {
// CHECK: auto _rev0 = [&

// An early return inside a loop leaves mid-iteration: the reverse sweep runs
// from the exit point back through the taped iterations, so the lambda must
// bind the loop's stored state.
double loopEarly(double x, double y) {
  double s = 0;
  for (int i = 0; i < 5; ++i) {
    s += x * y;
    if (s > 20)
      return s;
  }
  return s;
}

// CHECK-LABEL: void loopEarly_grad(double x, double y, double *_d_x, double *_d_y) {
// CHECK: auto _rev0 = [&

// The early return fires after some primal state (a, b) is already built, so
// the lambda captures a partially-computed forward sweep.
double midEarly(double x, double y) {
  double a = x * y;
  double b = a + x;
  if (b > 50)
    return a * b;
  return b * y;
}

// CHECK-LABEL: void midEarly_grad(double x, double y, double *_d_x, double *_d_y) {
// CHECK: auto _rev0 = [&

// A recursive callee is requested twice: once for the self-call and once for
// the enclosing call. The two requests dedup, and the flag recording the early
// return must survive that dedup or the pullback would lose the lambda shape.
double recEarly(double x, double y) {
  if (x > y)
    return recEarly(x - 1, y);
  return x * y;
}

// CHECK-LABEL: void recEarly_pullback(double x, double y, double _d_y0, double *_d_x, double *_d_y) {
// CHECK: auto _rev0 = [&

double callsRec(double x, double y) { return recEarly(x, y); }

double nested_early_return(double x, double y) {
  if (x > 0) {
    x = x * y; 
  }
  if (x < 0) {
    return 0;  
  }
  return x;
}

// CHECK-LABEL: void nested_early_return_grad(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:     bool _cond0 = false;
// CHECK-NEXT:     double _t0 = 0.;
// CHECK-NEXT:     bool _cond1 = false;


// The early return is the very first statement, so the forward-sweep split
// at the lambda's insertion point starts empty: n, a, b, and a's
// pre-multiplication snapshot are all declared after it, yet the reverse
// sweep (inside the lambda) reads them for the product rule on `a *= b`.
// Those decls must still end up visible -- and correctly valued -- before
// the lambda, not left behind it where CodeGen would mistake a captured-but-
// not-yet-declared local for a block-scope static (vgvassilev/clad#1940).
// `b, c` is one DeclStmt with mixed needs -- b is captured and must move, c
// is not captured at all -- so the hoister has to take that statement apart
// and leave c where it was.
double declAfterEarly(double x, double y) {
  if (y == 0)
    return 1;
  double n = y;
  double a = n + x;
  double b = n, c = 2 * x;
  a *= b;
  return a + c;
}

// CHECK-LABEL: void declAfterEarly_grad(double x, double y, double *_d_x, double *_d_y) {
// CHECK: auto _rev0 = [&
// CHECK-NOT: this is a clad bug

// `r` and `c` are read by the reverse sweep, so the hoister splits them:
// declaration before the lambda, initializer left behind as an assignment.
// Neither survives that as spelled -- a reference cannot be declared unbound
// and re-seated, a const cannot be assigned -- so the clone of `r` is a
// pointer (as for a reference promoted to the function global scope) and `c`
// loses its const (vgvassilev/clad#1954).
double refAfterEarly(double x, double y) {
  if (y == 0)
    return 1;
  double a = x * y;
  double& r = a;
  const double c = r * r;
  double d = c * c;
  return d;
}

// CHECK-LABEL: void refAfterEarly_grad(double x, double y, double *_d_x, double *_d_y) {
// CHECK: double *_d_r = &_d_a;
// CHECK-NEXT: double _dummy0 = 0.;
// CHECK-NEXT: double *r = &_dummy0;
// CHECK-NEXT: double _d_c = 0.;
// CHECK-NEXT: double c = 0.;
// CHECK: auto _rev0 = [&
// CHECK: r = &a;
// CHECK-NEXT: c = *r * *r;
// CHECK-NOT: this is a clad bug

// The same split with a record referent: the hoisted pointer clone must still
// point somewhere on the early path, so the placeholder is a fresh
// zero-initialized dummy -- possible because zero init of Pair, whose default
// constructor is trivial, cannot run into a user-provided one.
struct Pair {
  double a;
  double b;
};

double structRefAfterEarly(double x, double y) {
  if (y == 0)
    return 1;
  Pair p;
  p.a = x * y;
  p.b = x + y;
  Pair& r = p;
  double d = r.a * r.b;
  return d;
}

// CHECK-LABEL: void structRefAfterEarly_grad(double x, double y, double *_d_x, double *_d_y) {
// CHECK: Pair _dummy0 = {0., 0.};
// CHECK-NEXT: Pair *r = &_dummy0;
// CHECK: auto _rev0 = [&
// CHECK: r = &p;
// CHECK-NOT: this is a clad bug

// Without TBR the sweep also restores p's members from snapshots, so `p`
// itself is captured and hoisted -- moved wholesale, its default init would
// leave indeterminate members for the early path's sweep (and for the
// snapshots, taken right after the hoisted declaration) to read. The hoist
// must add a zero initializer.
// CHECK-NOTBR-LABEL: void structRefAfterEarly_grad(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NOTBR: Pair p = {0., 0.};
// CHECK-NOTBR: double _t0 = p.a;

// A record whose member initializers make the default constructor non-trivial
// is not one clad will conjure: the placeholder stays null, and only the
// fall-through path is defined.
struct Seeded {
  double a = 1.;
  double b = 0.;
};

double nontrivialRefAfterEarly(double x, double y) {
  if (y == 0)
    return 1;
  Seeded s;
  s.a = x * y;
  s.b = x + y;
  Seeded& r = s;
  double d = r.a * r.b;
  return d;
}

// CHECK-LABEL: void nontrivialRefAfterEarly_grad(double x, double y, double *_d_x, double *_d_y) {
// CHECK: Seeded *r = nullptr;
// CHECK: auto _rev0 = [&
// CHECK: r = &s;
// CHECK-NOT: this is a clad bug

int main() {
  double dx = 0, dy = 0;

  INIT_GRADIENT(noEarly);
  TEST_GRADIENT(noEarly, /*numOfDerivativeArgs=*/2, 3, 5, &dx, &dy); // CHECK-EXEC: {5.00, 3.00}

  dx = dy = 0;
  INIT_GRADIENT(singleEarly);
  // x > y takes the early return: d(x*x) = {2x, 0}.
  TEST_GRADIENT(singleEarly, /*numOfDerivativeArgs=*/2, 5, 3, &dx, &dy); // CHECK-EXEC: {10.00, 0.00}
  dx = dy = 0;
  // x <= y falls through to the tail return: d(x*y) = {y, x}.
  TEST_GRADIENT(singleEarly, /*numOfDerivativeArgs=*/2, 3, 5, &dx, &dy); // CHECK-EXEC: {5.00, 3.00}

  dx = dy = 0;
  INIT_GRADIENT(multiEarly);
  TEST_GRADIENT(multiEarly, /*numOfDerivativeArgs=*/2, 20, 1, &dx, &dy); // CHECK-EXEC: {1.00, 0.00}
  dx = dy = 0;
  TEST_GRADIENT(multiEarly, /*numOfDerivativeArgs=*/2, 1, 20, &dx, &dy); // CHECK-EXEC: {0.00, 40.00}
  dx = dy = 0;
  TEST_GRADIENT(multiEarly, /*numOfDerivativeArgs=*/2, 2, 3, &dx, &dy); // CHECK-EXEC: {3.00, 2.00}

  dx = dy = 0;
  INIT_GRADIENT(loopEarly);
  // s exceeds 20 on the 4th iteration: returns 4*x*y, so d = {4y, 4x}.
  TEST_GRADIENT(loopEarly, /*numOfDerivativeArgs=*/2, 2, 3, &dx, &dy); // CHECK-EXEC: {12.00, 8.00}
  dx = dy = 0;
  // s never exceeds 20: falls through to the tail, 5*x*y, so d = {5y, 5x}.
  TEST_GRADIENT(loopEarly, /*numOfDerivativeArgs=*/2, 1, 1, &dx, &dy); // CHECK-EXEC: {5.00, 5.00}

  dx = dy = 0;
  INIT_GRADIENT(midEarly);
  // b > 50 takes the early return a*b = x*x*y*y + x*x*y.
  TEST_GRADIENT(midEarly, /*numOfDerivativeArgs=*/2, 8, 7, &dx, &dy); // CHECK-EXEC: {896.00, 960.00}
  dx = dy = 0;
  // b <= 50 falls through to the tail b*y = x*y*y + x*y.
  TEST_GRADIENT(midEarly, /*numOfDerivativeArgs=*/2, 6, 7, &dx, &dy); // CHECK-EXEC: {56.00, 90.00}

  dx = dy = 0;
  INIT_GRADIENT(callsRec);
  TEST_GRADIENT(callsRec, /*numOfDerivativeArgs=*/2, 3, 1, &dx, &dy); // CHECK-EXEC: {1.00, 1.00}

  dx = dy = 0;
  INIT_GRADIENT(nested_early_return);
  // x > 0 triggers x = x * y, falling through to return x. 
  TEST_GRADIENT(nested_early_return, /*numOfDerivativeArgs=*/2, 5, 3, &dx, &dy); // CHECK-EXEC: {3.00, 5.00}
  dx = dy = 0;
  // x < 0 takes the early return 0. 
  TEST_GRADIENT(nested_early_return, /*numOfDerivativeArgs=*/2, -5, 3, &dx, &dy); // CHECK-EXEC: {0.00, 0.00}

  dx = dy = 0;
  INIT_GRADIENT(declAfterEarly);
  // y == 0 takes the early return: d(1) = {0, 0}.
  TEST_GRADIENT(declAfterEarly, /*numOfDerivativeArgs=*/2, 3, 0, &dx, &dy); // CHECK-EXEC: {0.00, 0.00}
  dx = dy = 0;
  // y != 0 falls through: (y + x) * y + 2x, so d = {y + 2, x + 2y}.
  TEST_GRADIENT(declAfterEarly, /*numOfDerivativeArgs=*/2, 3, 5, &dx, &dy); // CHECK-EXEC: {7.00, 13.00}

  dx = dy = 0;
  INIT_GRADIENT(refAfterEarly);
  // y != 0 falls through: (x * y)^4, so d = {4(xy)^3 y, 4(xy)^3 x}.
  TEST_GRADIENT(refAfterEarly, /*numOfDerivativeArgs=*/2, 1, 2, &dx, &dy); // CHECK-EXEC: {64.00, 32.00}
  dx = dy = 0;
  // y == 0 takes the early return: d(1) = {0, 0}. The sweep still dereferences
  // `r`, which this path never bound, so its placeholder must point somewhere.
  TEST_GRADIENT(refAfterEarly, /*numOfDerivativeArgs=*/2, 1, 0, &dx, &dy); // CHECK-EXEC: {0.00, 0.00}

  dx = dy = 0;
  INIT_GRADIENT(structRefAfterEarly);
  // y != 0 falls through: (x * y) * (x + y), so d = {y * (2x + y), x * (x + 2y)}.
  TEST_GRADIENT(structRefAfterEarly, /*numOfDerivativeArgs=*/2, 1, 2, &dx, &dy); // CHECK-EXEC: {8.00, 5.00}
  dx = dy = 0;
  // y == 0 takes the early return; the sweep dereferences the never-bound `r`,
  // which is defined only because the placeholder points at the Pair dummy.
  TEST_GRADIENT(structRefAfterEarly, /*numOfDerivativeArgs=*/2, 1, 0, &dx, &dy); // CHECK-EXEC: {0.00, 0.00}

  dx = dy = 0;
  INIT_GRADIENT(nontrivialRefAfterEarly);
  // Fall-through path only: with the null placeholder, the early path's sweep
  // would dereference the never-bound `r`.
  TEST_GRADIENT(nontrivialRefAfterEarly, /*numOfDerivativeArgs=*/2, 1, 2, &dx, &dy); // CHECK-EXEC: {8.00, 5.00}
}
