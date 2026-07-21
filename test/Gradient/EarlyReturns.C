// RUN: %cladclang %s -I%S/../../include -oEarlyReturns.out 2>&1 | %filecheck %s
// RUN: ./EarlyReturns.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr %s -I%S/../../include -oEarlyReturns.out
// RUN: ./EarlyReturns.out | %filecheck_exec %s
//
// The synthesized reverse lambda makes Clang's CodeGen emit a closure
// constructor/destructor, whose exception landing-pad path
// (EHScopeStack::requiresLandingPad) reads an uninitialized value inside
// libclang on some runtimes. The read is Clang-internal and does not affect
// the generated derivative; skip the run under Valgrind like the other tests.
// XFAIL: valgrind

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
}
