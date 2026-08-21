// RUN: %cladclang %s -I%S/../../include -oCountedReverseLoops.out 2>&1 \
// RUN:   | %filecheck %s
// RUN: ./CountedReverseLoops.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr %s \
// RUN:   -I%S/../../include -oCountedReverseLoops.out
// RUN: ./CountedReverseLoops.out | %filecheck_exec %s

// A loop whose iteration count the reverse sweep can recompute from the loop's
// own bounds does not need the forward sweep to count for it. The useful half
// of what follows is the negative half: every loop clad must keep counting,
// because recomputing there would silently produce a wrong gradient rather
// than a slow one.

#include "clad/Differentiator/Differentiator.h"

#include <cmath>
#include <cstdio>

// -- Recomputed ------------------------------------------------------------

// A literal bound is known here, so the count is spelled out.
double literalBound(const double* x) {
  double s = 0;
  for (int i = 0; i < 4; i++)
    s += x[i] * x[i];
  return s;
}
// CHECK: void literalBound_grad(const double *x, double *_d_x) {
// CHECK-NOT: _t0++;
// CHECK: for (_t0 = 4{{U|UL|ULL}}; _t0; _t0--) {

// A constexpr variable, an enumerator or a template argument is as constant as
// a literal, and is the usual way a dimension is written.
constexpr int Dim = 3;
double constexprBound(const double* x) {
  double s = 0;
  for (int i = 0; i < Dim; i++)
    s += x[i] * x[i];
  return s;
}
// CHECK: void constexprBound_grad(const double *x, double *_d_x) {
// CHECK: for (_t0 = 3{{U|UL|ULL}}; _t0; _t0--) {

// A parameter the primal never writes still reads the same in the reverse
// sweep. The guard is the loop's own comparison, so a bound of zero or less
// gives a count of zero rather than an enormous one.
double paramBound(const double* x, int n) {
  double s = 0;
  for (int i = 0; i < n; i++)
    s += x[i] * x[i];
  return s;
}
// CHECK: void paramBound_grad_0(const double *x, int n, double *_d_x) {
// CHECK: for (_t0 = n > 0 ? (unsigned {{int|long|long long}})n : 0{{U|UL|ULL}}; _t0; _t0--) {

// An inclusive bound runs once more, and a non-zero start that many fewer.
double inclusiveBound(const double* x, int n) {
  double s = 0;
  for (int i = 1; i <= n; i++)
    s += x[i] * x[i];
  return s;
}
// CHECK: void inclusiveBound_grad_0(const double *x, int n, double *_d_x) {
// CHECK: for (_t0 = n >= 1 ? (unsigned {{int|long|long long}})n : 0{{U|UL|ULL}}; _t0; _t0--) {

// Nested loops are where counting cost the most: the inner counter used to be
// a whole clad::tape, pushed once per outer iteration. Recomputed, it is a
// plain variable the inner reverse loop assigns on entry.
double nested(const double* x) {
  double s = 0;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 2; ++j)
      s += x[i] * x[j];
  return s;
}
// CHECK: void nested_grad(const double *x, double *_d_x) {
// CHECK-NOT: clad::tape<unsigned {{int|long|long long}}>
// CHECK: for (_t0 = 3{{U|UL|ULL}}; _t0; _t0--) {
// CHECK-NEXT: i--;
// CHECK-NEXT: for (_t1 = 2{{U|UL|ULL}}; _t1; _t1--) {

// A second index walked alongside the induction variable joins the increment
// with a comma. That still steps the induction variable by one.
double commaStep(const double* x) {
  double s = 0;
  int p = 0;
  for (int i = 0; i < 4; i++, p++)
    s += x[i] * p;
  return s;
}
// CHECK: void commaStep_grad(const double *x, double *_d_x) {
// CHECK: for (_t0 = 4{{U|UL|ULL}}; _t0; _t0--) {

// -- Counted ---------------------------------------------------------------

// The bound is written in the loop, so what it reads in the reverse sweep is
// not what the forward sweep saw.
double variableBound(const double* x, int n) {
  double s = 0;
  for (int i = 0; i < n; i++) {
    s += x[i] * x[i];
    n = n - 1;
  }
  return s;
}
// CHECK: void variableBound_grad_0(const double *x, int n, double *_d_x) {
// CHECK: _t0++;
// CHECK: for (; _t0; _t0--) {

// A `break` stops the loop before its bound says so.
double earlyBreak(const double* x, int n) {
  double s = 0;
  for (int i = 0; i < n; i++) {
    if (x[i] < 0)
      break;
    s += x[i] * x[i];
  }
  return s;
}
// CHECK: void earlyBreak_grad_0(const double *x, int n, double *_d_x) {
// CHECK: _t0++;

// The body moves the induction variable itself.
double skips(const double* x, int n) {
  double s = 0;
  for (int i = 0; i < n; i++) {
    s += x[i] * x[i];
    if (x[i] > 4)
      i = i + 1;
  }
  return s;
}
// CHECK: void skips_grad_0(const double *x, int n, double *_d_x) {
// CHECK: _t0++;
// CHECK: for (; _t0; _t0--) {

// A callee moves it, through a non-const reference. Nothing at the loop says
// so; only the parameter's type does.
void advance(int& i) { i += 1; }
double skipsViaCall(const double* x, int n) {
  double s = 0;
  for (int i = 0; i < n; i++) {
    s += x[i] * x[i];
    advance(i);
  }
  return s;
}
// CHECK: void skipsViaCall_grad_0(const double *x, int n, double *_d_x) {
// CHECK: _t0++;
// CHECK: for (; _t0; _t0--) {

// A bound whose address escapes may be written through the pointer.
double escapedBound(const double* x, int n) {
  int* p = &n;
  double s = 0;
  for (int i = 0; i < n; i++) {
    s += x[i] * x[i];
    *p = *p - 1;
  }
  return s;
}
// CHECK: void escapedBound_grad_0(const double *x, int n, double *_d_x) {
// CHECK: _t0++;
// CHECK: for (; _t0; _t0--) {

// The step is not one.
double stride2(const double* x, int n) {
  double s = 0;
  for (int i = 0; i < n; i += 2)
    s += x[i] * x[i];
  return s;
}
// CHECK: void stride2_grad_0(const double *x, int n, double *_d_x) {
// CHECK: _t0++;
// CHECK: for (; _t0; _t0--) {

// Only `for` is recognised; an equivalent while loop is not.
double whileLoop(const double* x, int n) {
  double s = 0;
  int i = 0;
  while (i < n) {
    s += x[i] * x[i];
    i++;
  }
  return s;
}
// CHECK: void whileLoop_grad_0(const double *x, int n, double *_d_x) {
// CHECK: _t0++;

// An early return elsewhere in the function can skip the forward loop while
// the master reverse sweep still runs. A count recomputed there would be the
// full one for a loop that never executed.
double earlyReturn(const double* x, int n) {
  if (n < 0)
    return 0;
  double s = 0;
  for (int i = 0; i < 4; i++)
    s += x[i] * x[i];
  return s;
}
// CHECK: void earlyReturn_grad_0(const double *x, int n, double *_d_x) {
// CHECK: _t0++;

// -- Values ----------------------------------------------------------------

#define CHECK_GRAD(NAME, N)                                                    \
  do {                                                                         \
    auto g = clad::gradient(NAME, "x");                                        \
    double dx[8] = {0, 0, 0, 0, 0, 0, 0, 0};                                   \
    g.execute(x, N, dx);                                                       \
    bool ok = true;                                                            \
    for (int k = 0; k < 8; k++) {                                              \
      double xp[8], xm[8];                                                     \
      for (int j = 0; j < 8; j++) {                                            \
        xp[j] = x[j];                                                          \
        xm[j] = x[j];                                                          \
      }                                                                        \
      xp[k] += h;                                                              \
      xm[k] -= h;                                                              \
      double fd = (NAME(xp, N) - NAME(xm, N)) / (2 * h);                       \
      ok = ok && std::abs(dx[k] - fd) <= 1e-5 * std::max(1.0, std::abs(fd));   \
    }                                                                          \
    printf("%s(%d): %s\n", #NAME, (int)(N), ok ? "ok" : "MISMATCH");           \
  } while (0)

int main() {
  const double h = 1e-5;
  double x[8] = {0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5};

  // Derivatives are dumped in the order they are first requested, so ask for
  // them in the order the primals appear above and the checks above read
  // alongside the code they check.
  auto gl = clad::gradient(literalBound, "x");
  auto gc = clad::gradient(constexprBound, "x");
  clad::gradient(paramBound, "x");
  clad::gradient(inclusiveBound, "x");
  auto gn = clad::gradient(nested, "x");
  auto gs = clad::gradient(commaStep, "x");
  clad::gradient(variableBound, "x");
  clad::gradient(earlyBreak, "x");
  clad::gradient(skips, "x");
  clad::gradient(skipsViaCall, "x");
  clad::gradient(escapedBound, "x");
  clad::gradient(stride2, "x");
  clad::gradient(whileLoop, "x");
  clad::gradient(earlyReturn, "x");

  // The recomputed loops, including the counts a wrong recomputation would
  // get wrong: an empty loop, and one whose bound is negative.
  CHECK_GRAD(paramBound, 4);
  CHECK_GRAD(paramBound, 0);
  CHECK_GRAD(paramBound, -1);
  CHECK_GRAD(inclusiveBound, 3);
  CHECK_GRAD(inclusiveBound, 0);
  // CHECK-EXEC: paramBound(4): ok
  // CHECK-EXEC: paramBound(0): ok
  // CHECK-EXEC: paramBound(-1): ok
  // CHECK-EXEC: inclusiveBound(3): ok
  // CHECK-EXEC: inclusiveBound(0): ok

  // The counted ones, where a recomputed count would disagree with the number
  // of iterations the forward sweep actually ran.
  CHECK_GRAD(variableBound, 6);
  CHECK_GRAD(skips, 6);
  // skipsViaCall is checked above for the code clad emits, but not for its
  // value: advance_pullback replays `i += 1` without restoring i, so the
  // reverse sweep indexes one past where the forward sweep was. That is a
  // separate defect in the pullback of an int taken by non-const reference,
  // unrelated to how the loop is counted.
  CHECK_GRAD(escapedBound, 6);
  CHECK_GRAD(stride2, 7);
  CHECK_GRAD(whileLoop, 5);
  CHECK_GRAD(earlyReturn, 2);
  // CHECK-EXEC: variableBound(6): ok
  // CHECK-EXEC: skips(6): ok
  // CHECK-EXEC: escapedBound(6): ok
  // CHECK-EXEC: stride2(7): ok
  // CHECK-EXEC: whileLoop(5): ok
  // CHECK-EXEC: earlyReturn(2): ok

  // The bound-free ones take no count argument.
  double dl[8] = {0}, dc[8] = {0}, dn[8] = {0}, ds[8] = {0};
  gl.execute(x, dl);
  gc.execute(x, dc);
  gn.execute(x, dn);
  gs.execute(x, ds);
  printf("literalBound: %.2f %.2f\n", dl[0], dl[3]);
  // CHECK-EXEC: literalBound: 1.00 7.00
  printf("constexprBound: %.2f %.2f\n", dc[0], dc[2]);
  // CHECK-EXEC: constexprBound: 1.00 5.00
  // nested() is (x0+x1+x2)*(x0+x1), so d/dx0 = (x0+x1) + (x0+x1+x2) = 6.5 and
  // d/dx2 = (x0+x1) = 2.
  printf("nested: %.2f %.2f\n", dn[0], dn[2]);
  // CHECK-EXEC: nested: 6.50 2.00
  printf("commaStep: %.2f %.2f\n", ds[0], ds[3]);
  // CHECK-EXEC: commaStep: 0.00 3.00
  return 0;
}
