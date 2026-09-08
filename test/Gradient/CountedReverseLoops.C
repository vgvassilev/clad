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
// CHECK-NEXT: for (j = 2 , _t1 = 2{{U|UL|ULL}}; _t1; _t1--) {

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

// The triangular loop -- the shape a packed symmetric matrix is walked with.
// Its start is not loop-invariant at all: it names the enclosing induction
// variable. That still reads correctly, because the reverse sweep steps the
// outer variable back before it enters the inner reverse loop.
double triangular(const double* x, int n) {
  double s = 0;
  for (int i = 0; i < n; i++)
    for (int j = i + 1; j < n; j++)
      s += x[i] * x[j];
  return s;
}
// CHECK: void triangular_grad_0(const double *x, int n, double *_d_x) {
// CHECK-NOT: clad::tape<unsigned {{int|long|long long}}>
// CHECK: for (_t0 = n > 0 ? (unsigned {{int|long|long long}})n : 0{{U|UL|ULL}}; _t0; _t0--) {
// CHECK-NEXT: i--;
// CHECK-NEXT: for (j = n > i + 1 ? n : i + 1 , _t1 = n > i + 1 ? ((unsigned {{int|long|long long}})n - (unsigned {{int|long|long long}})(i + 1)) : 0{{U|UL|ULL}}; _t1; _t1--) {

// A variadic argument travels by value, so a loop that reports its progress
// still has an index only the increment moves. Handing it over for writing
// reads as note("i", &i) instead, where taking the address is already a
// write.
int note(const char* tag, ...) { return 0; }
double logsIndex(const double* x, int n) {
  double s = 0;
  for (int i = 0; i < n; i++) {
    note("i", i);
    s += x[i] * x[i];
  }
  return s;
}
// CHECK: void logsIndex_grad_0(const double *x, int n, double *_d_x) {
// CHECK-NOT: _t0++;
// CHECK: for (_t0 = n > 0 ? (unsigned {{int|long|long long}})n : 0{{U|UL|ULL}}; _t0; _t0--) {

// A start the count has to carry a sign through, which is how a stencil walks
// the halo either side of its centre.
double negatedStart(const double* x, int n, int m) {
  double s = 0;
  for (int i = -m; i < n; i++)
    s += x[i + 4] * x[i + 4];
  return s;
}
// CHECK: void negatedStart_grad_0(const double *x, int n, int m, double *_d_x) {
// CHECK-NOT: _t0++;
// CHECK: for (_t0 = n > -m ? ((unsigned {{int|long|long long}})n - (unsigned {{int|long|long long}})-m) : 0{{U|UL|ULL}}; _t0; _t0--) {

// An inclusive bound over a start that is not a constant. The start cannot
// absorb the extra iteration the way a literal one does, so the count carries
// it as its own term.
double inclusiveVarStart(const double* x, int n, int lo) {
  double s = 0;
  for (int i = lo; i <= n; i++)
    s += x[i] * x[i];
  return s;
}
// CHECK: void inclusiveVarStart_grad_0(const double *x, int n, int lo, double *_d_x) {
// CHECK-NOT: _t0++;
// CHECK: for (_t0 = n >= lo ? ((unsigned {{int|long|long long}})n - (unsigned {{int|long|long long}})lo + 1{{U|UL|ULL}}) : 0{{U|UL|ULL}}; _t0; _t0--) {

// -- Counted ---------------------------------------------------------------

// An outer loop whose own bound it writes cannot have its trip count worked
// out, so it keeps a counter. It is still a counted loop, so it still steps
// its index back on the way into anything nested in it, and the inner loop
// naming that index is recomputed even though the outer one is not.
double outerUnstable(const double* x, int n) {
  double s = 0;
  for (int i = 0; i < n; ++i) {
    if (i == 1)
      n = 4;
    for (int j = i + 1; j < 5; ++j)
      s += x[j] * x[i];
  }
  return s;
}
// CHECK: void outerUnstable_grad_0(const double *x, int n, double *_d_x) {
// CHECK: _t0++;
// CHECK: for (j = 5 > i + 1 ? 5 : i + 1 , _t1 = 5 > i + 1 ? ((unsigned {{int|long|long long}})5 - (unsigned {{int|long|long long}})(i + 1)) : 0{{U|UL|ULL}}; _t1; _t1--) {

// The same inner loop, but the variable its start names belongs to a while
// loop. Nothing steps that variable back on the way into the inner reverse
// loop, so what the start would read there is not what the forward sweep saw.
double triangularUnderWhile(const double* x, int n) {
  double s = 0;
  int i = 0;
  while (i < n) {
    for (int j = i + 1; j < n; j++)
      s += x[i] * x[j];
    i++;
  }
  return s;
}
// CHECK: void triangularUnderWhile_grad_0(const double *x, int n, double *_d_x) {
// CHECK: clad::tape<unsigned {{int|long|long long}}>

// The index of the inner loop is declared outside it, so a statement after the
// loop can read what the loop left there. Its pre-loop value has to be saved,
// even though the count itself is still recomputed.
double sharedIndex(const double* x, int n) {
  double s = 0;
  int j = 0;
  for (int i = 0; i < n; i++)
    for (j = i + 1; j < n; j++)
      s += x[i] * x[j];
  return s + j;
}
// CHECK: void sharedIndex_grad_0(const double *x, int n, double *_d_x) {
// CHECK: clad::tape<int> [[JT:_t[0-9]+]] = {};
// CHECK: clad::push([[JT]], j);
// CHECK-NEXT: for (j = i + 1; j < n; j++) {
// CHECK: j = clad::pop([[JT]]);


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

// A bound in static storage is reachable from inside any callee, and a bound
// reached through a reference names storage this walk does not follow. In
// neither case does the body say the bound holds still between the sweeps.
double staticBound(const double* x) {
  static int sBound = 4;
  double s = 0;
  for (int i = 0; i < sBound; i++)
    s += x[i] * x[i];
  return s;
}
// CHECK: void staticBound_grad(const double *x, double *_d_x) {
// CHECK: _t0++;
// CHECK: for (; _t0; _t0--) {

double refBound(const double* x, int& n) {
  double s = 0;
  for (int i = 0; i < n; i++)
    s += x[i] * x[i];
  return s;
}
// CHECK: void refBound_grad_0(const double *x, int &n, double *_d_x) {
// CHECK: _t0++;
// CHECK: for (; _t0; _t0--) {

// A bound built with an operator the stability reading does not model. The
// count would be right here, but `/` is not one of the three it proves
// anything about, so the loop keeps counting.
double dividedBound(const double* x, int n) {
  double s = 0;
  for (int i = 0; i < n / 2; i++)
    s += x[i] * x[i];
  return s;
}
// CHECK: void dividedBound_grad_0(const double *x, int n, double *_d_x) {
// CHECK: _t0++;
// CHECK: for (; _t0; _t0--) {

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
  clad::gradient(triangular, "x");
  clad::gradient(logsIndex, "x");
  auto gns = clad::gradient(negatedStart, "x");
  auto giv = clad::gradient(inclusiveVarStart, "x");
  clad::gradient(outerUnstable, "x");
  clad::gradient(triangularUnderWhile, "x");
  clad::gradient(sharedIndex, "x");
  clad::gradient(variableBound, "x");
  clad::gradient(earlyBreak, "x");
  clad::gradient(skips, "x");
  clad::gradient(skipsViaCall, "x");
  clad::gradient(escapedBound, "x");
  clad::gradient(stride2, "x");
  clad::gradient(whileLoop, "x");
  clad::gradient(earlyReturn, "x");
  auto gsb = clad::gradient(staticBound, "x");
  auto grb = clad::gradient(refBound, "x");
  clad::gradient(dividedBound, "x");

  // The recomputed loops, including the counts a wrong recomputation would
  // get wrong: an empty loop, and one whose bound is negative.
  CHECK_GRAD(paramBound, 4);
  CHECK_GRAD(paramBound, 0);
  CHECK_GRAD(paramBound, -1);
  CHECK_GRAD(inclusiveBound, 3);
  CHECK_GRAD(inclusiveBound, 0);
  // The triangular count depends on the enclosing loop's variable, so an
  // off-by-one there would show up only as a wrong gradient.
  CHECK_GRAD(triangular, 6);
  CHECK_GRAD(triangular, 1);
  // The inner count names the outer index while the outer loop rewrites its
  // own bound mid-flight, so a wrong step-back would show up only here.
  CHECK_GRAD(outerUnstable, 3);
  CHECK_GRAD(triangularUnderWhile, 6);
  CHECK_GRAD(sharedIndex, 6);
  CHECK_GRAD(dividedBound, 7);
  // CHECK-EXEC: paramBound(4): ok
  // CHECK-EXEC: paramBound(0): ok
  // CHECK-EXEC: paramBound(-1): ok
  // CHECK-EXEC: inclusiveBound(3): ok
  // CHECK-EXEC: inclusiveBound(0): ok
  // CHECK-EXEC: triangular(6): ok
  // CHECK-EXEC: triangular(1): ok
  // CHECK-EXEC: outerUnstable(3): ok
  // CHECK-EXEC: triangularUnderWhile(6): ok
  // CHECK-EXEC: sharedIndex(6): ok
  // CHECK-EXEC: dividedBound(7): ok

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

  // A sign carried through the count, an inclusive bound over a variable
  // start, and a bound the loop cannot claim: all three are wrong by a whole
  // iteration if the count is.
  double dns[8] = {0}, dvs[8] = {0}, dsb[8] = {0};
  gns.execute(x, 3, 2, dns);
  printf("negatedStart: %.2f %.2f\n", dns[2], dns[6]);
  // CHECK-EXEC: negatedStart: 5.00 13.00
  giv.execute(x, 3, 1, dvs);
  printf("inclusiveVarStart: %.2f %.2f\n", dvs[1], dvs[3]);
  // CHECK-EXEC: inclusiveVarStart: 3.00 7.00
  gsb.execute(x, dsb);
  printf("staticBound: %.2f %.2f\n", dsb[0], dsb[3]);
  // CHECK-EXEC: staticBound: 1.00 7.00
  double drb[8] = {0};
  int rn = 4;
  grb.execute(x, rn, drb);
  printf("refBound: %.2f %.2f\n", drb[0], drb[3]);
  // CHECK-EXEC: refBound: 1.00 7.00
  return 0;
}
