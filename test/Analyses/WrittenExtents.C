// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -fdump-analysis=loop %s \
// RUN:   -I%S/../../include -oWrittenExtents.out 2>&1 | %filecheck %s
// RUN: ./WrittenExtents.out

// What a callee writes through each pointer parameter, proven from its body
// and expressed in its own parameters so a call site can evaluate it. The
// whitelist is deliberately narrow; the useful half of the report is what
// comes back `unknown`, because that is what a caller must stay conservative
// about. Widening the whitelist may turn an `unknown` below into a range, but
// must never turn a range into a different range.

#include "clad/Differentiator/Differentiator.h"

// The canonical shape: a counted loop over a parameter bound.
void subtract(int d, const double* x, const double* y, double* out) {
  for (int i = 0; i < d; i++)
    out[i] = x[i] - y[i];
}
// CHECK: written-extent: subtract: d = none
// CHECK-NEXT: written-extent: subtract: x = none
// CHECK-NEXT: written-extent: subtract: y = none
// CHECK-NEXT: written-extent: subtract: out = [0, d)

// Two loops write `out`, the second from `i + 1` rather than zero. Both stay
// inside [0, d), so the two records agree and the parameter is still proven.
void qtimesx(int d, const double* Qd, const double* x, double* out) {
  for (int i = 0; i < d; i++)
    out[i] = Qd[i] * x[i];
  for (int i = 0; i < d; i++)
    for (int j = i + 1; j < d; j++)
      out[j] = out[j] + x[i];
}
// CHECK: written-extent: qtimesx: d = none
// CHECK-NEXT: written-extent: qtimesx: Qd = none
// CHECK-NEXT: written-extent: qtimesx: x = none
// CHECK-NEXT: written-extent: qtimesx: out = [0, d)

// Two different constant offsets do not describe one range, so they widen to
// unknown rather than to a guessed hull.
void constIdx(double* v) {
  v[0] = 1.0;
  v[2] = 3.0;
}
// CHECK: written-extent: constIdx: v = unknown (writes do not describe one range at line [[@LINE-2]])

// A constant bound is as usable as a parameter one.
void fixedLoop(double* v) {
  for (int i = 0; i < 5; ++i)
    v[i] = v[i] * 2;
}
// CHECK: written-extent: fixedLoop: v = [0, 5)

// The index is not an induction variable, so nothing bounds the write.
void dataDependent(int n, const double* x, double* out) {
  for (int i = 0; i < n; i++)
    out[(int)x[i]] = x[i];
}
// CHECK: written-extent: dataDependent: n = none
// CHECK-NEXT: written-extent: dataDependent: x = none
// CHECK-NEXT: written-extent: dataDependent: out = unknown (index is neither a constant nor a variable at line [[@LINE-4]])

// Only a loop stepping by one is recognised, and only when it is spelled with
// ++. The rest report unknown, which is the safe answer, and these cases pin
// it: a step of two writes every other element, so [0, d) would merely be
// wasteful, but a loop counting down writes none of [init, bound) at all, and
// an extent that claims a range the loop never touches is a wrong gradient.
// Widening the recogniser means deciding what each of these writes first.
void plusEqualOne(int d, double* out) {
  for (int i = 0; i < d; i += 1)
    out[i] = 1;
}
// CHECK: written-extent: plusEqualOne: out = unknown (index not stepped by a counted loop at line [[@LINE-2]])

void stepsByTwo(int d, double* out) {
  for (int i = 0; i < d; i += 2)
    out[i] = 1;
}
// CHECK: written-extent: stepsByTwo: out = unknown (index not stepped by a counted loop at line [[@LINE-2]])

void stepsByParameter(int d, int k, double* out) {
  for (int i = 0; i < d; i += k)
    out[i] = 1;
}
// CHECK: written-extent: stepsByParameter: out = unknown (index not stepped by a counted loop at line [[@LINE-2]])

void countsDown(int d, double* out) {
  for (int i = d - 1; i > 0; --i)
    out[i] = 1;
}
// CHECK: written-extent: countsDown: out = unknown (index not stepped by a counted loop at line [[@LINE-2]])

// The header alone is not enough: a body that changes the index or the bound
// breaks what the header promised, and an inclusive bound reaches one past the
// range [0, Bound) can describe. Each would otherwise report [0, d) while
// writing outside it -- an under-approximation, the one error a caller cannot
// survive. (The bodies stay inside the buffer because this test also runs
// them; what the analysis reacts to is the assignment, not its value.)
void inclusiveBound(int d, double* out) {
  for (int i = 0; i <= d; ++i)
    out[i] = 1;
}
// CHECK: written-extent: inclusiveBound: out = unknown (index not stepped by a counted loop at line [[@LINE-2]])

void movesIndex(int d, double* out) {
  for (int i = 0; i < d; ++i) {
    i += 2;
    out[i] = 1;
  }
}
// CHECK: written-extent: movesIndex: out = unknown (index not stepped by a counted loop at line [[@LINE-3]])

void raisesBound(int d, double* out) {
  for (int i = 0; i < d; ++i) {
    out[i] = 1;
    d = 3;
  }
}
// CHECK: written-extent: raisesBound: out = unknown (index not stepped by a counted loop at line [[@LINE-4]])

// Taking the index's address hands it to anything: the loop no longer owns it.
void escapesIndex(int d, double* out) {
  for (int i = 0; i < d; ++i) {
    int* p = &i;
    out[i] = *p;
  }
}
// CHECK: written-extent: escapesIndex: out = unknown (index not stepped by a counted loop at line [[@LINE-3]])

// The bound has to hold still for the whole call, not only inside the loop: a
// caller works the range out from the argument it passed. This one writes
// [0, 2d) while a caller substituting its own d would record [0, d).
void raisesBoundBefore(int d, double* out) {
  d = d * 2;
  for (int i = 0; i < d; ++i)
    out[i] = 1;
}
// CHECK: written-extent: raisesBoundBefore: out = unknown (index not stepped by a counted loop at line [[@LINE-2]])

// A bound taken by reference names storage a callee can change under us, so
// the value a call site reads is not the value the loop will compare against.
void referenceBound(int& d, double* out) {
  for (int i = 0; i < d; ++i)
    out[i] = 1;
}
// CHECK: written-extent: referenceBound: out = unknown (the loop's bound is not one a call site can use at line [[@LINE-2]])

// Only `for` is recognised; an equivalent while loop is not.
void whileLoop(int n, double* out) {
  int i = 0;
  while (i < n) {
    out[i] = out[i] * 2;
    i++;
  }
}
// CHECK: written-extent: whileLoop: n = none
// CHECK-NEXT: written-extent: whileLoop: out = unknown (index not stepped by a counted loop at line [[@LINE-5]])

// A second index walked alongside the induction variable joins the increment
// with a comma. That still steps the induction variable by one.
void commaStep(int d, const double* w, double* out) {
  int p = 0;
  for (int i = 0; i < d; i++, p++)
    out[i] = w[p];
}
// CHECK: written-extent: commaStep: d = none
// CHECK-NEXT: written-extent: commaStep: w = none
// CHECK-NEXT: written-extent: commaStep: out = [0, d)

// A bound does not have to be spelled as a literal to be one. A dimension is
// usually a constexpr variable, an enumerator or a template argument, and a
// call site can work any of those out as readily as the number.
constexpr int Dim = 6;
enum { EnumDim = 5 };
void constexprBound(double* v) {
  for (int i = 0; i < Dim; i++)
    v[i] = v[i] * 2;
}
// CHECK: written-extent: constexprBound: v = [0, 6)

void enumBound(double* v) {
  for (int i = 0; i < EnumDim; i++)
    v[i] = v[i] * 2;
}
// CHECK: written-extent: enumBound: v = [0, 5)

// The loop is counted, but its bound is a local no call site can work out, so
// the range cannot be put in terms of this function's parameters.
void localBound(int d, double* out) {
  int n = d * 2;
  for (int i = 0; i < n; i++)
    out[i] = 1;
}
// CHECK: written-extent: localBound: d = none
// CHECK-NEXT: written-extent: localBound: out = unknown (the loop's bound is not one a call site can use at line [[@LINE-3]])

// A start below zero reaches out[-1], which [0, d) does not describe.
void negativeStart(int d, double* out) {
  for (int i = -1; i < d; ++i)
    out[i] = 1;
}
// CHECK: written-extent: negativeStart: out = unknown

// A bound that folds below zero is refused rather than zero-extended into
// almost the whole address space.
void negativeBound(double* out) {
  for (int i = 0; i < -1; ++i)
    out[i] = 1;
}
// CHECK: written-extent: negativeBound: out = unknown

// Two loops over the same buffer with different bounds do not describe one
// range, and the hull of the two is not something either loop proved.
void twoBounds(int d, int k, double* out) {
  for (int i = 0; i < d; ++i)
    out[i] = 1;
  for (int i = 0; i < k; ++i)
    out[i] = 2;
}
// CHECK: written-extent: twoBounds: out = unknown (writes do not describe one range at line [[@LINE-2]])

// The index may be declared outside the loop and only assigned in the header.
// That is the shape clad's own reverse sweeps are emitted in, so it has to be
// recognised as readily as the declaring form.
void assignedIndex(int d, double* out) {
  int i = 0;
  for (i = 0; i < d; ++i)
    out[i] = 1;
}
// CHECK: written-extent: assignedIndex: out = [0, d)

// A range already proven, then a write nothing bounds. The hull of the two is
// not something either write proved, so the parameter goes back to unknown.
void provenThenLoose(int d, double* out) {
  for (int i = 0; i < d; ++i)
    out[i] = 1;
  out[d + 1] = 2;
}
// CHECK: written-extent: provenThenLoose: out = unknown

// Shapes the recogniser declines, each still leaving the constant subscript
// in the body provable. A floating induction variable makes the iteration
// count depend on rounding.
void floatIndex(double* out) {
  for (double x = 0; x < 3; ++x)
    out[0] = out[0] + x;
}
// CHECK: written-extent: floatIndex: out = [0, 1)

// No condition at all bounds nothing.
void noCondition(double* out) {
  for (;;) {
    out[0] = 1;
    break;
  }
}
// CHECK: written-extent: noCondition: out = [0, 1)

// The condition compares an expression rather than the variable itself.
void computedCondition(int d, double* out) {
  for (int i = 0; i - d < 0; ++i)
    out[0] = 1;
}
// CHECK: written-extent: computedCondition: out = [0, 1)

// A header with no initialiser leaves the start value unknown.
void noInit(int d, double* out) {
  int i = 0;
  for (; i < d; ++i)
    out[0] = 1;
}
// CHECK: written-extent: noInit: out = [0, 1)

// A scalar written through a dereference is a single element.
void scalarOut(double a, double* err) { *err = a * a; }
// CHECK: written-extent: scalarOut: a = none
// CHECK-NEXT: written-extent: scalarOut: err = [0, 1)

double f(double a) {
  double x[4] = {a, a, a, a};
  double y[4] = {a, a, a, a};
  double o[4] = {0, 0, 0, 0};
  double o2[4] = {0, 0, 0, 0};
  subtract(4, x, y, o);
  qtimesx(4, x, y, o2);
  constIdx(o);
  fixedLoop(o);
  dataDependent(4, x, o);
  plusEqualOne(4, o);
  stepsByTwo(4, o);
  stepsByParameter(4, 2, o);
  countsDown(4, o);
  inclusiveBound(2, o);
  movesIndex(2, o);
  raisesBound(2, o);
  escapesIndex(2, o);
  raisesBoundBefore(2, o);
  int rd = 4;
  referenceBound(rd, o);
  whileLoop(4, o);
  commaStep(4, y, o2);
  constexprBound(o2);
  enumBound(o2);
  localBound(2, o);
  negativeStart(2, o);
  negativeBound(o);
  twoBounds(2, 3, o);
  assignedIndex(2, o);
  provenThenLoose(2, o);
  floatIndex(o);
  noCondition(o);
  computedCondition(2, o);
  noInit(2, o);
  double e = 0;
  scalarOut(a, &e);
  return o[0] + o2[0] + e;
}

int main() {
  auto g = clad::gradient(f);
  double da = 0;
  g.execute(1.0, &da);
  return 0;
}
