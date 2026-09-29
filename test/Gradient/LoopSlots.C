// RUN: %cladclang %s -I%S/../../include -oLoopSlots.out 2>&1 | %filecheck %s
// RUN: ./LoopSlots.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr %s -I%S/../../include -oLoopSlots.out
// RUN: ./LoopSlots.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -Rclad-analysis=loop %s \
// RUN:   -I%S/../../include -oLoopSlots.out 2>&1 | %filecheck --check-prefix=CHECK-REPORT %s
//
// A value stored once per iteration of a loop the analysis counted with a
// literal takes a slot in an array of that many, at the loop's index, instead
// of a push onto a tape; the reverse sweep steps the index back and reads the
// slot by it. The plain cases are in Loops.C and the other loop tests; here
// are the edges: where the index starts, where the array would be too big,
// where the loop never runs, where a return can skip it, where a condition is
// what is stored, and what the report says of a loop whose count is not a
// literal.

#include "clad/Differentiator/Differentiator.h"
#include "../TestUtils.h"

// An index that does not start at zero is offset into the array.
double offsetSlot(double x) {
  double t = 1;
  for (int i = 2; i < 5; i++)
    t *= x * i;
  return t;
}

// CHECK: void offsetSlot_grad(double x, double *_d_x) {
// CHECK-NEXT:     int _d_i = 0;
// CHECK-NEXT:     int i = 0;
// CHECK-NEXT:     double _t1[3];
// CHECK-NEXT:     double _d_t = 0.;
// CHECK-NEXT:     double t = 1;
// CHECK-NEXT:     unsigned {{int|long|long long}} _t0;
// CHECK-NEXT:     for (i = 2; i < 5; i++) {
// CHECK-NEXT:         _t1[i - 2] = t;
// CHECK-NEXT:         t *= x * i;
// CHECK-NEXT:     }
// CHECK-NEXT:     _d_t += 1;
// CHECK-NEXT:     for (_t0 = 3{{U|UL|ULL}}; _t0; _t0--) {
// CHECK-NEXT:         i--;
// CHECK-NEXT:         t = _t1[i - 2];
// CHECK-NEXT:         double _r_d0 = _d_t;
// CHECK-NEXT:         _d_t = 0.;
// CHECK-NEXT:         _d_t += _r_d0 * (x * i);
// CHECK-NEXT:         *_d_x += t * _r_d0 * i;
// CHECK-NEXT:         _d_i += x * t * _r_d0;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// Past a page of slots the tape, which grows on the heap, is the right home.
double bigLoop(double x) {
  double t = 1;
  for (int i = 0; i < 600; i++)
    t = t * x + 1;
  return t;
}

// CHECK: void bigLoop_grad(double x, double *_d_x) {
// CHECK: clad::tape<double> _t1 = {};
// CHECK-NOT: _t1[i]

// A loop that never runs has no slot to declare: an array of none is not a
// type, so the store keeps its tape, which no push ever reaches.
double emptyLoop(double x) {
  double t = 1;
  for (int i = 0; i < 0; i++)
    t *= x;
  return t;
}

// CHECK: void emptyLoop_grad(double x, double *_d_x) {
// CHECK: clad::tape<double> _t1 = {};

// A loop a return can skip is not counted at all -- the reverse loop would
// run over a record the forward sweep never wrote -- so its store keeps the
// tape, whose pops the reverse loop's own counter guards.
double earlyReturnLoop(double x) {
  if (x < 0)
    return x;
  double t = 1;
  for (int i = 0; i < 3; i++)
    t *= x;
  return t;
}

// CHECK: void earlyReturnLoop_grad(double x, double *_d_x) {
// CHECK: clad::tape<double> _t1 = {};
// CHECK-NOT: _t1[i]

// CHECK-REPORT: LoopSlots.C:[[# @LINE - 9]]:3: remark: clad adds a counter to this loop and increments it every iteration
// CHECK-REPORT: LoopSlots.C:[[# @LINE - 10]]:3: note: the function can return before reaching this loop
// CHECK-REPORT: LoopSlots.C:[[# @LINE - 11]]:3: note: to avoid this, make it a counted loop (CLAD1001)

// A branch condition inside the loop takes a slot too.
double condSlot(double x) {
  double s = 0;
  for (int i = 0; i < 4; i++) {
    double v = x * i;
    s += (v > x) ? v * v : v;
  }
  return s;
}

// CHECK: void condSlot_grad(double x, double *_d_x) {
// CHECK: bool _cond0[4];
// CHECK: _cond0[i] = (v > x);
// CHECK: if (_cond0[i]) {

// A bound the loop cannot count keeps the tape, and the report says why.
double runtimeBound(double x, int n) {
  double t = 1;
  for (int i = 0; i < n; i++)
    t *= x;
  return t;
}

// CHECK: void runtimeBound_grad_0(double x, int n, double *_d_x) {
// CHECK: clad::tape<double> _t1 = {};
// CHECK: clad::push(_t1, t);
// CHECK: t = clad::pop(_t1);

// CHECK-REPORT: LoopSlots.C:[[# @LINE - 10]]:3: remark: clad pushes what this loop stores onto a tape and pops it back, instead of keeping it in an array indexed by the loop
// CHECK-REPORT: LoopSlots.C:[[# @LINE - 11]]:23: note: the start or the bound is not a literal, so the count is not known until run time
// CHECK-REPORT: LoopSlots.C:[[# @LINE - 12]]:3: note: to avoid this, make it a loop with a literal count (CLAD1005)

int main() {
  double dx = 0;
  INIT_GRADIENT(offsetSlot);
  TEST_GRADIENT(offsetSlot, /*numOfDerivativeArgs=*/1, 1, &dx); // CHECK-EXEC: {72.00}

  dx = 0;
  INIT_GRADIENT(bigLoop);
  // t = x^600 + x^599 + ... + 1; at x = 1 the derivative is 600 * 601 / 2.
  TEST_GRADIENT(bigLoop, /*numOfDerivativeArgs=*/1, 1, &dx); // CHECK-EXEC: {180300.00}

  dx = 0;
  INIT_GRADIENT(emptyLoop);
  TEST_GRADIENT(emptyLoop, /*numOfDerivativeArgs=*/1, 2, &dx); // CHECK-EXEC: {0.00}

  dx = 0;
  INIT_GRADIENT(earlyReturnLoop);
  TEST_GRADIENT(earlyReturnLoop, /*numOfDerivativeArgs=*/1, 2, &dx); // CHECK-EXEC: {12.00}
  dx = 0;
  TEST_GRADIENT(earlyReturnLoop, /*numOfDerivativeArgs=*/1, -2, &dx); // CHECK-EXEC: {1.00}

  dx = 0;
  INIT_GRADIENT(condSlot);
  TEST_GRADIENT(condSlot, /*numOfDerivativeArgs=*/1, 1, &dx); // CHECK-EXEC: {27.00}

  dx = 0;
  INIT_GRADIENT(runtimeBound, "x");
  TEST_GRADIENT(runtimeBound, /*numOfDerivativeArgs=*/1, 2, 3, &dx); // CHECK-EXEC: {12.00}
}
