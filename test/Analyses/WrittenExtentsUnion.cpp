// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -fdump-analysis=loop %s \
// RUN:   -I%S/../../include -oWrittenExtentsUnion.out 2>&1 | %filecheck %s
// RUN: ./WrittenExtentsUnion.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr %s \
// RUN:   -I%S/../../include -oWrittenExtentsUnion.out
// RUN: ./WrittenExtentsUnion.out | %filecheck_exec %s
//
// FileCheck rather than %filecheck below: the wrapper rejects any line with
// the word "note:" in it, and a report is mostly notes.
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -Rclad-analysis=loop \
// RUN:   -fsyntax-only %s -I%S/../../include 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-WHY %s

// Writes to one buffer combine when one provably contains the rest, as a
// constant range does. Clearing a matrix and then setting a few entries is
// the common shape.

#include "clad/Differentiator/Differentiator.h"

#include <cstdio>

void clearThenSet(double k, double* v) {
  for (int i = 0; i < 4; i++)
    v[i] = v[i] * k;
  v[2] = v[2] * k;
  *v = v[0] * k;
}
// CHECK: written-extent: clearThenSet: k = none
// CHECK-NEXT: written-extent: clearThenSet: v = [0, 4)

// The order does not matter: the range may come after the element.
void setThenClear(double k, double* v) {
  v[3] = v[3] * k;
  for (int i = 0; i < 4; i++)
    v[i] = v[i] * k;
}
// CHECK: written-extent: setThenClear: k = none
// CHECK-NEXT: written-extent: setThenClear: v = [0, 4)

// Two constant ranges nest, and the longer one is their union.
void twoConstRanges(double k, double* v) {
  for (int i = 0; i < 2; i++)
    v[i] = v[i] * k;
  for (int i = 0; i < 4; i++)
    v[i] = v[i] * k;
  for (int i = 0; i < 3; i++)
    v[i] = v[i] * k;
}
// CHECK: written-extent: twoConstRanges: k = none
// CHECK-NEXT: written-extent: twoConstRanges: v = [0, 4)

// An element at the bound is outside the range, and [0, 5) would claim the
// whole of it written; nothing proves the buffer is that long until the
// element is, so this stays a disagreement.
void elementPastEnd(double k, double* v) {
  for (int i = 0; i < 3; i++)
    v[i] = v[i] * k;
  v[3] = v[3] * k;
}
// CHECK: written-extent: elementPastEnd: k = none
// CHECK-NEXT: written-extent: elementPastEnd: v = unknown (two writes here cover ranges that are not one range at line [[@LINE-3]])

// A bound a caller passes is not known here, so nothing says the element is
// inside it.
void paramRangeAndElement(int d, double k, double* v) {
  for (int i = 0; i < d; i++)
    v[i] = v[i] * k;
  v[1] = v[1] * k;
}
// CHECK: written-extent: paramRangeAndElement: d = none
// CHECK-NEXT: written-extent: paramRangeAndElement: k = none
// CHECK-NEXT: written-extent: paramRangeAndElement: v = unknown (two writes here cover ranges that are not one range at line [[@LINE-4]])

// Neither does a constant range say anything about one bounded by a
// parameter, however the two are ordered.
void constAndParamRange(int d, double k, double* v) {
  for (int i = 0; i < 4; i++)
    v[i] = v[i] * k;
  for (int i = 0; i < d; i++)
    v[i] = v[i] * k;
}
// CHECK: written-extent: constAndParamRange: d = none
// CHECK-NEXT: written-extent: constAndParamRange: k = none
// CHECK-NEXT: written-extent: constAndParamRange: v = unknown (two writes here cover ranges that are not one range at line [[@LINE-4]])

double f(double k) {
  double a[4] = {k, k, k, k};
  double b[4] = {k, k, k, k};
  double c[4] = {k, k, k, k};
  double d[4] = {k, k, k, k};
  double e[4] = {k, k, k, k};
  double g[4] = {k, k, k, k};
  clearThenSet(k, a);
  setThenClear(k, b);
  twoConstRanges(k, c);
  elementPastEnd(k, d);
  paramRangeAndElement(2, k, e);
  constAndParamRange(2, k, g);
  return a[0] + a[2] + b[3] + c[1] + c[3] + d[3] + e[1] + g[1];
}

// The three provable callees record one range each; the rest keep a tracker.
// CHECK: void f_grad(double k, double *_d_k) {
// CHECK: clad::record_range(_rec0, a, 4UL);
// CHECK: clad::record_range(_rec1, b, 4UL);
// CHECK: clad::record_range(_rec2, c, 4UL);
// CHECK: elementPastEnd_reverse_forw(
// CHECK: paramRangeAndElement_reverse_forw(
// CHECK: constAndParamRange_reverse_forw(

int main() {
  double dk = 0;
  clad::gradient(f).execute(1.5, &dk);
  // a[0], a[2], b[3], e[1] and g[1] come to k^3, c[1] to k^4, and c[3] and
  // d[3] to k^2: 15k^2 + 4k^3 + 4k.
  std::printf("%.4f\n", dk);
  // CHECK-EXEC: 53.2500
  return 0;
}

// What the reader is told at the first write the union refuses.
// CHECK-WHY-NOT: 'v': clad watches every address
// CHECK-WHY: WrittenExtentsUnion.cpp:[[# @LINE - 65]]:3: remark: 'v': clad watches every address written here instead of copying one range
// CHECK-WHY: note: two writes here cover ranges that are not one range
// CHECK-WHY: note: to avoid this, make it a bounded write (CLAD1003)
