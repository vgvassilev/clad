// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -fdump-analysis=loop %s \
// RUN:   -I%S/../../include -oWrittenExtentsAffine.out 2>&1 | %filecheck %s
// RUN: ./WrittenExtentsAffine.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr %s \
// RUN:   -I%S/../../include -oWrittenExtentsAffine.out
// RUN: ./WrittenExtentsAffine.out | %filecheck_exec %s
//
// FileCheck rather than %filecheck below: the wrapper rejects any line with
// the word "note:" in it, and a report is mostly notes.
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -Rclad-analysis=loop \
// RUN:   -fsyntax-only %s -I%S/../../include 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-WHY %s

// A subscript that is a constant plus constant multiples of the indices of
// loops with constant bounds is bounded by its greatest value. `m[i * 6 + j]`
// over a 6 x 6 matrix is the shape.

#include "clad/Differentiator/Differentiator.h"

#include <cstdio>

// The row-major nest.
void rowMajor(double k, double* m) {
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 4; j++)
      m[i * 4 + j] = m[i * 4 + j] * k;
}
// CHECK: written-extent: rowMajor: k = none
// CHECK-NEXT: written-extent: rowMajor: m = [0, 12)

// A block offset in a local, a reversed index, an inclusive bound, and a
// subscript that folds to a constant without being spelled as one.
void blocks(double k, double* m) {
  for (int p = 0; p < 2; p++) {
    int o = 2 * p;
    m[(o + 1) * 4 + o] = m[(o + 1) * 4 + o] * k;
  }
  for (int i = 0; i < 4; i++)
    m[3 - i] = m[3 - i] * k;
  for (int i = 0; i <= 2; i++)
    m[i + 1] = m[i + 1] * k;
  m[1 * 4 + 0] = m[1 * 4 + 0] * k;
}
// CHECK: written-extent: blocks: k = none
// CHECK-NEXT: written-extent: blocks: m = [0, 15)

// A single value is a single element.
void folded(double k, double* m) { m[1 * 4 + 0] = m[1 * 4 + 0] * k; }
// CHECK: written-extent: folded: k = none
// CHECK-NEXT: written-extent: folded: m = [4, 5)

// None of the following is bounded, each for the reason in front of it.
//
// A stride the loop changes is no constant multiple.
void variantStride(double k, double* m) {
  int s = 1;
  for (int i = 0; i < 3; i++) {
    m[i * s] = m[i * s] * k;
    s = 2;
  }
}
// CHECK: written-extent: variantStride: k = none
// CHECK-NEXT: written-extent: variantStride: m = unknown (the index is neither a constant nor a variable at line [[@LINE-5]])

// Nor is a stride a caller passes, whose value is not known here.
void paramStride(int s, double k, double* m) {
  for (int i = 0; i < 3; i++)
    m[i * s] = m[i * s] * k;
}
// CHECK: written-extent: paramStride: s = none
// CHECK-NEXT: written-extent: paramStride: k = none
// CHECK-NEXT: written-extent: paramStride: m = unknown (the index is neither a constant nor a variable at line [[@LINE-4]])

// A product of two indices is not a sum of multiples of either.
void square(double k, double* m) {
  for (int i = 0; i < 3; i++)
    m[i * i] = m[i * i] * k;
}
// CHECK: written-extent: square: k = none
// CHECK-NEXT: written-extent: square: m = unknown (the index is neither a constant nor a variable at line [[@LINE-3]])

// A loop that can leave early need not reach its last index, and nothing
// says the buffer is as long as the one the bound implies.
void earlyExit(double k, double* m) {
  for (int i = 0; i < 8; i++) {
    if (m[2 * i] < 0)
      break;
    m[2 * i] = m[2 * i] * k;
  }
}
// CHECK: written-extent: earlyExit: k = none
// CHECK-NEXT: written-extent: earlyExit: m = unknown (the index is neither a constant nor a variable at line [[@LINE-4]])

// An inner start that moves with the outer index makes the two depend on
// each other, so the corners of the box need not be taken.
void triangle(double k, double* m) {
  for (int i = 0; i < 3; i++)
    for (int j = i; j < 3; j++)
      m[i * 3 + j] = m[i * 3 + j] * k;
}
// CHECK: written-extent: triangle: k = none
// CHECK-NEXT: written-extent: triangle: m = unknown (the index is neither a constant nor a variable at line [[@LINE-3]])

// An offset written again after its declaration is not its initialiser.
void movedOffset(double k, double* m) {
  for (int p = 0; p < 2; p++) {
    int o = 2 * p;
    o += 1;
    m[o] = m[o] * k;
  }
}
// CHECK: written-extent: movedOffset: k = none
// CHECK-NEXT: written-extent: movedOffset: m = unknown (no counted loop steps the index of this write at line [[@LINE-4]])

// An index declared before the loop could be reached before it.
void outsideIndex(double k, double* m) {
  int i = 0;
  for (i = 0; i < 3; i++)
    m[2 * i] = m[2 * i] * k;
}
// CHECK: written-extent: outsideIndex: k = none
// CHECK-NEXT: written-extent: outsideIndex: m = unknown (the index is neither a constant nor a variable at line [[@LINE-3]])

// A subscript that can go below zero is outside [0, anything), even where a
// branch keeps it from doing so: the analysis does not read conditions.
void belowZero(double k, double* m) {
  for (int i = 0; i < 3; i++)
    if (i > 0)
      m[i - 1] = m[i - 1] * k;
}
// CHECK: written-extent: belowZero: k = none
// CHECK-NEXT: written-extent: belowZero: m = unknown (the index is neither a constant nor a variable at line [[@LINE-3]])

// A signed step below zero on the way is fine: nothing wraps.
void signedDetour(double k, double* m) {
  for (int i = 0; i < 3; i++)
    m[i - 1 + 1] = m[i - 1 + 1] * k;
}
// CHECK: written-extent: signedDetour: k = none
// CHECK-NEXT: written-extent: signedDetour: m = [0, 3)

// The sum is only what the expression computes if no step of it wraps. An
// unsigned `i - 1u` does at i = 0, so even though adding one back lands in
// range, the analysis declines rather than reason modulo 2^32.
void wraps(double k, double* m) {
  for (unsigned i = 0; i < 3; i++)
    m[i - 1u + 1u] = m[i - 1u + 1u] * k;
}
// CHECK: written-extent: wraps: k = none
// CHECK-NEXT: written-extent: wraps: m = unknown (the index is neither a constant nor a variable at line [[@LINE-3]])

double f(double k) {
  double m[9][16];
  for (int r = 0; r < 9; r++)
    for (int c = 0; c < 16; c++)
      m[r][c] = k;
  rowMajor(k, m[0]);
  blocks(k, m[1]);
  folded(k, m[2]);
  variantStride(k, m[3]);
  paramStride(2, k, m[4]);
  square(k, m[5]);
  earlyExit(k, m[6]);
  triangle(k, m[7]);
  movedOffset(k, m[8]);
  outsideIndex(k, m[8]);
  belowZero(k, m[8]);
  signedDetour(k, m[8]);
  wraps(k, m[8]);
  double s = 0;
  for (int r = 0; r < 9; r++)
    for (int c = 0; c < 16; c++)
      s += m[r][c];
  return s;
}

// The bounded callees record a range; the rest keep their trackers.
// CHECK: void f_grad(double k, double *_d_k) {
// CHECK: clad::record_range(_rec0, m[0], 12UL);
// CHECK: clad::record_range(_rec1, m[1], 15UL);
// CHECK: variantStride_reverse_forw(
// CHECK: paramStride_reverse_forw(
// CHECK: square_reverse_forw(
// CHECK: earlyExit_reverse_forw(
// CHECK: triangle_reverse_forw(
// CHECK: movedOffset_reverse_forw(
// CHECK: outsideIndex_reverse_forw(
// CHECK: belowZero_reverse_forw(
// CHECK: clad::record_range(_rec2, m[8], 3UL);
// CHECK: wraps_reverse_forw(

int main() {
  double dk = 0;
  clad::gradient(f).execute(1.5, &dk);
  std::printf("%.4f\n", dk);
  // CHECK-EXEC: 308.1250
  return 0;
}

// The first refusal, as the reader sees it.
// CHECK-WHY-NOT: 'm': clad watches every address
// CHECK-WHY: WrittenExtentsAffine.cpp:[[# @LINE - 144]]:7: remark: 'm': clad watches every address written here instead of copying one range
// CHECK-WHY: note: the index is neither a constant nor a variable
// CHECK-WHY: note: to avoid this, make it a bounded write (CLAD1003)
