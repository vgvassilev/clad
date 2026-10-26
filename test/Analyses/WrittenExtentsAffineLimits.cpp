// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -fdump-analysis=loop \
// RUN:   -fsyntax-only %s -I%S/../../include 2>&1 | %filecheck %s

// A negated index is affine too. Subscripts whose constants or ranges are too
// large to reason about safely are refused. These are only analyzed, never
// run.

#include "clad/Differentiator/Differentiator.h"

void negated(double k, double* m) {
  for (int i = 0; i < 4; ++i)
    m[-i + 3] = m[-i + 3] * k;
}
// CHECK: written-extent: negated: k = none
// CHECK-NEXT: written-extent: negated: m = [0, 4)

void hugeConstant(double k, double* m) {
  for (long i = 0; i < 4; ++i)
    m[i + 2199023255552L] = m[i + 2199023255552L] * k;
}
// CHECK: written-extent: hugeConstant: k = none
// CHECK-NEXT: written-extent: hugeConstant: m = unknown

void hugeRange(double k, double* m) {
  for (long i = 0; i < 4; ++i)
    m[i * 549755813888L] = m[i * 549755813888L] * k;
}
// CHECK: written-extent: hugeRange: k = none
// CHECK-NEXT: written-extent: hugeRange: m = unknown

void hugeCoefficient(double k, double* m) {
  for (long i = 0; i < 4; ++i)
    m[(i * 1048576L) * 549755813888L] = k;
}
// CHECK: written-extent: hugeCoefficient: k = none
// CHECK-NEXT: written-extent: hugeCoefficient: m = unknown

void hugeOffset(double k, double* m) {
  for (long i = 0; i < 4; ++i)
    m[(i + 549755813888L) * 549755813888L] = k;
}
// CHECK: written-extent: hugeOffset: k = none
// CHECK-NEXT: written-extent: hugeOffset: m = unknown

// Only unary plus and minus keep an index affine.
void notIndex(double k, double* m) {
  for (int i = 0; i < 4; ++i)
    m[!i] = k;
}
// CHECK: written-extent: notIndex: k = none
// CHECK-NEXT: written-extent: notIndex: m = unknown

double f(double k) {
  double m[8] = {};
  negated(k, m);
  hugeConstant(k, m);
  hugeRange(k, m);
  hugeCoefficient(k, m);
  hugeOffset(k, m);
  notIndex(k, m);
  return m[0];
}

int main() { clad::gradient(f); }
