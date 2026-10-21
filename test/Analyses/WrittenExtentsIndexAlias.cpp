// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -fdump-analysis=loop %s \
// RUN:   -I%S/../../include -oWrittenExtentsIndexAlias.out 2>&1 | %filecheck %s
// RUN: ./WrittenExtentsIndexAlias.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr %s \
// RUN:   -I%S/../../include -oWrittenExtentsIndexAlias.out
// RUN: ./WrittenExtentsIndexAlias.out | %filecheck_exec %s

// A callee taking the index, or the bound, by `int&` moves it, so a write
// indexed by it is not inside [0, bound).

#include "clad/Differentiator/Differentiator.h"

#include <cstdio>

void skip(int& i) { i += 2; }

void viaCallee(double* out) {
  for (int i = 0; i < 4; i++) {
    skip(i);
    out[i] = out[i] * out[i];
  }
}
// CHECK: written-extent: viaCallee: out = unknown (no counted loop steps the index of this write at line [[@LINE-3]])

// The same holds for the bound, over the whole function.
void boundViaCallee(int d, double* out) {
  skip(d);
  for (int i = 0; i < d; i++)
    out[i] = out[i] * out[i];
}
// CHECK: written-extent: boundViaCallee: d = none
// CHECK-NEXT: written-extent: boundViaCallee: out = unknown (no counted loop steps the index of this write at line [[@LINE-3]])

double f(double x) {
  double a[8] = {x, x, x, x, x, x, x, x};
  viaCallee(a);
  viaCallee(a);
  double b[8] = {x, x, x, x, x, x, x, x};
  boundViaCallee(2, b);
  boundViaCallee(2, b);
  return a[5] + b[3];
}

int main() {
  double dx = 0;
  clad::gradient(f).execute(1.5, &dx);
  // a[5] and b[3] are each squared twice, x^4, so 8x^3.
  std::printf("%.4f\n", dx);
  // CHECK-EXEC: 27.0000
  return 0;
}
