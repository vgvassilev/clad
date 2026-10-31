// RUN: %cladclang %s -I%S/../../include -oCountedLoopIndexAlias.out
// RUN: ./CountedLoopIndexAlias.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr %s \
// RUN:   -I%S/../../include -oCountedLoopIndexAlias.out
// RUN: ./CountedLoopIndexAlias.out | %filecheck_exec %s
//
// FileCheck rather than %filecheck below: the wrapper rejects any line with
// the word "note:" in it, and a report is mostly notes.
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -Rclad-analysis=loop \
// RUN:   -fsyntax-only %s -I%S/../../include 2>&1 | FileCheck %s

// A local reference bound to the index moves it like an assignment does, so
// the loop is not counted.

#include "clad/Differentiator/Differentiator.h"

#include <cstdio>

void viaReference(double* out) {
  // CHECK: CountedLoopIndexAlias.cpp:[[@LINE+3]]:3: remark: clad adds a counter to this loop and increments it every iteration
  // CHECK: note: the body assigns the index too
  // CHECK: note: to avoid this, make it a counted loop (CLAD1001)
  for (int i = 0; i < 4; i++) {
    int& r = i;
    r += 2;
    out[i] = out[i] * out[i];
  }
}

// A reference to const cannot write, and leaves the loop counted.
// CHECK-NOT: CountedLoopIndexAlias.cpp:[[@LINE+2]]:3: remark: clad adds a counter
void viaConstReference(double* out) {
  for (int i = 0; i < 4; i++) {
    const int& r = i;
    out[r] = out[r] * out[r];
  }
}

double f(double x) {
  double a[8] = {x, x, x, x, x, x, x, x};
  viaReference(a);
  viaConstReference(a);
  return a[2] + a[5] + a[3];
}

int main() {
  double dx = 0;
  clad::gradient(f).execute(1.5, &dx);
  // a[2] and a[5] are squared, then a[2] and a[3] are, so x^4 + x^2 + x^2:
  // 4x^3 + 4x.
  std::printf("%.4f\n", dx);
  // CHECK-EXEC: 19.5000
  return 0;
}
