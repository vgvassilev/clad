// RUN: %cladclang %s -I%S/../../include -oLocalReferenceRecords.out 2>&1 | %filecheck %s
// RUN: ./LocalReferenceRecords.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr %s \
// RUN:   -I%S/../../include -oLocalReferenceRecords.out
// RUN: ./LocalReferenceRecords.out | %filecheck_exec %s

// A reverse_forw does not record storage its own locals own, since the
// caller's restore would write into a dead frame. A local reference bound to
// such a local names the same storage.

#include "clad/Differentiator/Differentiator.h"

#include <cstdio>

void scale(double& v) { v = 3 * v; }

void bumpLocal(double* out) {
  double t = out[0];
  double& rt = t;
  rt *= 2;
  scale(rt);
  out[1] = t;
}

// CHECK: inline void bumpLocal_reverse_forw(double *out, double *_d_out, clad::restore_tracker &_tracker0) {
// CHECK-NOT:      _tracker0.store({{r?}}t)
// CHECK:          _tracker0.store(out[1]);
// CHECK-NEXT:     out[1] = t;

void bumpIndex(double* out) {
  for (int i = 0; i < 4; i++) {
    int& r = i;
    r += 2;
    out[i] = out[i] * out[i];
  }
}

// CHECK: inline void bumpIndex_reverse_forw(double *out, double *_d_out, clad::restore_tracker &_tracker0) {
// CHECK:          int &r = i;
// CHECK-NEXT:     r += 2;
// CHECK-NEXT:     _tracker0.store(out[i]);
// CHECK-NEXT:     out[i] = out[i] * out[i];

double f(double x) {
  double a[8] = {x, x, x, x, x, x, x, x};
  bumpLocal(a);
  bumpIndex(a);
  return a[1] + a[2] + a[5];
}

int main() {
  double dx = 0;
  clad::gradient(f).execute(1.5, &dx);
  // a[1] = 6x, and the loop squares a[2] and a[5]: 6 + 4x.
  std::printf("%.4f\n", dx); // CHECK-EXEC: 12.0000
  return 0;
}
