// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -fdump-analysis=loop %s \
// RUN:   -I%S/../../include -oWrittenExtentsLocalArray.out 2>&1 | %filecheck %s
// RUN: ./WrittenExtentsLocalArray.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr %s \
// RUN:   -I%S/../../include -oWrittenExtentsLocalArray.out
// RUN: ./WrittenExtentsLocalArray.out | %filecheck_exec %s
//
// FileCheck rather than %filecheck below: the wrapper rejects any line with
// the word "note:" in it, and a report is mostly notes.
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -Rclad-analysis=loop \
// RUN:   -fsyntax-only %s -I%S/../../include 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-WHY %s

// A write into an array this function declares lands in its own frame, which
// no parameter can alias: an array subscripted in place, or a member of one
// reached with `.`.

#include "clad/Differentiator/Differentiator.h"

#include <cstdio>

// The shape of a transport map assembled in a scratch array and then applied
// to the buffer in one counted loop. Each callee reads what it overwrites, so
// its pullback needs the values from before the call and the caller has to
// keep them.
void fill(double k, double* out) {
  double loc[4] = {0., 0., 0., 0.};
  loc[1] = k;
  loc[2] = k * k;
  for (int n = 0; n < 4; n++)
    out[n] = out[n] * loc[n];
}
// CHECK: written-extent: fill: k = none
// CHECK-NEXT: written-extent: fill: out = [0, 4)

// Two dimensions decay twice; a member reached with `.` is still inside the
// local.
struct Pair {
  double v[2];
};
void fill2d(double k, double* out) {
  double m[2][2];
  Pair p;
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 2; j++)
      m[i][j] = k * (i + j);
  p.v[0] = m[1][1];
  p.v[1] = *m[1];
  for (int n = 0; n < 2; n++)
    out[n] = out[n] * p.v[n];
}
// CHECK: written-extent: fill2d: k = none
// CHECK-NEXT: written-extent: fill2d: out = [0, 2)

// The array is local but its elements are pointers, and one of them points
// at `out`: the write below lands in the parameter's buffer. (Its value is a
// constant, and the caller does not read it back: what is under test is the
// extent, and clad loses the adjoint of a store through a pointer it keeps in
// an array or a member, which is a different matter.)
void viaPointerArray(double k, double* out) {
  double* ptrs[1] = {out};
  ptrs[0][1] = 2.;
  out[0] = k * out[0];
}
// CHECK: written-extent: viaPointerArray: k = none
// CHECK-NEXT: written-extent: viaPointerArray: out = unknown (this writes through a pointer that cannot be traced back to a parameter at line [[@LINE-4]])

// A local pointer holding a parameter's address is still that buffer.
void viaLocalPointer(double k, double* out) {
  double* q = out;
  q[1] = k * q[1];
  out[0] = k * out[0];
}
// CHECK: written-extent: viaLocalPointer: k = none
// CHECK-NEXT: written-extent: viaLocalPointer: out = unknown (this writes through a pointer that cannot be traced back to a parameter at line [[@LINE-4]])

// A local struct whose member points elsewhere is no different.
struct Ref {
  double* p;
};
void viaPointerMember(double k, double* out) {
  Ref r = {out};
  r.p[1] = 2.;
  out[0] = k * out[0];
}
// CHECK: written-extent: viaPointerMember: k = none
// CHECK-NEXT: written-extent: viaPointerMember: out = unknown (this writes through a pointer that cannot be traced back to a parameter at line [[@LINE-4]])

// A local reference names an object that is not this frame's.
void viaLocalReference(double k, Pair* b, double* out) {
  Pair& r = *b;
  r.v[1] = k * r.v[1];
  out[0] = k * out[0];
}
// CHECK: written-extent: viaLocalReference: k = none
// CHECK-NEXT: written-extent: viaLocalReference: b = unknown (this writes through a pointer that cannot be traced back to a parameter at line [[@LINE-4]])
// CHECK-NEXT: written-extent: viaLocalReference: out = unknown (this writes through a pointer that cannot be traced back to a parameter at line [[@LINE-5]])

double f(double k) {
  double r[4] = {k, k, k, k};
  fill(k, r);
  double s[2] = {k, k};
  fill2d(k, s);
  double a[2] = {k, k}, b[2] = {k, k}, c[2] = {k, k}, d[2] = {k, 0};
  Pair e = {{0, k}};
  viaPointerArray(k, a);
  viaLocalPointer(k, b);
  viaPointerMember(k, c);
  viaLocalReference(k, &e, d);
  return r[1] + r[2] + s[0] + s[1] + a[0] + b[0] * b[1] + c[0] +
         e.v[1] * d[0];
}

// With the extent proven, the call records the range it writes instead of
// tracking every address; the negative cases keep the tracker.
// CHECK: void f_grad(double k, double *_d_k) {
// CHECK: clad::record_range(_rec0, r, 4UL);
// CHECK: clad::record_range(_rec1, s, 2UL);
// CHECK: viaPointerArray_reverse_forw(
// CHECK: viaLocalPointer_reverse_forw(
// CHECK: viaPointerMember_reverse_forw(
// CHECK: viaLocalReference_reverse_forw(

int main() {
  double dk = 0;
  clad::gradient(f).execute(1.5, &dk);
  // 2k + 3k^2 from fill, 4k + 2k from fill2d, 2k from each of a[0] and c[0],
  // and 4k^3 from each of the two products that come to k^4.
  std::printf("%.2f\n", dk);
  // CHECK-EXEC: 51.75
  return 0;
}

// The writes into the locals cost nothing: the first bounded-write report is
// the one through `ptrs`.
// CHECK-WHY-NOT: bounded write
// CHECK-WHY: WrittenExtentsLocalArray.cpp:62:3: remark: 'out': clad watches every address written here instead of copying one range
// CHECK-WHY: note: this writes through a pointer that cannot be traced back to a parameter
// CHECK-WHY: note: to avoid this, make it a bounded write (CLAD1003)
