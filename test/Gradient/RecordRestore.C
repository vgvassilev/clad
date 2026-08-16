// RUN: %cladclang %s -I%S/../../include -oRecordRestore.out 2>&1 | %filecheck %s
// RUN: ./RecordRestore.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr %s \
// RUN:   -I%S/../../include -oRecordRestore.out
// RUN: ./RecordRestore.out | %filecheck_exec %s

// Verifies that a record stored on a tape by copy is also restored by copy
// (clad::back then clad::pop). clad::pop returns the element moved out of its
// slot, and a move-assigning restore would replace the record's heap storage,
// dangling any reference previously handed out into it.

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

struct S {
  double a;
  double b;
};

void scale(S& s) {
  s.a = 3 * s.a;
  s.b = 3 * s.b;
}

double f(double x) {
  S s;
  s.a = x;
  s.b = 2 * x;
  double t = 0;
  for (int i = 0; i < 1; i++) {
    scale(s);
    t += s.a * s.a + s.b;
  }
  return t;
}

// CHECK: void f_grad(double x, double *_d_x) {
// CHECK: clad::push(_t1, s);
// CHECK: scale_reverse_forw(s, _d_s, _tracker0);
// CHECK: _tracker0.restore();
// CHECK-NEXT: {
// CHECK-NEXT:     s = clad::back(_t1);
// CHECK-NEXT:     clad::pop(_t1);
// CHECK-NEXT: }
// CHECK-NEXT: scale_pullback(s, &_d_s);

int main() {
  auto g = clad::gradient(f);
  double dx = 0;
  g.execute(1.0, &dx);
  // f = (3x)^2 + 6x => df/dx = 18x + 6
  printf("dx=%.2f\n", dx); // CHECK-EXEC: dx=24.00
}
