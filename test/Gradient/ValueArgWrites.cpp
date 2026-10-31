// RUN: %cladclang %s -I%S/../../include -oValueArgWrites.out 2>&1 \
// RUN:   | %filecheck %s
// RUN: ./ValueArgWrites.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -fdisable-analysis=tbr \
// RUN:   %s -I%S/../../include -oValueArgWrites.out
// RUN: ./ValueArgWrites.out | %filecheck_exec %s

// A struct passed by value can point into the caller's memory, through a
// pointer or a reference. A callee that writes through it changes that
// memory, which its pullback replays from: the call has to record what it
// changed there.

#include "clad/Differentiator/Differentiator.h"

#include <cstdio>

struct View {
  double* p;
};

void square(View v) { v.p[0] *= v.p[0]; }

double f(double x) {
  double y = x;
  square(View{&y});
  return y;
}

// CHECK: void f_grad(double x, double *_d_x) {
// CHECK: square_reverse_forw(View({&y}), {&_d_y}, _tracker0);
// CHECK: _tracker0.restore();
// CHECK-NEXT: View _r0 = {&_d_y};
// CHECK-NEXT: square_pullback(View({&y}), &_r0);

struct Ref {
  double& r;
};

void scale(Ref q, double k) { q.r *= k; }

double g(double x) {
  double y = x;
  scale(Ref{y}, x);
  return y;
}

// CHECK: void scale_reverse_forw(Ref q, double k, Ref _d_q, double _d_k, clad::restore_tracker &_tracker0) {
// CHECK-NEXT: _tracker0.store(q.r);

int main() {
  double dx = 0;
  clad::gradient(f).execute(3, &dx);
  printf("%.2f\n", dx); // CHECK-EXEC: 6.00
  dx = 0;
  clad::gradient(g).execute(3, &dx);
  printf("%.2f\n", dx); // CHECK-EXEC: 6.00
}
