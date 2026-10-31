// RUN: %cladclang %s -I%S/../../include -oLiveness.out 2>&1 | %filecheck %s
// RUN: ./Liveness.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -fdisable-analysis=liveness \
// RUN:   %s -I%S/../../include -oLiveness.out 2>&1 \
// RUN:   | %filecheck --check-prefix=CHECK-OFF %s
// RUN: ./Liveness.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -Rclad-analysis=liveness \
// RUN:   -DREMARKS -fsyntax-only %s -I%S/../../include 2>&1 \
// RUN:   | %filecheck --check-prefix=CHECK-REMARK %s

// A call whose results nothing reads need not run in the forward sweep: the
// reverse sweep puts back what it changed before running its pullback.

#include "clad/Differentiator/Differentiator.h"

#include <cstdio>

struct S {
  double a, b;
};

void step(S& s, double k) {
  s.a = s.a * s.b;
  s.b = s.b * k;
}

// The second call is the last thing to touch s.
void twice(S& s, double k) {
  step(s, k);
  step(s, k);
}

// The forward call is spelled step or step_reverse_forw, depending on what
// the call records.
// CHECK: void twice_pullback(S &s, double k, S *_d_s, double *_d_k) {
// CHECK: {{step|step_reverse_forw}}(s, k
// CHECK-NOT: {{step|step_reverse_forw}}(s, k
// CHECK: step_pullback(s, k
// CHECK-OFF: void twice_pullback(S &s, double k, S *_d_s, double *_d_k) {
// CHECK-OFF: {{step|step_reverse_forw}}(s, k
// CHECK-OFF: {{step|step_reverse_forw}}(s, k

// The return reads s, so the call stays.
double square(double x) {
  S s{x, 2};
  twice(s, x);
  return s.a * s.a;
}

// CHECK: void square_grad(double x, double *_d_x) {
// CHECK: {{twice|twice_reverse_forw}}(s, x

// Nothing reads s after the call.
double unread(double x) {
  S s{x, 2};
  double r = s.a * x;
  twice(s, x);
  return r;
}

// CHECK: void unread_grad(double x, double *_d_x) {
// CHECK: double r = s.a * x;
// CHECK-NOT: {{twice|twice_reverse_forw}}(s, x
// CHECK: _d_r += 1;

// Leaving a call out would leave out the writes in its arguments.
double sideEffect(double x) {
  S s{x, 2};
  int i = 0;
  double r = s.a * x;
  step(s, x + i++);
  step(s, i = 2);
  return r * i;
}

// CHECK: void sideEffect_grad(double x, double *_d_x) {
// CHECK: {{step|step_reverse_forw}}(s,
// CHECK: {{step|step_reverse_forw}}(s,
// CHECK: step_pullback(s,

// Each of these keeps its call, and says why.
double escapes(double x) {
  S s{x, 2};
  S* p = &s;
  double r = p->a * x;
  step(s, x);
  // CHECK-REMARK: :[[@LINE-1]]:3: remark: clad runs this call in the forward sweep, although nothing reads what it computes there
  // CHECK-REMARK: :[[@LINE-2]]:3: note: the function takes the address of something this call writes, so a read through it cannot be ruled out
  // CHECK-REMARK: note: to avoid this, make it a call whose results nothing reads (CLAD1006)
  return r;
}

struct Q {
  double v;
  double* p;
  void scale(double k) { *p *= k; }
};

double holds(double x) {
  double y = x;
  Q q{x, &y};
  double r = y * x;
  q.scale(x);
  // CHECK-REMARK: :[[@LINE-1]]:3: remark: clad runs this call in the forward sweep, although nothing reads what it computes there
  // CHECK-REMARK: :[[@LINE-2]]:3: note: this call writes into an object that holds a pointer, so it may change memory outside it
  // CHECK-REMARK: note: to avoid this, make it a call whose results nothing reads (CLAD1006)
  return r;
}

void count(S& s) {
  static int calls = 0;
  calls++;
  s.a *= 2;
}

double global(double x) {
  S s{x, 2};
  double r = s.a * x;
  count(s);
  // CHECK-REMARK: :[[@LINE-1]]:3: remark: clad runs this call in the forward sweep, although nothing reads what it computes there
  // CHECK-REMARK: :[[@LINE-2]]:3: note: the callee, or a function it calls, writes a global
  // CHECK-REMARK: note: to avoid this, make it a call whose results nothing reads (CLAD1006)
  return r;
}

void copyOut(S& s, double* out) { *out = s.a; }

double pointer(double x) {
  S s{x, 2};
  double o = 0;
  double r = s.a * x;
  copyOut(s, &o);
  // CHECK-REMARK: :[[@LINE-1]]:3: remark: clad runs this call in the forward sweep, although nothing reads what it computes there
  // CHECK-REMARK: :[[@LINE-2]]:3: note: the callee writes through a pointer argument
  // CHECK-REMARK: note: to avoid this, make it a call whose results nothing reads (CLAD1006)
  return r;
}

[[clad::non_differentiable]] void runHook(S& s, void (*hook)(S&)) {
  if (hook)
    hook(s);
}
void callHook(S& s) {
  runHook(s, nullptr);
  s.a *= 2;
}

double indirect(double x) {
  S s{x, 2};
  double r = s.a * x;
  callHook(s);
  // CHECK-REMARK: :[[@LINE-1]]:3: remark: clad runs this call in the forward sweep, although nothing reads what it computes there
  // CHECK-REMARK: :[[@LINE-2]]:3: note: the callee, or a function it calls, calls through a pointer, so what it writes is not known
  // CHECK-REMARK: note: to avoid this, make it a call whose results nothing reads (CLAD1006)
  return r;
}

#ifdef REMARKS
void opaque(S& s);
namespace clad::custom_derivatives {
void opaque_pullback(S& s, S* d_s) { d_s->a *= 2; }
} // namespace clad::custom_derivatives

double noBody(double x) {
  S s{x, 2};
  double r = s.a * x;
  opaque(s);
  // CHECK-REMARK: :[[@LINE-1]]:3: remark: clad runs this call in the forward sweep, although nothing reads what it computes there
  // CHECK-REMARK: :[[@LINE-2]]:3: note: the callee has no body in this file, so what else it changes is not known
  // CHECK-REMARK: note: to avoid this, make it a call whose results nothing reads (CLAD1006)
  return r;
}

// Through an array, a pointer, a member and a reference argument.
void viaArray(S& s) {
  static double keep[1];
  keep[0] = s.a;
}
void viaPointer(S& s) {
  static double* keep = nullptr;
  *keep = s.a;
}
void viaMember(S& s) {
  static S keep;
  keep.a = s.a;
}
void viaArgument(S& s) {
  static S keep;
  step(keep, s.a);
}

double globals(double x) {
  S a{x, 2}, b{x, 2}, c{x, 2}, d{x, 2};
  double r = a.a * b.a * c.a * d.a * x;
  viaArray(a);
  // CHECK-REMARK: :[[@LINE-1]]:3: remark: clad runs this call in the forward sweep, although nothing reads what it computes there
  // CHECK-REMARK: :[[@LINE-2]]:3: note: the callee, or a function it calls, writes a global
  // CHECK-REMARK: note: to avoid this, make it a call whose results nothing reads (CLAD1006)
  viaPointer(b);
  // CHECK-REMARK: :[[@LINE-1]]:3: remark: clad runs this call in the forward sweep, although nothing reads what it computes there
  // CHECK-REMARK: :[[@LINE-2]]:3: note: the callee, or a function it calls, writes a global
  // CHECK-REMARK: note: to avoid this, make it a call whose results nothing reads (CLAD1006)
  viaMember(c);
  // CHECK-REMARK: :[[@LINE-1]]:3: remark: clad runs this call in the forward sweep, although nothing reads what it computes there
  // CHECK-REMARK: :[[@LINE-2]]:3: note: the callee, or a function it calls, writes a global
  // CHECK-REMARK: note: to avoid this, make it a call whose results nothing reads (CLAD1006)
  viaArgument(d);
  // CHECK-REMARK: :[[@LINE-1]]:3: remark: clad runs this call in the forward sweep, although nothing reads what it computes there
  // CHECK-REMARK: :[[@LINE-2]]:3: note: the callee, or a function it calls, writes a global
  // CHECK-REMARK: note: to avoid this, make it a call whose results nothing reads (CLAD1006)
  return r;
}

#endif

int main() {
  double dx = 0;
  clad::gradient(square).execute(3, &dx);
  printf("%.2f\n", dx); // CHECK-EXEC: 1728.00
  dx = 0;
  clad::gradient(unread).execute(3, &dx);
  printf("%.2f\n", dx); // CHECK-EXEC: 6.00
  dx = 0;
  clad::gradient(sideEffect).execute(3, &dx);
  printf("%.2f\n", dx); // CHECK-EXEC: 12.00
  dx = 0;
  clad::gradient(escapes).execute(3, &dx);
  printf("%.2f\n", dx); // CHECK-EXEC: 6.00
  dx = 0;
  clad::gradient(holds).execute(3, &dx);
  printf("%.2f\n", dx); // CHECK-EXEC: 6.00
  dx = 0;
  clad::gradient(global).execute(3, &dx);
  printf("%.2f\n", dx); // CHECK-EXEC: 6.00
  dx = 0;
  clad::gradient(pointer).execute(3, &dx);
  printf("%.2f\n", dx); // CHECK-EXEC: 6.00
  dx = 0;
  clad::gradient(indirect).execute(3, &dx);
  printf("%.2f\n", dx); // CHECK-EXEC: 6.00
#ifdef REMARKS
  clad::gradient(noBody);
  clad::gradient(globals);
#endif
}
