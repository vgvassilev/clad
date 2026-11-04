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
  double* po = &o;
  double r = s.a * x;
  copyOut(s, po);
  // CHECK-REMARK: :[[@LINE-1]]:3: remark: clad runs this call in the forward sweep, although nothing reads what it computes there
  // CHECK-REMARK: :[[@LINE-2]]:3: note: the callee writes through a pointer argument
  // CHECK-REMARK: note: to avoid this, make it a call whose results nothing reads (CLAD1006)
  return r;
}

// `&o` writes o, as passing o by reference does, and nothing reads o after.
double address(double x) {
  S s{x, 2};
  double o = 0;
  double r = s.a * x;
  copyOut(s, &o);
  return r;
}

// CHECK: void address_grad(double x, double *_d_x) {
// CHECK: double r = s.a * x;
// CHECK-NOT: copyOut(s, &o);
// CHECK: _d_r += 1;

// The address of o is also taken outside the call, so o may be read later.
double addressTwice(double x) {
  S s{x, 2};
  double o = 0;
  double* q = &o;
  double r = s.a * x;
  copyOut(s, &o);
  // CHECK-REMARK: :[[@LINE-1]]:3: remark: clad runs this call in the forward sweep, although nothing reads what it computes there
  // CHECK-REMARK: :[[@LINE-2]]:3: note: the function takes the address of something this call writes, so a read through it cannot be ruled out
  // CHECK-REMARK: note: to avoid this, make it a call whose results nothing reads (CLAD1006)
  return r + *q;
}

void cube(double x, double* out) { *out = x * x * x; }

// Only the return reads out. A gradient does not use the value returned.
double returned(double x) {
  double out;
  cube(x, &out);
  return out;
}

// CHECK: void returned_grad(double x, double *_d_x) {
// CHECK-NOT: cube(x, &out);
// CHECK: cube_pullback(x, &out

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

void sqRef(double& v) { v = v * v; }

// The call writes t through a local reference to it, and t is read after.
double throughLocalRef(double x) {
  double t = 2 * x;
  double& rt = t;
  sqRef(rt);
  return t * x;
}

// The pullback leaves out the stores into out: its reverse sweep reads only Q
// and x, and the caller restores out where it reads it.
void matvec(const double* __restrict__ Q, const double* __restrict__ x,
            double* __restrict__ out) {
  for (int i = 0; i < 2; i++) {
    out[i] = Q[2 * i] * x[0];
    out[i] = out[i] + Q[2 * i + 1] * x[1];
  }
}

double useMatvec(double a) {
  double Q[4] = {a, 1, 2, a};
  double x[2] = {a, 3};
  double out[2];
  matvec(Q, x, out);
  return out[0] * out[1];
}

// CHECK: void matvec_pullback(const double *__restrict Q, const double *__restrict x, double *__restrict out, double *_d_Q, double *_d_x, double *_d_out) {
// CHECK-NEXT:     int _d_i = 0;
// CHECK-NEXT:     int i = 0;
// CHECK-NEXT:     unsigned int _t0;
// CHECK-NEXT:     for (i = 0; i < 2; i++) {
// CHECK-NEXT:     }

// Without restrict, out may alias Q or x, which the reverse sweep reads, so
// the stores stay.
void matvecAlias(const double* Q, const double* x, double* out) {
  for (int i = 0; i < 2; i++) {
    out[i] = Q[2 * i] * x[0];
    out[i] = out[i] + Q[2 * i + 1] * x[1];
  }
}

double useMatvecAlias(double a) {
  double Q[4] = {a, 1, 2, a};
  double x[2] = {a, 3};
  double out[2];
  matvecAlias(Q, x, out);
  return out[0] * out[1];
}

// CHECK: void matvecAlias_pullback(const double *Q, const double *x, double *out, double *_d_Q, double *_d_x, double *_d_out) {
// CHECK:     for (i = 0; i < 2; i++) {
// CHECK-NEXT:         out[i] = Q[2 * i] * x[0];
// CHECK-NEXT:         out[i] = out[i] + Q[2 * i + 1] * x[1];
// CHECK-NEXT:     }

// The reverse sweep of the product reads out, so the stores stay.
void scaleOut(const double* __restrict__ x, double* __restrict__ out) {
  out[0] = x[0];
  out[0] = out[0] * x[1];
}

double useScaleOut(double a) {
  double x[2] = {a, a};
  double out[1];
  scaleOut(x, out);
  return out[0];
}

// CHECK: void scaleOut_pullback(const double *__restrict x, double *__restrict out, double *_d_x, double *_d_out) {
// CHECK-NEXT:     out[0] = x[0];
// CHECK-NEXT:     double _t0 = out[0];
// CHECK-NEXT:     out[0] = out[0] * x[1];

double twice(double v) { return 2 * v; }

// A call or a side effect in a stored value keeps the stores.
void callOut(const double* __restrict__ x, double* __restrict__ out) {
  out[0] = twice(x[0]);
}

void stepOut(const double* __restrict__ x, double* __restrict__ out) {
  int k = 1;
  out[0] = x[0] * k++;
  out[1] = x[1] * k;
}

double useCallOut(double a) {
  double x[1] = {a};
  double out[1];
  callOut(x, out);
  return out[0] * a;
}

double useStepOut(double a) {
  double x[2] = {a, a};
  double out[2];
  stepOut(x, out);
  return out[0] * out[1];
}

// CHECK: void callOut_pullback(const double *__restrict x, double *__restrict out, double *_d_x, double *_d_out) {
// CHECK-NEXT:     out[0] = twice(x[0]);

// CHECK: void stepOut_pullback(const double *__restrict x, double *__restrict out, double *_d_x, double *_d_out) {
// CHECK-NEXT:     int _d_k = 0;
// CHECK-NEXT:     int k = 1;
// CHECK-NEXT:     double _t0 = k++;
// CHECK-NEXT:     out[0] = x[0] * _t0;
// CHECK-NEXT:     out[1] = x[1] * k;

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
  clad::gradient(address).execute(3, &dx);
  printf("%.2f\n", dx); // CHECK-EXEC: 6.00
  // 2 x + 1
  dx = 0;
  clad::gradient(addressTwice).execute(3, &dx);
  printf("%.2f\n", dx); // CHECK-EXEC: 7.00
  // 3 x^2
  dx = 0;
  clad::gradient(returned).execute(3, &dx);
  printf("%.2f\n", dx); // CHECK-EXEC: 27.00
  dx = 0;
  clad::gradient(indirect).execute(3, &dx);
  printf("%.2f\n", dx); // CHECK-EXEC: 6.00
  // 4 x^3
  dx = 0;
  clad::gradient(throughLocalRef).execute(3, &dx);
  printf("%.2f\n", dx); // CHECK-EXEC: 108.00
  // 15 a^2 + 15
  dx = 0;
  clad::gradient(useMatvec).execute(2, &dx);
  printf("%.2f\n", dx); // CHECK-EXEC: 75.00
  dx = 0;
  clad::gradient(useMatvecAlias).execute(2, &dx);
  printf("%.2f\n", dx); // CHECK-EXEC: 75.00
  // 2 a
  dx = 0;
  clad::gradient(useScaleOut).execute(2, &dx);
  printf("%.2f\n", dx); // CHECK-EXEC: 4.00
  // 4 a
  dx = 0;
  clad::gradient(useCallOut).execute(3, &dx);
  printf("%.2f\n", dx); // CHECK-EXEC: 12.00
  // 4 a
  dx = 0;
  clad::gradient(useStepOut).execute(3, &dx);
  printf("%.2f\n", dx); // CHECK-EXEC: 12.00
#ifdef REMARKS
  clad::gradient(noBody);
  clad::gradient(globals);
#endif
}
