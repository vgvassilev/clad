// RUN: %cladclang -std=c++17 -I%S/../../include %s -o %t 2>&1 | %filecheck %s
// RUN: %t | %filecheck_exec %s

// Clad cannot size a tangent buffer for a `const double*` parameter, so a
// pushforward is handed a null tangent for it and null checks the reads through
// it. Pointer arithmetic used to lose that encoding: `_d_xlArr + 1` is 0x8
// rather than null, so the check sailed through and the derivative dereferenced
// a small integer. Reads written on the arithmetic itself were not checked at
// all, only bare declaration references were.
//
// These are the shapes RooFit's code generation emits for a multi-channel
// likelihood, so every such hessian crashed at run time.

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

// Initializer: the derived pointer outlives the expression that built it.
double indexed(double* params, double const* xlArr) {
  double const* q = xlArr + 1;
  return params[0] * params[0] * q[0] + params[0] * params[1] * xlArr[0];
}
double stored_ptr(double* params, double const* xlArr) {
  return indexed(params, xlArr);
}

// CHECK: clad::ValueAndPushforward<double, double> indexed_pushforward(double *params, const double *xlArr, double *_d_params, const double *_d_xlArr) {
// CHECK-NEXT: const double *_d_q = (_d_xlArr ? _d_xlArr + 1 : nullptr);
// CHECK-NEXT: const double *q = xlArr + 1;

// A read on the arithmetic itself is guarded on the root, so the check does
// not repeat the arithmetic.
double deref(double* params, double const* xlArr) {
  unsigned int i = 1;
  return params[0] * params[0] * *(xlArr + i);
}
double deref_arith(double* params, double const* xlArr) {
  return deref(params, xlArr);
}

// CHECK: clad::ValueAndPushforward<double, double> deref_pushforward(double *params, const double *xlArr, double *_d_params, const double *_d_xlArr) {
// CHECK: _d_xlArr ? *(_d_xlArr + i) : 0.

// Assignment, compound assignment and increment. The last two step the tangent
// in place, so the step itself is what gets skipped while it is null.
double reassigned(double* params, double const* xlArr) {
  double const* q = xlArr;
  q = xlArr + 1;
  return params[0] * params[0] * q[0];
}
double assign_ptr(double* params, double const* xlArr) {
  return reassigned(params, xlArr);
}

// CHECK: clad::ValueAndPushforward<double, double> reassigned_pushforward(double *params, const double *xlArr, double *_d_params, const double *_d_xlArr) {
// CHECK: _d_q = (_d_xlArr ? _d_xlArr + 1 : nullptr);
// CHECK-NEXT: q = xlArr + 1;

double advanced(double* params, double const* xlArr) {
  double const* q = xlArr;
  q += 1;
  return params[0] * params[0] * q[0];
}
double compound_ptr(double* params, double const* xlArr) {
  return advanced(params, xlArr);
}

// CHECK: clad::ValueAndPushforward<double, double> advanced_pushforward(double *params, const double *xlArr, double *_d_params, const double *_d_xlArr) {
// CHECK: (_d_q ? _d_q += 1 : nullptr);
// CHECK-NEXT: q += 1;

double stepped(double* params, double const* xlArr) {
  double const* q = xlArr;
  q++;
  return params[0] * params[0] * q[0];
}
double incr_ptr(double* params, double const* xlArr) {
  return stepped(params, xlArr);
}

// CHECK: clad::ValueAndPushforward<double, double> stepped_pushforward(double *params, const double *xlArr, double *_d_params, const double *_d_xlArr) {
// CHECK: (_d_q ? _d_q++ : nullptr);
// CHECK-NEXT: q++;

// A derived pointer has to reach a further callee null, so that the callee's
// own check answers truthfully.
double leaf(double const* v) { return v[0]; }
double forwarded(double* params, double const* xlArr) {
  double const* q = xlArr + 1;
  return params[0] * params[0] * leaf(q);
}
double pass_ptr(double* params, double const* xlArr) {
  return forwarded(params, xlArr);
}

// Passing the arithmetic itself is what reaches the call-argument guard; a
// pointer bound to a variable first was already nulled by its initializer.
double forwarded_arith(double* params, double const* xlArr) {
  return params[0] * params[0] * leaf(xlArr + 1);
}
double pass_arith(double* params, double const* xlArr) {
  return forwarded_arith(params, xlArr);
}

// CHECK: clad::ValueAndPushforward<double, double> forwarded_arith_pushforward(double *params, const double *xlArr, double *_d_params, const double *_d_xlArr) {
// CHECK: leaf_pushforward(xlArr + 1, (_d_xlArr ? _d_xlArr + 1 : nullptr));

// An assignment is not a step, so the walk ends there instead of reaching
// `xlArr` through it. The tangent it hands on is already null when `_d_xlArr`
// is, and the outer assignment has to pass that on rather than test `_d_p`,
// which the same statement is still writing.
double chained(double* params, double const* xlArr) {
  double const* p = xlArr;
  double const* q = xlArr;
  q = p = xlArr + 1;
  return params[0] * params[0] * q[0];
}
double chain_ptr(double* params, double const* xlArr) {
  return chained(params, xlArr);
}

// CHECK: clad::ValueAndPushforward<double, double> chained_pushforward(double *params, const double *xlArr, double *_d_params, const double *_d_xlArr) {
// CHECK: _d_q = _d_p = (_d_xlArr ? _d_xlArr + 1 : nullptr);
// CHECK-NEXT: q = p = xlArr + 1;

// A tangent that steps a pointer is both tested and read through, so it has to
// be evaluated once. Guarding the step in place stepped it twice, which reads
// past the end of the tangent buffer.
double stepped_read(double* x) {
  double* q; // no initializer, so the planner marks its tangent maybe-null
  q = x;     // ... while at run time it is the real tangent of `x`
  return *(q += 1);
}
double step_read(double* x) { return stepped_read(x); }

double stepped_read_post(double* x) {
  double* q;
  q = x;
  return *(q++);
}
double step_read_post(double* x) { return stepped_read_post(x); }

// CHECK: clad::ValueAndPushforward<double, double> stepped_read_pushforward(double *x, double *_d_x) {
// CHECK: double *_t0 = (_d_q ? _d_q += 1 : nullptr);
// CHECK-NEXT: return {*(q += 1), (_t0 ? *_t0 : 0.)};

int main() {
  double params[2] = {2., 3.};
  double xlArr[2] = {5., 7.};

  double m[4] = {};
  clad::hessian(stored_ptr, "params[0:1]").execute(params, xlArr, m);
  printf("stored_ptr %.2f %.2f %.2f %.2f\n", m[0], m[1], m[2], m[3]);
  // CHECK-EXEC: stored_ptr 14.00 5.00 5.00 0.00

  double m2[1] = {};
  clad::hessian(deref_arith, "params[0]").execute(params, xlArr, m2);
  printf("deref_arith %.2f\n", m2[0]);
  // CHECK-EXEC: deref_arith 14.00

  double m3[1] = {};
  clad::hessian(pass_ptr, "params[0]").execute(params, xlArr, m3);
  printf("pass_ptr %.2f\n", m3[0]);
  // CHECK-EXEC: pass_ptr 14.00

  double m4[1] = {};
  clad::hessian(assign_ptr, "params[0]").execute(params, xlArr, m4);
  printf("assign_ptr %.2f\n", m4[0]);
  // CHECK-EXEC: assign_ptr 14.00

  double m5[1] = {};
  clad::hessian(compound_ptr, "params[0]").execute(params, xlArr, m5);
  printf("compound_ptr %.2f\n", m5[0]);
  // CHECK-EXEC: compound_ptr 14.00

  double m6[1] = {};
  clad::hessian(incr_ptr, "params[0]").execute(params, xlArr, m6);
  printf("incr_ptr %.2f\n", m6[0]);
  // CHECK-EXEC: incr_ptr 14.00

  double m7[1] = {};
  clad::hessian(pass_arith, "params[0]").execute(params, xlArr, m7);
  printf("pass_arith %.2f\n", m7[0]);
  // CHECK-EXEC: pass_arith 14.00

  double m8[1] = {};
  clad::hessian(chain_ptr, "params[0]").execute(params, xlArr, m8);
  printf("chain_ptr %.2f\n", m8[0]);
  // CHECK-EXEC: chain_ptr 14.00

  double v[3] = {2., 3., 5.};
  printf("step_read %.2f\n",
         clad::differentiate(step_read, "x[1]").execute(v));
  // CHECK-EXEC: step_read 1.00
  printf("step_read_post %.2f\n",
         clad::differentiate(step_read_post, "x[0]").execute(v));
  // CHECK-EXEC: step_read_post 1.00
}
