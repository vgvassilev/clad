// RUN: %cladclang %s -I%S/../../include -oPullbackState.out 2>&1 | %filecheck %s
// RUN: ./PullbackState.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr %s -I%S/../../include -oPullbackState.out
// RUN: ./PullbackState.out | %filecheck_exec %s

// clad::pullback_state threading: a custom reverse_forw fills a trailing
// pullback_state<Payload>& out-param in the forward sweep, and clad hands the
// same carrier to the matching pullback (by value) in the reverse sweep. The
// pullback here contributes state.data to the adjoint, so the gradient is
// 1 + data if the carrier threads and 1 if it does not -- a direct check that
// the value crosses from forward to reverse without a shared or global stash.

#include "clad/Differentiator/Differentiator.h"
#include "../TestUtils.h"

// Pointer return makes clad route the call through a reverse_forw / pullback
// pair (isMemoryType(return)), the seam pullback_state rides on.
double* g(double* p) { return p; }

namespace clad::custom_derivatives {
clad::ValueAndAdjoint<double*, double*>
g_reverse_forw(double* p, double* d_p, clad::pullback_state<double>& state) {
  state.data = 10; // stash a sentinel only the pullback can observe
  return {p, d_p};
}
void g_pullback(double* p, double* d_p, clad::pullback_state<double> state) {
  *d_p += state.data; // adds the threaded sentinel on top of the seed
}
} // namespace clad::custom_derivatives

double f(double x) { return *g(&x); }

//CHECK: void f_grad(double x, double *_d_x) {
//CHECK-NEXT:     clad::pullback_state<double> _state0 = {0.};
//CHECK-NEXT:     clad::ValueAndAdjoint<double *, double *> _t0 = clad::custom_derivatives::g_reverse_forw(&x, _d_x, _state0);
//CHECK-NEXT:     {
//CHECK-NEXT:         *_t0.adjoint += 1;
//CHECK-NEXT:         clad::custom_derivatives::g_pullback(&x, _d_x, _state0);
//CHECK-NEXT:     }
//CHECK-NEXT: }

// A void, in-place op whose primal clad could call directly. Because its
// reverse_forw carries pullback_state, clad must still route through the
// reverse_forw and never elide it -- otherwise the carrier is left empty and
// the gradient is silently 1 instead of 8.
void vop(double* p) {}

namespace clad::custom_derivatives {
void vop_reverse_forw(double* p, double* d_p, clad::pullback_state<double>& state) {
  state.data = 7;
}
void vop_pullback(double* p, double* d_p, clad::pullback_state<double> state) {
  *d_p += state.data;
}
} // namespace clad::custom_derivatives

double fv(double x) {
  double v = x;
  vop(&v);
  return v;
}

//CHECK: void fv_grad(double x, double *_d_x) {
//CHECK-NEXT:     double _d_v = 0.;
//CHECK-NEXT:     double v = x;
//CHECK-NEXT:     clad::pullback_state<double> _state0 = {0.};
//CHECK-NEXT:     clad::custom_derivatives::vop_reverse_forw(&v, &_d_v, _state0);
//CHECK-NEXT:     _d_v += 1;
//CHECK-NEXT:     clad::custom_derivatives::vop_pullback(&v, &_d_v, _state0);
//CHECK-NEXT:     *_d_x += _d_v;
//CHECK-NEXT: }

// Reentrancy: two state-carrying calls in one gradient must each get their own
// carrier -- the multi-call case the static stack this replaces got wrong. The
// carriers stash different sentinels onto different inputs, so a shared or
// crossed stash would swap the two gradients.
void op3(double* p) {}

namespace clad::custom_derivatives {
void op3_reverse_forw(double* p, double* d_p, clad::pullback_state<double>& state) {
  state.data = 3;
}
void op3_pullback(double* p, double* d_p, clad::pullback_state<double> state) {
  *d_p += state.data;
}
} // namespace clad::custom_derivatives

double f_two(double x, double y) {
  double a = x;
  double b = y;
  vop(&a); // its carrier stashes 7 -> d/dx = 1 + 7 = 8
  op3(&b); // a distinct carrier stashes 3 -> d/dy = 1 + 3 = 4
  return a + b;
}

// Templated custom derivatives (the thrust sort_by_key shape): the state param
// is declared on a FunctionTemplateDecl, which the scheduler must see through
// when it extends the expected signature. If it does not, neither derivative
// matches, clad differentiates the empty primal instead, and the gradient is
// silently 1 instead of 6.
template <typename T> void tvop(T* p) {}

namespace clad::custom_derivatives {
template <typename T>
void tvop_reverse_forw(T* p, T* d_p, clad::pullback_state<double>& state) {
  state.data = 5;
}
template <typename T>
void tvop_pullback(T* p, T* d_p, clad::pullback_state<double> state) {
  *d_p += state.data;
}
} // namespace clad::custom_derivatives

double f_tmpl(double x) {
  double v = x;
  tvop(&v);
  return v;
}

//CHECK: void f_tmpl_grad(double x, double *_d_x) {
//CHECK-NEXT:     double _d_v = 0.;
//CHECK-NEXT:     double v = x;
//CHECK-NEXT:     clad::pullback_state<double> _state0 = {0.};
//CHECK-NEXT:     clad::custom_derivatives::tvop_reverse_forw(&v, &_d_v, _state0);
//CHECK-NEXT:     _d_v += 1;
//CHECK-NEXT:     clad::custom_derivatives::tvop_pullback(&v, &_d_v, _state0);
//CHECK-NEXT:     *_d_x += _d_v;
//CHECK-NEXT: }

int main() {
  double dx = 0;
  INIT_GRADIENT(f);
  TEST_GRADIENT(f, /*numOfDerivativeArgs=*/1, 3, &dx); // CHECK-EXEC: 11.00

  dx = 0;
  INIT_GRADIENT(fv);
  TEST_GRADIENT(fv, /*numOfDerivativeArgs=*/1, 3, &dx); // CHECK-EXEC: 8.00

  dx = 0;
  double dy = 0;
  INIT_GRADIENT(f_two);
  TEST_GRADIENT(f_two, /*numOfDerivativeArgs=*/2, 1, 1, &dx, &dy); // CHECK-EXEC: {8.00, 4.00}

  dx = 0;
  INIT_GRADIENT(f_tmpl);
  TEST_GRADIENT(f_tmpl, /*numOfDerivativeArgs=*/1, 3, &dx); // CHECK-EXEC: 6.00
}
