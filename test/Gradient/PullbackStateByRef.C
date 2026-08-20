// RUN: %cladclang %s -I%S/../../include -oPullbackStateByRef.out 2>&1 | %filecheck %s
// RUN: ./PullbackStateByRef.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr %s -I%S/../../include -oPullbackStateByRef.out
// RUN: ./PullbackStateByRef.out | %filecheck_exec %s

// A pullback may take its clad::pullback_state carrier by reference as well as
// by value. By value cannot express a payload that owns storage: a payload
// holding a clad::tape<T> has no copy constructor, so a by-value pullback does
// not compile at all. Recording values in the reverse_forw and popping them in
// the pullback -- the shape a replay-free pullback needs -- therefore requires
// the by-reference form.

#include "clad/Differentiator/Differentiator.h"
#include "../TestUtils.h"

struct Recorded {
  clad::tape<double> values; // non-copyable: forces the by-reference form
};

double* rec(double* p) { return p; }

namespace clad::custom_derivatives {
clad::ValueAndAdjoint<double*, double*>
rec_reverse_forw(double* p, double* d_p, clad::pullback_state<Recorded>& state) {
  // Forward sweep records; the pullback consumes without re-running anything.
  clad::push(state.data.values, 3.0);
  clad::push(state.data.values, 4.0);
  return {p, d_p};
}
void rec_pullback(double* p, double* d_p, clad::pullback_state<Recorded>& state) {
  *d_p += clad::pop(state.data.values); // 4
  *d_p += clad::pop(state.data.values); // 3
}
} // namespace clad::custom_derivatives

double f(double x) { return *rec(&x); }

// The carrier is declared once and passed to both calls as an lvalue, so the
// same object serves the reverse_forw's out-param and the pullback's reference.
//CHECK: void f_grad(double x, double *_d_x) {
//CHECK-NEXT:     clad::pullback_state<Recorded> _state0 = {{.*}};
//CHECK-NEXT:     clad::ValueAndAdjoint<double *, double *> _t0 = clad::custom_derivatives::rec_reverse_forw(&x, _d_x, _state0);
//CHECK-NEXT:     {
//CHECK-NEXT:         *_t0.adjoint += 1;
//CHECK-NEXT:         clad::custom_derivatives::rec_pullback(&x, _d_x, _state0);
//CHECK-NEXT:     }
//CHECK-NEXT: }

// The by-value convention still matches, so existing custom derivatives keep
// working unchanged.
double* val(double* p) { return p; }

namespace clad::custom_derivatives {
clad::ValueAndAdjoint<double*, double*>
val_reverse_forw(double* p, double* d_p, clad::pullback_state<double>& state) {
  state.data = 5;
  return {p, d_p};
}
void val_pullback(double* p, double* d_p, clad::pullback_state<double> state) {
  *d_p += state.data;
}
} // namespace clad::custom_derivatives

double fv(double x) { return *val(&x); }

//CHECK: void fv_grad(double x, double *_d_x) {
//CHECK-NEXT:     clad::pullback_state<double> _state0 = {0.};
//CHECK-NEXT:     clad::ValueAndAdjoint<double *, double *> _t0 = clad::custom_derivatives::val_reverse_forw(&x, _d_x, _state0);
//CHECK-NEXT:     {
//CHECK-NEXT:         *_t0.adjoint += 1;
//CHECK-NEXT:         clad::custom_derivatives::val_pullback(&x, _d_x, _state0);
//CHECK-NEXT:     }
//CHECK-NEXT: }

int main() {
  auto df = clad::gradient(f);
  double dx = 0;
  df.execute(2.0, &dx);
  // seed 1, plus the two recorded values popped in reverse: 1 + 4 + 3
  printf("%.2f\n", dx); // CHECK-EXEC: 8.00

  auto dfv = clad::gradient(fv);
  double dxv = 0;
  dfv.execute(2.0, &dxv);
  printf("%.2f\n", dxv); // CHECK-EXEC: 6.00
}
