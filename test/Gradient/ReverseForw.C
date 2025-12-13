// RUN: %cladclang %s -I%S/../../include -oReverseForw.out 2>&1 | %filecheck %s
// RUN: ./ReverseForw.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr %s -I%S/../../include -oReverseForw.out
// RUN: ./ReverseForw.out | %filecheck_exec %s
// XFAIL: valgrind

#include "clad/Differentiator/Differentiator.h"
#include "../TestUtils.h"

double* nested(double* p, int n) {
    int i;
    if (n > 0)
        i = 1;
    else
        i = -1;
    *p *= i;
    return p;
}

//CHECK: clad::ValueAndAdjoint<double *, double *> nested_reverse_forw(double *p, int n, double *_d_p, int _d_n, clad::restore_tracker &_tracker0) {
//CHECK-NEXT:     int _d_i = 0;
//CHECK-NEXT:     int i;
//CHECK-NEXT:     {
//CHECK-NEXT:         bool _cond0 = n > 0;
//CHECK-NEXT:         if (_cond0)
//CHECK-NEXT:             i = 1;
//CHECK-NEXT:         else
//CHECK-NEXT:             i = -1;
//CHECK-NEXT:     }
//CHECK-NEXT:     _tracker0.store(*p);
//CHECK-NEXT:     *p *= i;
//CHECK-NEXT:     return {p, _d_p};
//CHECK-NEXT: }

//CHECK: void nested_pullback(double *p, int n, double *_d_p, int *_d_n) {
//CHECK-NEXT:     bool _cond0;
//CHECK-NEXT:     int _d_i = 0;
//CHECK-NEXT:     int i;
//CHECK-NEXT:     {
//CHECK-NEXT:         _cond0 = n > 0;
//CHECK-NEXT:         if (_cond0)
//CHECK-NEXT:             i = 1;
//CHECK-NEXT:         else
//CHECK-NEXT:             i = -1;
//CHECK-NEXT:     }
//CHECK-NEXT:     double _t0 = *p;
//CHECK-NEXT:     *p *= i;
//CHECK-NEXT:     {
//CHECK-NEXT:         *p = _t0;
//CHECK-NEXT:         double _r_d0 = *_d_p;
//CHECK-NEXT:         *_d_p = 0.;
//CHECK-NEXT:         *_d_p += _r_d0 * i;
//CHECK-NEXT:         _d_i += *p * _r_d0;
//CHECK-NEXT:     }
//CHECK-NEXT:     if (_cond0)
//CHECK-NEXT:         _d_i = 0;
//CHECK-NEXT:     else
//CHECK-NEXT:         _d_i = 0;
//CHECK-NEXT: }

double f1(double x) {
    return *nested(&x, -6);
}

//CHECK: void f1_grad(double x, double *_d_x) {
//CHECK-NEXT:     clad::restore_tracker _tracker0 = {};
//CHECK-NEXT:     clad::ValueAndAdjoint<double *, double *> _t0 = nested_reverse_forw(&x, -6, _d_x, 0, _tracker0);
//CHECK-NEXT:     {
//CHECK-NEXT:         *_t0.adjoint += 1;
//CHECK-NEXT:         _tracker0.restore();
//CHECK-NEXT:         int _r0 = 0;
//CHECK-NEXT:         nested_pullback(&x, -6, _d_x, &_r0);
//CHECK-NEXT:     }
//CHECK-NEXT: }

enum State {
  should_return,
  no_return
};

double* filter(double* p, State s) {
    if (s == State::should_return)
        return p;
    return nullptr;
}

//CHECK: clad::ValueAndAdjoint<double *, double *> filter_reverse_forw(double *p, State s, double *_d_p, State _d_s) {
//CHECK-NEXT:     {
//CHECK-NEXT:         bool _cond0 = s == State::should_return;
//CHECK-NEXT:         if (_cond0)
//CHECK-NEXT:             return {p, _d_p};
//CHECK-NEXT:     }
//CHECK-NEXT:     return {nullptr, nullptr};
//CHECK-NEXT: }

//CHECK: void filter_pullback(double *p, State s, double *_d_p, State *_d_s) {
//CHECK-NEXT:     bool _cond0;
//CHECK-NEXT:     {
//CHECK-NEXT:         _cond0 = s == State::should_return;
//CHECK-NEXT:         if (_cond0)
//CHECK-NEXT:             goto _label0;
//CHECK-NEXT:     }
//CHECK-NEXT:     if (_cond0)
//CHECK-NEXT:       _label0:
//CHECK-NEXT:         ;
//CHECK-NEXT: }

double f2(double x) {
    return *filter(&x, State::should_return);
}

//CHECK: void f2_grad(double x, double *_d_x) {
//CHECK-NEXT:     clad::ValueAndAdjoint<double *, double *> _t0 = filter_reverse_forw(&x, State::should_return, _d_x, static_cast<State>(0U));
//CHECK-NEXT:     {
//CHECK-NEXT:         *_t0.adjoint += 1;
//CHECK-NEXT:         State _r0 = static_cast<State>(0U);
//CHECK-NEXT:         filter_pullback(&x, State::should_return, _d_x, &_r0);
//CHECK-NEXT:     }
//CHECK-NEXT: }

int main() {
  double dx = 0;
  INIT_GRADIENT(f1);
  TEST_GRADIENT(f1, /*numOfDerivativeArgs=*/1, -9, &dx); // CHECK-EXEC: -1.00

  dx = 0;
  INIT_GRADIENT(f2);
  TEST_GRADIENT(f2, /*numOfDerivativeArgs=*/1, 3, &dx); // CHECK-EXEC: 1.00
}
