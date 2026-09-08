// RUN: %cladclang %s -I%S/../../include -oReverseForw.out 2>&1 | %filecheck %s
// RUN: ./ReverseForw.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr %s -I%S/../../include -oReverseForw.out
// RUN: ./ReverseForw.out | %filecheck_exec %s

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
//CHECK-NEXT:     _tracker0.clear();
//CHECK-NEXT:     clad::ValueAndAdjoint<double *, double *> _t0 = nested_reverse_forw(&x, -6, _d_x, 0, _tracker0);
//CHECK-NEXT:     {
//CHECK-NEXT:         *_t0.adjoint += 1;
//CHECK-NEXT:         _tracker0.restore();
//CHECK-NEXT:         int _r0 = 0;
//CHECK-NEXT:         nested_pullback(&x, -6, _d_x, &_r0);
//CHECK-NEXT:         _tracker0.restore();
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
//CHECK-NEXT:     bool _cond0 = false;
//CHECK-NEXT:     {
//CHECK-NEXT:         _cond0 = s == State::should_return;
//CHECK-NEXT:         if (_cond0)
//CHECK-NEXT:             return {p, _d_p};
//CHECK-NEXT:     }
//CHECK-NEXT:     return {nullptr, nullptr};
//CHECK-NEXT: }

//CHECK: void filter_pullback(double *p, State s, double *_d_p, State *_d_s) {
//CHECK-NEXT:     bool _cond0 = false;
//CHECK-NEXT:     auto _rev0 = [&] {
//CHECK-NEXT:         if (_cond0)
//CHECK-NEXT:             ;
//CHECK-NEXT:     };
//CHECK-NEXT:     {
//CHECK-NEXT:         _cond0 = s == State::should_return;
//CHECK-NEXT:         if (_cond0) {
//CHECK-NEXT:             _rev0();
//CHECK-NEXT:             return;
//CHECK-NEXT:         }
//CHECK-NEXT:     }
//CHECK-NEXT:     _rev0();
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

// A short-circuit condition builds an `if (L) _cond = R` scaffold whose blocks
// nest; the condition rebuilt from the stored flags is placed in the enclosing
// block, so the flag stores must be hoisted instead of being declared inside
// the scaffold block.
void scale(double* C, double alpha, double beta) {
  if (alpha != 0. || (beta != 0. && C != nullptr))
    C[0] *= alpha * beta;
}

//CHECK: void scale_reverse_forw(double *C, double alpha, double beta, double *_d_C, double _d_alpha, double _d_beta, clad::restore_tracker &_tracker0) {
//CHECK-NEXT:     bool _cond0;
//CHECK-NEXT:     double _d_cond0;
//CHECK-NEXT:     _d_cond0 = 0.;
//CHECK-NEXT:     bool _cond1;
//CHECK-NEXT:     bool _cond2;
//CHECK-NEXT:     {
//CHECK-NEXT:         {
//CHECK-NEXT:             _cond1 = beta != 0.;
//CHECK-NEXT:             if (_cond1)
//CHECK-NEXT:                 _cond0 = C != nullptr;
//CHECK-NEXT:         }
//CHECK-NEXT:         _cond2 = alpha != 0. || (_cond1 && _cond0);
//CHECK-NEXT:         if (_cond2) {
//CHECK-NEXT:             _tracker0.store(C[0]);
//CHECK-NEXT:             C[0] *= alpha * beta;
//CHECK-NEXT:         }
//CHECK-NEXT:     }
//CHECK-NEXT: }

double f3(double x) {
  double arr[1] = {x};
  scale(arr, 2., 3.);
  return arr[0];
}

// For a reverse_forw call inside a loop the tracker declaration must sit at
// function scope: the matching restore() goes into the reverse sweep, which is
// a different block. A clear() at the former declaration point keeps the
// per-visit semantics.
double* mul2(double* p) {
  *p *= 2;
  return p;
}

double f4(double x) {
  double r = 0;
  for (int i = 0; i < 2; ++i)
    r += *mul2(&x);
  return r;
}

//CHECK: void f4_grad(double x, double *_d_x) {
//CHECK-NEXT:     int _d_i = 0;
//CHECK-NEXT:     int i = 0;
//CHECK-NEXT:     clad::tape<clad::restore_tracker> _tracker0 = {};
//CHECK-NEXT:     clad::tape<clad::ValueAndAdjoint<double *, double *> > _t1 = {};
//CHECK-NEXT:     double _d_r = 0.;
//CHECK-NEXT:     double r = 0;
//CHECK-NEXT:     unsigned {{int|long|long long}} _t0;
//CHECK-NEXT:     for (i = 0; i < 2; ++i) {
//CHECK-NEXT:         clad::push(_tracker0, clad::restore_tracker());
//CHECK-NEXT:         clad::push(_t1, mul2_reverse_forw(&x, _d_x, clad::back(_tracker0)));
//CHECK-NEXT:         r += *clad::back(_t1).value;
//CHECK-NEXT:     }
//CHECK-NEXT:     _d_r += 1;
//CHECK-NEXT:     for (_t0 = 2{{U|UL|ULL}}; _t0; _t0--) {
//CHECK-NEXT:         *clad::back(_t1).adjoint += _d_r;
//CHECK-NEXT:         clad::back(_tracker0).restore();
//CHECK-NEXT:         mul2_pullback(&x, _d_x);
//CHECK-NEXT:         clad::back(_tracker0).restore();
//CHECK-NEXT:         clad::pop(_tracker0);
//CHECK-NEXT:         clad::pop(_t1);
//CHECK-NEXT:     }
//CHECK-NEXT: }

// For a reverse_forw call inside a branch the ValueAndAdjoint store must also
// be hoisted to function scope: the `.adjoint` use goes into the reverse
// sweep, which is a different block than the forward `if`.
double f5(double x, int flag) {
  double r = x;
  if (flag)
    r += *mul2(&x);
  return r;
}

//CHECK: void f5_grad_0(double x, int flag, double *_d_x) {
//CHECK-NEXT:     int _d_flag = 0;
//CHECK-NEXT:     bool _cond0;
//CHECK-NEXT:     clad::restore_tracker _tracker0 = {};
//CHECK-NEXT:     clad::ValueAndAdjoint<double *, double *> _t0;
//CHECK-NEXT:     double _d_r = 0.;
//CHECK-NEXT:     double r = x;
//CHECK-NEXT:     {
//CHECK-NEXT:         _cond0 = flag;
//CHECK-NEXT:         if (_cond0) {
//CHECK-NEXT:             _tracker0.clear();
//CHECK-NEXT:             _t0 = mul2_reverse_forw(&x, _d_x, _tracker0);
//CHECK-NEXT:             r += *_t0.value;
//CHECK-NEXT:         }
//CHECK-NEXT:     }
//CHECK-NEXT:     _d_r += 1;
//CHECK-NEXT:     if (_cond0) {
//CHECK-NEXT:         *_t0.adjoint += _d_r;
//CHECK-NEXT:         _tracker0.restore();
//CHECK-NEXT:         mul2_pullback(&x, _d_x);
//CHECK-NEXT:         _tracker0.restore();
//CHECK-NEXT:     }
//CHECK-NEXT:     *_d_x += _d_r;
//CHECK-NEXT: }

// A reference-typed ValueAndAdjoint cannot be hoisted whole (a reference
// member cannot be declared unset and assigned afterwards); the store stays
// block-local and pointers to the referents are hoisted instead, so the
// reverse sweep stays in scope.
double& amplify(double& x) {
  x *= 2;
  return x;
}

double f6(double x, int flag) {
  double r = x;
  if (flag)
    r += amplify(x);
  return r;
}

//CHECK: void f6_grad_0(double x, int flag, double *_d_x) {
//CHECK-NEXT:     int _d_flag = 0;
//CHECK-NEXT:     bool _cond0;
//CHECK-NEXT:     double _t0;
//CHECK-NEXT:     double *_t2;
//CHECK-NEXT:     double *_t3;
//CHECK-NEXT:     double _d_r = 0.;
//CHECK-NEXT:     double r = x;
//CHECK-NEXT:     {
//CHECK-NEXT:         _cond0 = flag;
//CHECK-NEXT:         if (_cond0) {
//CHECK-NEXT:             _t0 = x;
//CHECK-NEXT:             clad::ValueAndAdjoint<double &, double &> _t1 = amplify_reverse_forw(x, *_d_x);
//CHECK-NEXT:             _t2 = &_t1.value;
//CHECK-NEXT:             _t3 = &_t1.adjoint;
//CHECK-NEXT:             r += *_t2;
//CHECK-NEXT:         }
//CHECK-NEXT:     }
//CHECK-NEXT:     _d_r += 1;
//CHECK-NEXT:     if (_cond0) {
//CHECK-NEXT:         *_t3 += _d_r;
//CHECK-NEXT:         x = _t0;
//CHECK-NEXT:         amplify_pullback(x, _d_x);
//CHECK-NEXT:     }
//CHECK-NEXT:     *_d_x += _d_r;
//CHECK-NEXT: }

int main() {
  double dx = 0;
  INIT_GRADIENT(f1);
  TEST_GRADIENT(f1, /*numOfDerivativeArgs=*/1, -9, &dx); // CHECK-EXEC: -1.00

  dx = 0;
  INIT_GRADIENT(f2);
  TEST_GRADIENT(f2, /*numOfDerivativeArgs=*/1, 3, &dx); // CHECK-EXEC: 1.00

  dx = 0;
  INIT_GRADIENT(f3);
  TEST_GRADIENT(f3, /*numOfDerivativeArgs=*/1, 1, &dx); // CHECK-EXEC: 6.00

  dx = 0;
  INIT_GRADIENT(f4);
  TEST_GRADIENT(f4, /*numOfDerivativeArgs=*/1, 1, &dx); // CHECK-EXEC: 6.00

  dx = 0;
  INIT_GRADIENT(f5, "x");
  TEST_GRADIENT(f5, /*numOfDerivativeArgs=*/1, 1, 1, &dx); // CHECK-EXEC: 3.00

  dx = 0;
  INIT_GRADIENT(f6, "x");
  TEST_GRADIENT(f6, /*numOfDerivativeArgs=*/1, 1, 1, &dx); // CHECK-EXEC: 3.00
  dx = 0;
  TEST_GRADIENT(f6, /*numOfDerivativeArgs=*/1, 1, 0, &dx); // CHECK-EXEC: 1.00
}
