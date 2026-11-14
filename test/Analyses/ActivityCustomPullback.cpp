// RUN: %cladclang %s -I%S/../../include -oActivityCustomPullback.out 2>&1 \
// RUN:   | %filecheck %s
// RUN: ./ActivityCustomPullback.out | %filecheck_exec %s
// RUN: %cladclang %s -I%S/../../include -fsyntax-only -Xclang -verify \
// RUN:   -DMISMATCH

// With activity analysis, a pullback of h is planned without e for the call
// that passes b, which nothing varies. The call to inner in it still uses the
// user's pullback, which takes every parameter: it is handed a zero adjoint
// for e. Whichever call comes first, every call uses the user's derivative.

#include "clad/Differentiator/Differentiator.h"

#include <cstdio>

struct E {
  double k;
};

double inner(const E& e, double x) { return x * e.k; }

namespace clad::custom_derivatives {
void inner_pullback(const E& e, double x, double d, E* de, double* dx) {
  *dx += d * e.k;
  de->k += d * x;
}
} // namespace clad::custom_derivatives

double h(const E& e, double x) { return inner(e, x); }

double fullFirst(double x) {
  E a{x};
  E b{3};
  return h(a, x) + h(b, x);
}

double reducedFirst(double x) {
  E a{x};
  E b{3};
  return h(b, x) + h(a, x);
}

// CHECK-NOT: void inner_pullback(
// CHECK: void h_pullback(const E &e, double x, double _d_y, double *_d_x) {
// CHECK: E _r{{[0-9]+}} = {0.};
// CHECK-NEXT: clad::custom_derivatives::inner_pullback(e, x, _d_y, &_r{{[0-9]+}}, &_r{{[0-9]+}});
// CHECK-NOT: void inner_pullback(

// The same for a call whose value nothing reads, and for a method.
double discard(const E& e, double x) {
  inner(e, x);
  return x;
}

struct Op {
  double s;
  double apply(const E& e, double x) const { return s * x * e.k; }
};

namespace clad::custom_derivatives::class_functions {
void apply_pullback(const Op* op, const E& e, double x, double d, Op* dop,
                    E* de, double* dx) {
  dop->s += d * x * e.k;
  *dx += d * op->s * e.k;
  de->k += d * op->s * x;
}
} // namespace clad::custom_derivatives::class_functions

double method(const E& e, double x) {
  Op op{2};
  return op.apply(e, x);
}

double others(double x) {
  E a{x};
  E b{3};
  return discard(b, x) + method(b, x) + discard(a, x) + method(a, x);
}

// CHECK-NOT: void apply_pullback(
// CHECK: void method_pullback(const E &e, double x, double _d_y, double *_d_x) {
// CHECK: E _r{{[0-9]+}} = {0.};
// CHECK-NEXT: clad::custom_derivatives::class_functions::apply_pullback(&op, e, x, _d_y, &_d_op, &_r{{[0-9]+}}, &_r{{[0-9]+}});
// CHECK-NOT: void apply_pullback(

#ifdef MISMATCH
// A user's pullback with the wrong signature is reported once, against the
// signature the user has to write, whichever call comes first.
double wrong(const E& e, double x) { return x * e.k; }

namespace clad::custom_derivatives {
void wrong_pullback(const E& e, double d, E* de) { de->k += d; } // expected-note {{candidate 'wrong_pullback' has different number of parameters}}
} // namespace clad::custom_derivatives

double g(const E& e, double x) { return wrong(e, x); } // expected-error {{user-defined derivative for 'wrong' was provided but not used; expected signature 'void (const E &, double, double, E *, double *)' does not match}}

double mismatch(double x) {
  E a{x};
  E b{3};
  return g(b, x) + g(a, x);
}

void reportMismatch() { clad::gradient<clad::opts::enable_va>(mismatch); }

// The same for a pullback that overrides clad's.
double wrongOver(const E& e, double x) { return x * e.k; }

namespace clad::custom_derivatives::overrides {
void wrongOver_pullback(const E& e, double d, E* de) { de->k += d; } // expected-note {{candidate 'wrongOver_pullback' has different number of parameters}}
} // namespace clad::custom_derivatives::overrides

double go(const E& e, double x) { return wrongOver(e, x); } // expected-error {{user-defined derivative for 'wrongOver' was provided but not used; expected signature 'void (const E &, double, double, E *, double *)' does not match}}

double mismatchOver(double x) {
  E a{x};
  E b{3};
  return go(b, x) + go(a, x);
}

void reportMismatchOver() {
  clad::gradient<clad::opts::enable_va>(mismatchOver);
}
#endif

int main() {
  double dx = 0;
  clad::gradient<clad::opts::enable_va>(fullFirst).execute(2, &dx);
  printf("%.2f\n", dx); // CHECK-EXEC: 7.00
  dx = 0;
  clad::gradient<clad::opts::enable_va>(reducedFirst).execute(2, &dx);
  printf("%.2f\n", dx); // CHECK-EXEC: 7.00
  dx = 0;
  clad::gradient<clad::opts::enable_va>(others).execute(2, &dx);
  printf("%.2f\n", dx); // CHECK-EXEC: 16.00
}
