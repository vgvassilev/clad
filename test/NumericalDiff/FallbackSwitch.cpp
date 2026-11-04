// The switches that decide whether clad falls back to numerical
// differentiation: -f[no-]numerical-diff-fallback for the translation unit,
// the last one deciding, and clad::opts::enable_numerical_diff_fallback /
// disable_numerical_diff_fallback for one request. std::tgamma and
// std::copysign have no body clad can see and no custom derivative, so a
// derivative through them is the fallback or zero.
//
// %cladclang passes -fno-numerical-diff-fallback.
// RUN: %cladclang %s -I%S/../../include -oFallbackSwitch.out -Xclang -verify \
// RUN:   -DTU_OFF 2>&1 | %filecheck --check-prefix=CHECK-OFF %s
// RUN: ./FallbackSwitch.out | %filecheck_exec --check-prefix=CHECK-EXEC-OFF %s
//
// RUN: %cladclang %s -I%S/../../include -oFallbackSwitch.out -Xclang -verify \
// RUN:   -Xclang -plugin-arg-clad -Xclang -fnumerical-diff-fallback -DTU_ON 2>&1
// RUN: ./FallbackSwitch.out | %filecheck_exec --check-prefix=CHECK-EXEC-ON %s
//
// Both halves of the pair at once are a contradiction, and so is asking for
// one derivative both ways: clad names it after the function and the mode.
// RUN: %cladclang -fsyntax-only -Xclang -verify %s -I%S/../../include -DBOTH
// RUN: %cladclang -fsyntax-only -Xclang -verify %s -I%S/../../include \
// RUN:   -DCONFLICT

// The translation unit's setting reaches headers as a macro.
#if (defined(TU_OFF) && !defined(CLAD_NUMERICAL_DIFF_FALLBACK_OFF)) ||        \
    (defined(TU_ON) && defined(CLAD_NUMERICAL_DIFF_FALLBACK_OFF))
#error "CLAD_NUMERICAL_DIFF_FALLBACK_OFF has to mean -fno-numerical-diff-fallback"
#endif

#include "clad/Differentiator/Differentiator.h"

#include <cmath>
#include <cstdio>

#if defined(BOTH) || defined(CONFLICT)
// Nothing here falls back, so the two derivatives would be the same.
double f(double x) { return x * x; }

#ifdef CONFLICT
// Each request still plans tgamma its own way; only g's derivative is shared.
// expected-warning@+3 2 {{attempted differentiation of function 'tgamma'}}
// expected-note@+2 {{falling back to numerical differentiation for 'tgamma'}}
// expected-note@+1 {{the numerical-diff fallback is off for this request; considering 'tgamma' as 0}}
double g(double x) { return std::tgamma(x); }

#if __clang_major__ >= 17
// A fallback reached inside a lambda counts for the function the lambda is
// in: its derivative builds the lambda's.
double inLambda(double x) {
  // expected-warning@+3 2 {{attempted differentiation of function 'tgamma'}}
  // expected-note@+2 {{falling back to numerical differentiation for 'tgamma'}}
  // expected-note@+1 {{the numerical-diff fallback is off for this request; considering 'tgamma' as 0}}
  auto l = [](double t) { return std::tgamma(t); };
  return l(x);
}
#endif

// The other way round: built off first, it is still the derivative the next
// request would get. (tgamma was reported for g.)
double offFirst(double x) { return std::tgamma(x); }

// Refused for its argument's effect in a conditional, the call has no
// derivative either way: the two requests share one. (Built with the fallback
// off, it is not warned about.)
double noFallback(double x) {
  double y = x;
  return x > 0 ? std::tgamma(y++) : 0.;
}

// A call of several arguments falls back in forward mode too.
// expected-warning@+3 2 {{attempted differentiation of function 'copysign'}}
// expected-note@+2 {{falling back to numerical differentiation for 'copysign'}}
// expected-note@+1 {{the numerical-diff fallback is off for this request; considering 'copysign' as 0}}
double twoArgs(double x, double y) { return std::copysign(x, y) * x; }
#endif

void request() {
#ifdef BOTH
  // expected-error@+2 {{both clad::opts::enable_numerical_diff_fallback and clad::opts::disable_numerical_diff_fallback are specified}}
  clad::differentiate<clad::opts::enable_numerical_diff_fallback |
                      clad::opts::disable_numerical_diff_fallback>(f, "x");
#else
  clad::differentiate<clad::opts::enable_numerical_diff_fallback>(f, "x");
  clad::differentiate<clad::opts::disable_numerical_diff_fallback>(f, "x");
  clad::differentiate<clad::opts::enable_numerical_diff_fallback>(g, "x");
  // expected-error@+1 {{the derivative of 'g' is already generated with numerical-diff-fallback on; every request that reaches 'g' uses that one derivative, so they have to ask for the fallback the same way}}
  clad::differentiate<clad::opts::disable_numerical_diff_fallback>(g, "x");
  clad::differentiate<clad::opts::disable_numerical_diff_fallback>(offFirst, "x");
  // expected-error@+1 {{the derivative of 'offFirst' is already generated with numerical-diff-fallback off;}}
  clad::differentiate<clad::opts::enable_numerical_diff_fallback>(offFirst, "x");
  clad::differentiate<clad::opts::disable_numerical_diff_fallback>(noFallback, "x");
  clad::differentiate<clad::opts::enable_numerical_diff_fallback>(noFallback, "x");
  clad::differentiate<clad::opts::enable_numerical_diff_fallback>(twoArgs, "x");
  // expected-error@+1 {{the derivative of 'twoArgs' is already generated with numerical-diff-fallback on;}}
  clad::differentiate<clad::opts::disable_numerical_diff_fallback>(twoArgs, "x");
#if __clang_major__ >= 17
  clad::gradient<clad::opts::enable_numerical_diff_fallback>(inLambda);
  // expected-error@+1 {{the derivative of 'inLambda' is already generated with numerical-diff-fallback on;}}
  clad::gradient<clad::opts::disable_numerical_diff_fallback>(inLambda);
#endif
#endif
}
#else

// expected-warning@+1 2 {{attempted differentiation of function 'tgamma'}}
double asTU(double x) { return std::tgamma(x); }
#ifdef TU_OFF
// expected-note@-2 2 {{the numerical-diff fallback is off for this request; considering 'tgamma' as 0}}
#else
// expected-note@-4 2 {{falling back to numerical differentiation for 'tgamma'}}
#endif

double askedOn(double x) { return std::tgamma(x); }
#ifdef TU_OFF
// expected-warning@-2 {{attempted differentiation of function 'tgamma'}}
// expected-note@-3 {{falling back to numerical differentiation for 'tgamma'}}
#endif

// clad reports tgamma once for each way it is asked to treat it, so this is
// reported with asTU when the translation unit is off, and here otherwise.
double askedOff(double x) { return std::tgamma(x); }
#ifdef TU_ON
// expected-warning@-2 {{attempted differentiation of function 'tgamma'}}
// expected-note@-3 {{the numerical-diff fallback is off for this request; considering 'tgamma' as 0}}
#endif

// The nested call falls back the way the request that reached it was asked,
// and is reported with askedOn.
double inner(double x) { return std::tgamma(x); }
double outer(double x) { return inner(x); }

// Several arguments go through central_difference, in either mode.
// expected-warning@+1 2 {{attempted differentiation of function 'copysign'}}
double twoArgs(double x, double y) { return std::copysign(x, y) * x; }
#ifdef TU_OFF
// expected-note@-2 2 {{the numerical-diff fallback is off for this request; considering 'copysign' as 0}}
#else
// expected-note@-4 2 {{falling back to numerical differentiation for 'copysign'}}
#endif

// Off, the call contributes nothing and leaves nothing behind.
// CHECK-OFF: void twoArgs_grad(double x, double y, double *_d_x, double *_d_y) {
// CHECK-OFF-NOT: NULL STATEMENT
// CHECK-OFF-NOT: _grad0
// CHECK-OFF: }

int main() {
  // tgamma'(3) = 2 * (3/2 - Euler's gamma), about 1.85.
  auto d = clad::differentiate(asTU, "x");
  printf("asTU %.2f\n", d.execute(3));
  // CHECK-EXEC-OFF: asTU 0.00
  // CHECK-EXEC-ON: asTU 1.85
  auto g = clad::gradient(asTU);
  double dx = 0;
  g.execute(3, &dx);
  printf("asTU grad %.2f\n", dx);
  // CHECK-EXEC-OFF: asTU grad 0.00
  // CHECK-EXEC-ON: asTU grad 1.85

  auto on = clad::differentiate<clad::opts::enable_numerical_diff_fallback>(askedOn, "x");
  printf("askedOn %.2f\n", on.execute(3));
  // A request that asks decides, whatever the command line says.
  // CHECK-EXEC: askedOn 1.85
  auto off =
      clad::differentiate<clad::opts::disable_numerical_diff_fallback>(askedOff, "x");
  printf("askedOff %.2f\n", off.execute(3));
  // CHECK-EXEC: askedOff 0.00

  auto n = clad::differentiate<clad::opts::enable_numerical_diff_fallback>(outer, "x");
  printf("outer %.2f\n", n.execute(3));
  // CHECK-EXEC: outer 1.85

  // d/dx copysign(x, y) * x is 2 * |x| for y > 0; off, only the factor x's
  // own derivative is left.
  auto two = clad::gradient(twoArgs);
  double dx2 = 0, dy2 = 0;
  two.execute(2, 3, &dx2, &dy2);
  printf("twoArgs %.2f %.2f\n", dx2, dy2);
  // CHECK-EXEC-OFF: twoArgs 2.00 0.00
  // CHECK-EXEC-ON: twoArgs 4.00 0.00
  auto twoFwd = clad::differentiate(twoArgs, "x");
  printf("twoArgs fwd %.2f\n", twoFwd.execute(2, 3));
  // CHECK-EXEC-OFF: twoArgs fwd 2.00
  // CHECK-EXEC-ON: twoArgs fwd 4.00
}
#endif
