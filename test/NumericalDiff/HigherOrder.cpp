// A derivative the fallback computed numerically is an ordinary call in the
// code clad differentiates next, and a hessian, a second order request and a
// pushforward all come back to one. Each used to emit code that did not
// compile; the first two are now said to be as far as numerical
// differentiation goes, and the third is computed.
// RUN: %cladnumdiffclang %s -I%S/../../include -oHigherOrder.out -Xclang -verify 2>&1
// RUN: ./HigherOrder.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"

#include <cmath>

extern "C" int printf(const char* fmt, ...);

// Reference values at x = 2, from G(2) = 1, psi(2) = 1 - gamma and
// psi'(2) = pi^2/6 - 1:
//   G'(2)  = G psi             = 0.422784
//   G''(2) = G (psi^2 + psi')  = 0.823681

// expected-warning@+3 {{attempted differentiation of function 'tgamma'}}
// expected-note@+2 {{falling back to numerical differentiation for 'tgamma'}}
// expected-warning@+1 {{the numerical derivative of 'tgamma' cannot be differentiated again: numerical differentiation has no derivative of its own; considering this derivative as 0}}
double gamma1(double x) { return std::tgamma(x); }
// expected-note@-1 {{a custom derivative for 'tgamma' is differentiated as many times as it is asked for}}

// A hessian differentiates the forward derivative in reverse mode, so the
// numerically differentiated call turns up there too -- as a call to derive,
// which is what used to fail.
// expected-warning@+3 {{attempted differentiation of function 'tgamma'}}
// expected-note@+2 {{falling back to numerical differentiation for 'tgamma'}}
// expected-warning@+1 {{the numerical derivative of 'tgamma' cannot be differentiated again}}
double gamma2(double x, double y) { return std::tgamma(x * y); }
// expected-note@-1 {{a custom derivative for 'tgamma' is differentiated as many times as it is asked for}}

// A float primal: the tangent numerical differentiation hands back is a
// `precision`, which is a double, and the pushforward it goes into is a
// ValueAndPushforward<float, float>. A braced initializer refuses the
// narrowing, so the tangent is converted like the value beside it.
//
// Which function the fallback reports here is the standard library's to
// decide -- libstdc++ sends std::tgamma(float) to __builtin_tgammaf from
// inside <cmath> -- so the pair is matched wherever it lands, and not at all
// where a custom derivative is available instead.
// expected-warning@* 0-2 {{attempted differentiation of function}}
// expected-note@* 0-2 {{falling back to numerical differentiation}}
float gammaf(float x) { return std::tgamma(x); }

float scaled(float x) { return gammaf(x) * x; }

int main() {
  auto d1 = clad::differentiate(gamma1, "x");
  printf("first order: %.2f\n", d1.execute(2)); // CHECK-EXEC: first order: 0.42

  // Zero, and said to be: the second derivative of a function clad can only
  // differentiate numerically is not available. The true value is 0.82.
  auto d2 = clad::differentiate<2>(gamma1, "x");
  printf("second order: %.2f\n", d2.execute(2)); // CHECK-EXEC: second order: 0.00

  // d2/dx2 and d2/dy2 are the second derivative of tgamma and go the same
  // way. d2/dxdy keeps the term the chain rule contributes outside the
  // numerical derivative, G'(2) = 0.42, and loses the x*y*G''(x*y) beside it.
  auto h = clad::hessian(gamma2);
  double matrix[4] = {0, 0, 0, 0};
  h.execute(2, 1, matrix);
  printf("hessian: %.2f %.2f %.2f %.2f\n", matrix[0], matrix[1], matrix[2],
         matrix[3]); // CHECK-EXEC: hessian: 0.00 0.42 0.42 0.00

  // Computed, not refused: d(G(x) * x)/dx at 2 is G'(2) * 2 + G(2) = 1.85.
  auto d3 = clad::differentiate(scaled, "x");
  printf("float pushforward: %.2f\n",
         (double)d3.execute(2)); // CHECK-EXEC: float pushforward: 1.85
}
