// RUN: %cladclang %s -I%S/../../include -oActivityPlannedPullback.out \
// RUN:   -Xclang -verify 2>&1 | %filecheck %s
// RUN: ./ActivityPlannedPullback.out | %filecheck_exec %s
// RUN: %cladclang %s -I%S/../../include -fsyntax-only -Xclang -verify \
// RUN:   -DREPORT_ONCE

// With activity analysis, a call that passes a local nothing varies gets the
// pullback without that parameter, and so does a call in that pullback that
// passes the parameter on. The planner plans these pullbacks too, and walks
// their bodies for what it records there: here, that inner moves a
// derivative through the global g.

#include "clad/Differentiator/Differentiator.h"

#include <cstdio>

struct E {
  double k;
};

double g = 0; // expected-warning {{gradient uses a global variable 'g'}}
void set(double v) { g = v; }
double get() { return g; }

double inner(const E& e, double x) {
  set(x * e.k);
  return get();
}

double h(const E& e, double x) { return inner(e, x); }

double f(double x) {
  E a{x};
  E b{3};
  return h(a, x) + h(b, x);
}

// CHECK: void inner_pullback(const E &e, double x, double _d_y, double *_d_x) {
// CHECK: get_pullback(_d_y);
// CHECK: set_pullback(x * e.k, &_r0);
// CHECK: *_d_x += _r0 * e.k;

#ifdef REPORT_ONCE
// The planner walks report() for each of its two pullbacks, and reports what
// is wrong in it once.
double sq(double x) { return x * x; }
namespace clad::custom_derivatives {
void sq_pullback(double x, double d, double* dx) { // expected-note {{candidate 'sq_pullback' has different number of parameters}}
  *dx += 2 * x * d;
}
} // namespace clad::custom_derivatives

double report(const E& e, double x) { return sq(x) * e.k; } // expected-error {{user-defined derivative for 'sq' was provided but not used}}

double r(double x) {
  E a{x};
  E b{3};
  return report(a, x) + report(b, x);
}

void reportOnce() { clad::gradient<clad::opts::enable_va>(r); }
#endif

int main() {
  auto df = clad::gradient<clad::opts::enable_va>(f);
  double dx = 0;
  df.execute(2, &dx);
  printf("%.2f\n", dx); // CHECK-EXEC: 7.00
}
