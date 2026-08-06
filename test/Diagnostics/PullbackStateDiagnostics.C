// RUN: %cladclang %s -I%S/../../include -fsyntax-only -Xclang -verify 2>&1 | %filecheck %s
//
// A custom reverse_forw may declare a trailing clad::pullback_state<S>&
// out-param to hand per-call state to its pullback; the pullback must then
// accept the matching clad::pullback_state<S> by value. When it does not, clad
// rejects the pullback and the "expected signature '...' does not match"
// diagnostic spells out the required parameter -- i.e. it suggests the shape of
// the forward declaration the author must write. This mirrors the existing
// signature-mismatch machinery in CustomDerivativeDiagnostics.C.

#include "clad/Differentiator/Differentiator.h"

// Case 1: the pullback omits the trailing clad::pullback_state<double>.
double* g(double* p);

namespace clad::custom_derivatives {
clad::ValueAndAdjoint<double*, double*>
g_reverse_forw(double* p, double* d_p, clad::pullback_state<double>& state) {
  return {p, d_p};
}
void g_pullback(double* p, double* d_p) { // expected-note {{candidate 'g_pullback' has different number of parameters (expected 3 but has 2)}}
}
} // namespace clad::custom_derivatives

double f1(double x) {
  return *g(&x);
  // The expected signature names the clad::pullback_state<double> the pullback
  // must accept -- the shape suggestion the author needs.
  // expected-error-re@-3 {{expected signature{{.*}}clad::pullback_state<double>{{.*}}does not match}}
  // expected-warning@-4 {{attempted differentiation of function 'g' without definition and no suitable overload was found in namespace 'custom_derivatives'}}
  // expected-note@-5 {{numerical differentiation is not viable for 'g'; considering 'g' as 0}}
}

// Case 2: the pullback accepts a pullback_state of the WRONG payload type; the
// suggestion must name the payload the reverse_forw actually produces (double).
double* h(double* p);

namespace clad::custom_derivatives {
clad::ValueAndAdjoint<double*, double*>
h_reverse_forw(double* p, double* d_p, clad::pullback_state<double>& state) {
  return {p, d_p};
}
void h_pullback(double* p, double* d_p, // expected-note {{candidate 'h_pullback' has type mismatch at 3rd parameter (expected 'pullback_state<double>' but has 'pullback_state<int>')}}
                clad::pullback_state<int> state) {
}
} // namespace clad::custom_derivatives

double f2(double x) {
  return *h(&x);
  // expected-error-re@-1 {{expected signature{{.*}}clad::pullback_state<double>{{.*}}does not match}}
  // expected-warning@-2 {{attempted differentiation of function 'h' without definition and no suitable overload was found in namespace 'custom_derivatives'}}
  // expected-note@-3 {{numerical differentiation is not viable for 'h'; considering 'h' as 0}}
}

// Case 3: same mismatch, but the primal has a visible body. clad still uses the
// matched reverse_forw, so the state carrier must be threaded even though the
// pullback did not resolve -- otherwise the reverse_forw call is left an
// argument short and clad crashes. This guards that recovery path.
double* k(double* p) { return p; }

namespace clad::custom_derivatives {
clad::ValueAndAdjoint<double*, double*>
k_reverse_forw(double* p, double* d_p, clad::pullback_state<double>& state) {
  return {p, d_p};
}
void k_pullback(double* p, double* d_p) { // expected-note {{candidate 'k_pullback' has different number of parameters (expected 3 but has 2)}}
}
} // namespace clad::custom_derivatives

double f3(double x) {
  return *k(&x); // expected-error-re {{expected signature{{.*}}clad::pullback_state<double>{{.*}}does not match}}
}

int main() {
  clad::gradient(f1, "x");
  clad::gradient(f2, "x");
  clad::gradient(f3, "x");
}
