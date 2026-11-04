// RUN: %cladclang -fsyntax-only -Xclang -verify -std=c++17 -I%S/../../include %s

// A pushforward returning a non-const reference must return a reference to the
// tangent, and a value without a derivative has none. `ref` is the reduced
// shape of libstdc++'s std::basic_string::replace.

#include "clad/Differentiator/Differentiator.h"

double& ref(double& x, int n);

double& h(double& x) {
  return ref(x, 1); // expected-warning {{attempted differentiation of function 'ref' without definition}} \
                    // expected-note {{considering 'ref' as 0}} \
                    // expected-error {{cannot differentiate 'h': the returned reference refers to a value without a derivative}}
}

double fn(double x) { return h(x); }

double global = 4;
double& to_global(double& x) {
  return global; // expected-error {{cannot differentiate 'to_global': the returned reference refers to a value without a derivative}}
}

double fn_global(double x) { return to_global(x) * x; }

// A class-typed member read through such a pointer has no tangent at all.
struct In {
  double a;
};
struct S {
  In in;
  double* ptr;
};
double d0 = 7;
S s{{2}, &d0};
S* sp = &s;
In& get(double& x) {
  return sp->in; // expected-warning {{'in' is accessed through an expression without a derivative; its derivative is treated as 0}} \
                 // expected-error {{cannot differentiate 'get': the returned reference refers to a value without a derivative}}
}

double fn_member(double x) { return get(x).a * x; }

void test() {
  auto d = clad::differentiate(fn, "x");
  auto d_global = clad::differentiate(fn_global, "x");
  auto d_member = clad::differentiate(fn_member, "x");
}
