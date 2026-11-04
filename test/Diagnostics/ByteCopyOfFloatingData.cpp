// RUN: %cladclang %s -I%S/../../include -fsyntax-only -Xclang -verify 2>&1
//
// A byte copy of floating-point data moves values clad has to differentiate,
// through `void*`. One between arrays of the same floating-point type is
// differentiated as an element-wise copy; any other is reported.

#include "clad/Differentiator/Differentiator.h"
#include <cstring>

// A copy of integers has no derivative to lose and is not reported.
struct Ints { int a, b; };
double int_memcpy(double k) {
  Ints a{1, 2}, b;
  __builtin_memcpy(&b, &a, sizeof a); // expected-warning 2 {{attempted differentiation of function '__builtin_memcpy' without definition}} \
                                    // expected-note 2 {{considering '__builtin_memcpy' as 0}}
  return k;
}

struct P { double x, y; };

// The defaulted assignment copies an array of P the same way.
struct V { P m[2]; };

double copy_assign(double k) {
  V a{};
  a.m[0].x = k;
  V b{};
  b = a; // expected-error {{cannot differentiate the defaulted copy assignment of 'V': it copies 'P[2]' with '__builtin_memcpy', which has no derivative}} \
         // expected-note {{provide custom 'operator_equal_reverse_forw' and 'operator_equal_pullback' for 'V'}}
  return b.m[0].x;
}

// The double may sit in a base class.
struct Base { double v; };
struct Derived : Base { int n; };
Derived gd, gs; // expected-warning {{gradient uses a global variable 'gd'}} expected-warning {{gradient uses a global variable 'gs'}}
double from_base(double k) {
  __builtin_memcpy(&gd, &gs, sizeof gs); // expected-error {{cannot differentiate '__builtin_memcpy' copying 'Derived', which holds floating-point values; its derivative would be zero}} \
                                         // expected-note {{copy the values with assignments}}
  return k;
}

double user_memcpy(double k) {
  P a[2] = {{k, k}, {k, k}};
  P b[2];
  std::memcpy(b, a, sizeof(a)); // expected-error {{cannot differentiate 'memcpy' copying 'P[2]'}} \
                                // expected-note {{copy the values with assignments}}
  return b[1].y;
}

// The floating-point end of a copy may be the source.
double through_bytes(double x) {
  unsigned char buf[sizeof(double)];
  std::memcpy(buf, &x, sizeof x); // expected-error {{cannot differentiate 'memcpy' copying 'double'}} \
                                  // expected-note {{copy the values with assignments}}
  double y;
  std::memcpy(&y, buf, sizeof y); // expected-error {{cannot differentiate 'memcpy' copying 'double'}} \
                                  // expected-note {{copy the values with assignments}}
  return y;
}

int main() {
  clad::gradient(int_memcpy);
  clad::gradient(copy_assign);
  clad::gradient(from_base);
  clad::gradient(user_memcpy);
  clad::gradient(through_bytes);
}
