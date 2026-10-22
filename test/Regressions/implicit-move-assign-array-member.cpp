// RUN: %cladclang %s -I%S/../../include -fsyntax-only -Xclang -verify 2>&1
//
// Clang's implicit move assignment memcpys an array member from the address of
// an xvalue, which Sema refuses to rebuild; clad crashed on the null. What is
// left is the byte-copy diagnostic.

#include "clad/Differentiator/Differentiator.h"

struct V { double m[2]; };

V twice(const V& a) {
  V r{};
  r.m[0] = 2 * a.m[0];
  return r;
}

double move_assign(double k) {
  V a{};
  a.m[0] = k;
  V b{};
  b = twice(a); // expected-error {{cannot differentiate the defaulted move assignment of 'V'}} \
                // expected-note {{provide custom 'operator_equal_reverse_forw' and 'operator_equal_pullback' for 'V'}}
  return b.m[0];
}

int main() { clad::gradient(move_assign); }
