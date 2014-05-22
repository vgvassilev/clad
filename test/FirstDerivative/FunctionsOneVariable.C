// RUN: %cladclang %s -I%S/../../include -fsyntax-only -Xclang -verify 2>&1 | FileCheck %s

#include "clad/Differentiator/Differentiator.h"

extern "C" int printf(const char* fmt, ...);

int f_simple(int x) {
  //  printf("This is f(x).\n");
  return x*x;
}
// CHECK: int f_simple_derived_x(int x) {
// CHECK-NEXT: return (1 * x + x * 1);
// CHECK-NEXT: }

int main () {
  int x = 4;
  diff(f_simple, x); // expected-error {{Must be an integral value}}
  // Here the second arg denotes the differentiation of f with respect to the
  // given arg.
  diff(f_simple, 2); // expected-error {{Invalid argument index 2 among 1 argument(s)}}
  diff(f_simple, -1); // expected-error {{Invalid argument index -1 among 1 argument(s)}}
  diff(f_simple, 1);
  
  //long y = 2;
  //diff(g, y);
  // FIXME:
  // Would it be better if the signature is diff(f, 1), where 1 is the number
  // of the respected parameter? This will help to get rid of the variadic
  // templates (C++11) feature? diff(g, 2);
  
  return 0;
}
