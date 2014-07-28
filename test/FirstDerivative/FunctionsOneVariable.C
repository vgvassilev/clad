// RUN: %cladclang %s -I%S/../../include -fsyntax-only -Xclang -verify 2>&1 | FileCheck %s

#include "clad/Differentiator/Differentiator.h"

extern "C" int printf(const char* fmt, ...);

int f_simple(int x) {
  //  printf("This is f(x).\n");
  return x*x;
}


int f_simple_negative(int x) {
  //  printf("This is f(x).\n");
  return -x*x;
}
// CHECK: int f_simple_negative_dx(int x) {
// CHECK-NEXT: return (-1 * x + -x * 1);
// CHECK-NEXT: }

int main () {
  int x = 4;
  clad::differentiate(f_simple, x); // expected-error {{Must be an integral value}}
  // Here the second arg denotes the differentiation of f with respect to the
  // given arg.
  //clad::differentiate(f_simple, 1);
  clad::differentiate<3>(f_simple, 0);
  clad::differentiate(f_simple, -1); // expected-error {{Invalid argument index -1 among 1 argument(s)}}
  clad::differentiate(f_simple, 0);
  clad::differentiate(f_simple_negative, 0);

  return 0;
}
