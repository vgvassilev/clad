// RUN: %cladclang %s -I%S/../../include -fsyntax-only -Xclang -verify 2>&1 | FileCheck %s

#include "clad/Differentiator/Differentiator.h"

//CHECK-NOT: {{.*error|warning|note:.*}}

extern "C" int printf(const char* fmt, ...);

float f_simple(float x) {
  //  printf("This is f(x).\n");
  return x*x;
}

//CHECK:float f_simple_darg0(float x) {
//CHECK-NEXT:    float _d_x = 1;
//CHECK-NEXT:    return _d_x * x + x * _d_x;
//CHECK-NEXT:}

//CHECK: float f_simple_d2arg0(float x) {
//CHECK-NEXT:    float _d_x = 1;
//CHECK-NEXT:    float _d__d_x = 0;
//CHECK-NEXT:    float _d_x0 = 1;
//CHECK-NEXT:    return _d__d_x * x + _d_x0 * _d_x + _d_x * _d_x0 + x * _d__d_x;
//CHECK-NEXT: }

//CHECK: float f_simple_d3arg0(float x) {
//CHECK-NEXT:    float _d_x = 1;
//CHECK-NEXT:    float _d__d_x = 0;
//CHECK-NEXT:    float _d_x0 = 1;
//CHECK-NEXT:    float _d__d__d_x = 0;
//CHECK-NEXT:    float _d__d_x1 = 0;
//CHECK-NEXT:    float _d__d_x0 = 0;
//CHECK-NEXT:    float _d_x02 = 1;
//CHECK-NEXT:    return _d__d__d_x * x + _d__d_x1 * _d_x + _d__d_x0 * _d_x0 + _d_x02 * _d__d_x + _d__d_x * _d_x02 + _d_x0 * _d__d_x0 + _d_x * _d__d_x1 + x * _d__d__d_x;
//CHECK-NEXT: }

int f_simple_negative(int x) {
  //  printf("This is f(x).\n");
  return -x*x;
}
// CHECK: int f_simple_negative_darg0(int x) {
// CHECK-NEXT: int _d_x = 1;
// CHECK-NEXT: int _t0 = -x;
// CHECK-NEXT: return -_d_x * x + _t0 * _d_x;
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
