// RUN: %cladclang %s -I%S/../../include -fsyntax-only -Xclang -verify 2>&1 | FileCheck %s

#include "clad/Differentiator/Differentiator.h"

//CHECK-NOT: {{.*error|warning|note:.*}}

extern "C" int printf(const char* fmt, ...);

int f_1(float y) {
  int x = 1, z = 3;
  return x * y * z; // x * z
}

// CHECK: int f_1_darg0(float y) {
// CHECK-NEXT: int x = 1, z = 3;
// CHECK-NEXT: return ((0 * y + x * 1.F) * z + x * y * 0);
// CHECK-NEXT: }

int f_2(int x, float y, int z) {
  return x * y * z; // y * z;
}

// CHECK: int f_2_darg0(int x, float y, int z) {
// CHECK-NEXT: return ((1 * y + x * 0.F) * z + x * y * 0);
// CHECK-NEXT: }

// x * z
// CHECK: int f_2_darg1(int x, float y, int z) {
// CHECK-NEXT: return ((0 * y + x * 1.F) * z + x * y * 0);
// CHECK-NEXT: }

// x * y
// CHECK: int f_2_darg2(int x, float y, int z) {
// CHECK-NEXT: return ((0 * y + x * 0.F) * z + x * y * 1);
// CHECK-NEXT: }

int f_3() {
  int x = 1, z = 3;
  float y = 2;
  return x * y * z; // should not be differentiated
}

int main () {
  int x = 4 * 5;
  clad::differentiate(f_1, 0);

  clad::differentiate(f_2, 0);

  clad::differentiate(f_2, 1);

  clad::differentiate(f_2, 2);

  clad::differentiate(f_2, -1); // expected-error {{Invalid argument index -1 among 3 argument(s)}}
  //expected-note@clad/Differentiator/Differentiator.h:97 {{candidate function [with N = 1, R = int, Args = <int, float, int>] not viable: no known conversion from 'int (int, float, int)' to 'unsigned int' for 2nd argument}}
  //expected-note@clad/Differentiator/Differentiator.h:104 {{candidate template ignored: could not match 'R (C::*)(Args...)' against 'int (*)(int, float, int)'}}

  clad::differentiate(f_2, -1); // expected-error {{Invalid argument index -1 among 3 argument(s)}}

  clad::differentiate(f_2, 3); // expected-error {{Invalid argument index 3 among 3 argument(s)}}

  clad::differentiate(f_2, 9); // expected-error {{Invalid argument index 9 among 3 argument(s)}}

  clad::differentiate(f_2, x); // expected-error {{Must be an integral value}}

  clad::differentiate(f_2, f_2); // expected-error {{no matching function for call to 'differentiate'}}

  clad::differentiate(f_3, 0); // expected-error {{Trying to differentiate function 'f_3' taking no arguments}}

  float one = 1.0;
  clad::differentiate(f_2, one); // expected-error {{Must be an integral value}}

  return 0;
}
