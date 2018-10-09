// RUN: %cladclang %s -I%S/../../include -fsyntax-only -Xclang -verify 2>&1 | FileCheck %s

#include "clad/Differentiator/Differentiator.h"

//CHECK-NOT: {{.*error|warning|note:.*}}

extern "C" int printf(const char* fmt, ...);

int f_1(float y) {
  int x = 1, z = 3;
  return x * y * z; // x * z
}

// CHECK: int f_1_darg0(float y) {
// CHECK-NEXT: float _d_y = 1;
// CHECK-NEXT: int _d_x = 0, _d_z = 0;
// CHECK-NEXT: int x = 1, z = 3;
// CHECK-NEXT: float _t0 = x * y;
// CHECK-NEXT: return (_d_x * y + x * _d_y) * z + _t0 * _d_z;
// CHECK-NEXT: }

int f_2(int x, float y, int z) {
  return x * y * z; // y * z;
}

// CHECK: int f_2_darg0(int x, float y, int z) {
// CHECK-NEXT: int _d_x = 1;
// CHECK-NEXT: float _d_y = 0;
// CHECK-NEXT: int _d_z = 0;
// CHECK-NEXT: float _t0 = x * y;
// CHECK-NEXT: return (_d_x * y + x * _d_y) * z + _t0 * _d_z;
// CHECK-NEXT: }

// x * z
// CHECK: int f_2_darg1(int x, float y, int z) {
// CHECK-NEXT: int _d_x = 0;
// CHECK-NEXT: float _d_y = 1;
// CHECK-NEXT: int _d_z = 0;
// CHECK-NEXT: float _t0 = x * y;
// CHECK-NEXT: return (_d_x * y + x * _d_y) * z + _t0 * _d_z;
// CHECK-NEXT: }

// x * y
// CHECK: int f_2_darg2(int x, float y, int z) {
// CHECK-NEXT: int _d_x = 0;
// CHECK-NEXT: float _d_y = 0;
// CHECK-NEXT: int _d_z = 1;
// CHECK-NEXT: float _t0 = x * y;
// CHECK-NEXT: return (_d_x * y + x * _d_y) * z + _t0 * _d_z;
// CHECK-NEXT: }

int f_3() {
  int x = 1, z = 3;
  float y = 2;
  return x * y * z; // should not be differentiated
}

int f_no_definition(int x); // expected-error {{attempted differentiation of function 'f_no_definition', which does not have a definition}}

int f_redeclared(int x) {
    return x;
}

int f_redeclared(int x);

// CHECK: int f_redeclared_darg0(int x) {
// CHECK-NEXT:   int _d_x = 1;
// CHECK-NEXT:   return _d_x;
// CHECK: }

int f_try_catch(int x)
  try {
    return x;
  }
  catch (int) {
    return 0;
  } // expected-warning {{attempted to differentiate unsupported statement, no changes applied}}

// CHECK: int f_try_catch_darg0(int x) {
// CHECK-NEXT:    int _d_x = 1;
// CHECK-NEXT:    try {
// CHECK-NEXT:        return x;
// CHECK-NEXT:    } catch (int) {
// CHECK-NEXT:        return 0;
// CHECK-NEXT:    }
// CHECK-NEXT: }

int main () {
  int x = 4 * 5;
  clad::differentiate(f_1, 0);

  clad::differentiate(f_2, 0);

  clad::differentiate(f_2, 1);

  clad::differentiate(f_2, 2);

  clad::differentiate(f_2, -1); // expected-error {{Invalid argument index -1 among 3 argument(s)}}
  // expected-note@clad/Differentiator/Differentiator.h:114 {{candidate function not viable: no known conversion from 'int (int, float, int)' to 'unsigned int' for 2nd argument}}
  // expected-note@clad/Differentiator/Differentiator.h:121 {{candidate template ignored: could not match 'R (C::*)(Args...)' against 'int (*)(int, float, int)'}}

  clad::differentiate(f_2, -1); // expected-error {{Invalid argument index -1 among 3 argument(s)}}

  clad::differentiate(f_2, 3); // expected-error {{Invalid argument index 3 among 3 argument(s)}}

  clad::differentiate(f_2, 9); // expected-error {{Invalid argument index 9 among 3 argument(s)}}

  clad::differentiate(f_2, x); // expected-error {{Must be an integral value}}

  clad::differentiate(f_2, f_2); // expected-error {{no matching function for call to 'differentiate'}}

  clad::differentiate(f_3, 0); // expected-error {{Trying to differentiate function 'f_3' taking no arguments}}

  float one = 1.0;
  clad::differentiate(f_2, one); // expected-error {{Must be an integral value}}

  clad::differentiate(f_no_definition, 0);

  clad::differentiate(f_redeclared, 0);

  clad::differentiate(f_try_catch, 0);

  return 0;
}
