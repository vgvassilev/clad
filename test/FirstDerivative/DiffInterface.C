// RUN: %cladclang  -ferror-limit=100 %s -I%S/../../include -fsyntax-only -Xclang -verify 2>&1 | %filecheck %s

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

int f_no_definition(int x);

int f_redeclared(int x) {
    return x;
}

int f_redeclared(int x);

// CHECK: int f_redeclared_darg0(int x) {
// CHECK-NEXT:   int _d_x = 1;
// CHECK-NEXT:   return _d_x;
// CHECK: }

int f_try_catch(int x)
  try { // expected-warning {{attempted to differentiate unsupported statement, no changes applied}}
    return x;
  }
  catch (int) {
    return 0;
  }
  catch (...) {
    return 1;
  }

// CHECK: int f_try_catch_darg0(int x) {
// CHECK-NEXT:    int _d_x = 1;
// CHECK-NEXT:    try {
// CHECK-NEXT:        return x;
// CHECK-NEXT:    } catch (int) {
// CHECK-NEXT:        return 0;
// CHECK-NEXT:    } catch (...) {
// CHECK-NEXT:        return 1;
// CHECK-NEXT:    }
// CHECK-NEXT: }

void fn_with_no_return(double x) { return; }

// CHECK: void fn_with_no_return_darg0(double x) {
// CHECK-NEXT:   double _d_x = 1;
// CHECK-NEXT: }

double fn_with_no_params() {
  return 11;
}

struct Complex {
  double real, im;
  double getReal() {
    return real;
  }
};

double fn_with_Complex_type_param(Complex c) {
  return c.real + c.im;
}

struct ComplexPair {
  Complex c1, c2;
};

double fn_with_ComplexPair_type_param(ComplexPair cp) {
  return 11;
};

int main () {
  int x = 4 * 5;
  clad::differentiate(f_1, 0);

  clad::differentiate(f_2, 0);

  clad::differentiate(f_2, 1);

  clad::differentiate(f_2, 2);

  clad::differentiate(f_2, -1); // expected-error {{Invalid argument index '-1' of '3' argument(s)}}

  clad::differentiate(f_2, 3); // expected-error {{Invalid argument index '3' of '3' argument(s)}}

  clad::differentiate(f_2, 9); // expected-error {{Invalid argument index '9' of '3' argument(s)}}

  clad::differentiate(f_2, x); // expected-error {{Failed to parse the parameters, must be a string or numeric literal}}

  clad::differentiate(f_2, f_2); // expected-error {{Failed to parse the parameters, must be a string or numeric literal}}

  clad::gradient(f_2, -1); // expected-error {{Invalid argument index '-1' of '3' argument(s)}}

  clad::gradient(f_2, "9"); // expected-error {{Invalid argument index '9' of '3' argument(s)}}
  
  clad::differentiate(f_3, 0); // expected-error {{Invalid argument index '0' of '0' argument(s)}}

  float one = 1.0;
  clad::differentiate(f_2, one); // expected-error {{Failed to parse the parameters, must be a string or numeric literal}}

  clad::differentiate(f_no_definition, 0); // expected-error {{attempted differentiation of function 'f_no_definition', which does not have a definition}}

  clad::differentiate(f_redeclared, 0);

  clad::differentiate(f_try_catch, 0);

  clad::differentiate(f_2, "x");
  clad::differentiate(f_2, " y ");
  clad::differentiate(f_2, "z");

  clad::differentiate(f_2, "x, y"); // expected-error {{Forward mode differentiation w.r.t. several parameters at once is not supported, call 'clad::differentiate' for each parameter separately}}
  clad::differentiate(f_2, "t"); // expected-error {{Requested parameter name 't' was not found among function parameters}}
  clad::differentiate(f_2, "x, x"); // expected-error {{Requested parameter 'x' was specified multiple times}}
  
  clad::differentiate(f_2, ""); // expected-error {{No parameters were provided}}
  clad::differentiate(fn_with_no_return, "x");
  clad::differentiate(fn_with_no_params); // expected-error {{Attempted to differentiate a function without parameters}}

  clad::differentiate(f_2, "x.mem1");                                   // expected-error {{Fields can only be provided for class type parameters. Field information is incorrectly specified in 'x.mem1' for non-class type parameter 'x'}}
  clad::differentiate(fn_with_Complex_type_param, "c.real.im");         // expected-error {{Path specified by fields in 'c.real.im' is invalid.}}
  clad::differentiate(fn_with_ComplexPair_type_param, "cp.c1");         // expected-error {{Attempted differentiation w.r.t. member 'cp.c1' which is not of real type.}}
  clad::differentiate(fn_with_Complex_type_param, "c.getReal");         // expected-error {{Path specified by fields in 'c.getReal' is invalid.}}
  clad::differentiate(fn_with_Complex_type_param, "c.invalidField");    // expected-error {{Path specified by fields in 'c.invalidField' is invalid.}}
  return 0;
}
