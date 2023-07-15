// RUN: %cladclang %s -I%S/../../include -fsyntax-only -Xclang -verify 2>&1 | FileCheck %s

//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

double f1(double x, double y) {
  return x*y;
}

// CHECK: void f1_dvec(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:   clad::array<double> _d_vector_x = {1., 0.};
// CHECK-NEXT:   clad::array<double> _d_vector_y = {0., 1.};
// CHECK-NEXT:   {
// CHECK-NEXT:     clad::array<double> _d_vector_return =  _d_vector_x * y + x * _d_vector_y;
// CHECK-NEXT:     *_d_x = _d_vector_return[0];
// CHECK-NEXT:     *_d_y = _d_vector_return[1];
// CHECK-NEXT:     return;
// CHECK-NEXT:   }
// CHECK-NEXT: }

double f2(double x, double y) {
  return x+y;
}

void f2_dvec(double x, double y, double *_d_x, double *_d_y);

// CHECK: void f2_dvec(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:   clad::array<double> _d_vector_x = {1., 0.};
// CHECK-NEXT:   clad::array<double> _d_vector_y = {0., 1.};
// CHECK-NEXT:   {
// CHECK-NEXT:     clad::array<double> _d_vector_return = _d_vector_x + _d_vector_y;
// CHECK-NEXT:     *_d_x = _d_vector_return[0];
// CHECK-NEXT:     *_d_y = _d_vector_return[1];
// CHECK-NEXT:     return;
// CHECK-NEXT:   }
// CHECK-NEXT: }

double f_try_catch(double x, double y)
  try { // expected-warning {{attempted to differentiate unsupported statement, no changes applied}}
    return x;
  }
  catch (int) {
    return 0;
  }

// CHECK: void f_try_catch_dvec(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:   clad::array<double> _d_vector_x = {1., 0.};
// CHECK-NEXT:   clad::array<double> _d_vector_y = {0., 1.};
// CHECK-NEXT:    try {
// CHECK-NEXT:        return x;
// CHECK-NEXT:    } catch (int) {
// CHECK-NEXT:        return 0;
// CHECK-NEXT:    }
// CHECK-NEXT: }

int main() {
  clad::differentiate<clad::opts::vector_mode>(f1);
  clad::differentiate<clad::opts::vector_mode>(f2);
  clad::differentiate<clad::opts::vector_mode>(f_try_catch);
  clad::differentiate<2, clad::opts::vector_mode>(f_try_catch); // expected-error {{Only first order derivative is supported for now in vector forward mode}}
  clad::differentiate<clad::opts::use_enzyme, clad::opts::vector_mode>(f1); // expected-error {{Enzyme's vector mode is not yet supported}}
  
  clad::gradient<clad::opts::vector_mode>(f1, "x, y, z"); // expected-error {{Reverse vector mode is not yet supported.}}
  return 0;
}
