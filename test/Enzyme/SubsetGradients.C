// RUN: %cladclang -fsyntax-only -Xclang -plugin-arg-clad -Xclang \
// RUN:   -fdump-derived-fn %s -I%S/../../include 2>&1 | %filecheck %s

// Differentiating only some parameters leaves the rest without an adjoint.
// Enzyme's positional convention cannot express that -- omitting a shadow
// shifts every later argument -- so such a request is annotated with the
// activity markers instead. A request covering the whole signature keeps the
// positional form.
//
// This test needs no Enzyme: it checks the call clad emits, which is the part
// that used to index past the end of the derivative's parameters and crash.

#include "clad/Differentiator/Differentiator.h"

double byValue(double x, double y, double z) { return x * y * z; }

double byPointer(const double* a, const double* b, int n) {
  double s = 0;
  for (int i = 0; i < n; i++)
    s += a[i] * b[i];
  return s;
}

double whole(double x, double y) { return x * y; }

int main() {
  clad::gradient<clad::opts::use_enzyme>(byValue, "x");
  clad::gradient<clad::opts::use_enzyme>(byPointer, "a");
  clad::gradient<clad::opts::use_enzyme>(whole);
}

// An active by-value argument is marked enzyme_out and its adjoint comes back
// in the returned struct; the untouched ones are enzyme_const.
// CHECK: void byValue_grad_0(double x, double y, double z, double *_d_x) {
// CHECK-NEXT: clad::EnzymeGradient<1{{U?}}> grad = __enzyme_autodiff_byValue(byValue, enzyme_out, x, enzyme_const, y, enzyme_const, z);
// CHECK-NEXT: *_d_x = grad.d_arr[0{{U?}}];

// An active pointer is marked enzyme_dup and followed by its shadow. The
// non-differentiated pointer and the integer are both enzyme_const.
// CHECK: void byPointer_grad_0(const double *a, const double *b, int n, double *_d_a) {
// CHECK: __enzyme_autodiff_byPointer(byPointer, enzyme_dup, a, _d_a, enzyme_const, b, enzyme_const, n);

// Every parameter differentiated: no markers, exactly as before.
// CHECK: void whole_grad_enzyme(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT: clad::EnzymeGradient<2{{U?}}> grad = __enzyme_autodiff_whole(whole, x, y);
