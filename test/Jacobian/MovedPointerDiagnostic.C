// RUN: %cladclang %s -I%S/../../include -fsyntax-only -Xclang -verify 2>&1

#include "clad/Differentiator/Differentiator.h"

void f_inc(double x, double* y, double* out) {
  y++; // expected-error {{vector mode cannot follow pointer 'y' once it is moved or reassigned; index it instead}}
  out[0] = *y * x;
}

void f_dec_in_expr(double x, double* y, double* out) {
  out[0] = *(--y) * x; // expected-error {{vector mode cannot follow pointer 'y' once it is moved or reassigned; index it instead}}
}

void f_add_assign(double x, double* y, double* out) {
  y += 1; // expected-error {{vector mode cannot follow pointer 'y' once it is moved or reassigned; index it instead}}
  out[0] = y[0] * x;
}

void f_rebind(double x, double* y, double* out) {
  y = out; // expected-error {{vector mode cannot follow pointer 'y' once it is moved or reassigned; index it instead}}
  out[0] = *y * x;
}

// Writing through the pointer is fine.
void f_write_through(double x, double* y, double* out) {
  *y += x;
  y[1] *= x;
  *(y + 1) -= x;
  out[0] = *y;
}

double g_vector(const double* arr, double w) {
  arr = arr + 1; // expected-error {{vector mode cannot follow pointer 'arr' once it is moved or reassigned; index it instead}}
  return *arr * w;
}

int main() {
  clad::jacobian(f_inc);
  clad::jacobian(f_dec_in_expr);
  clad::jacobian(f_add_assign);
  clad::jacobian(f_rebind);
  clad::jacobian(f_write_through);
  clad::differentiate<clad::opts::vector_mode>(g_vector, "arr, w");
}
