// RUN: %cladclang %s -I%S/../../include 2>&1 | %filecheck %s
//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

struct Complex {
    Complex(double preal, double pim) : real(preal), im(pim) {}
    double real = 0, im = 0;
};

Complex c(7, 9);

double fn_complex(double i, double j) {
    return c.real * i + c.im * j;
}

// CHECK: double fn_complex_darg0(double i, double j) {
// CHECK-NEXT: double _d_i = 1;
// CHECK-NEXT: double _d_j = 0;
// CHECK-NEXT: double &_t0 = c.real;
// CHECK-NEXT: double &_t1 = c.im;
// CHECK-NEXT: return 0. * i + _t0 * _d_i + 0. * j + _t1 * _d_j;
// CHECK-NEXT: }

struct Array {
    double data[5];
};

Array array;

double fn_array(double i, double j) {
    return array.data[0] * i + array.data[1] * j;
}

// CHECK: double fn_array_darg0(double i, double j) {
// CHECK-NEXT: double _d_i = 1;
// CHECK-NEXT: double _d_j = 0;
// CHECK-NEXT: double &_t0 = array.data[0];
// CHECK-NEXT: double &_t1 = array.data[1];
// CHECK-NEXT: return 0 * i + _t0 * _d_i + 0 * j + _t1 * _d_j;
// CHECK-NEXT: }

int main () {
  clad::differentiate(fn_complex, "i");
  clad::differentiate(fn_array, "i");
  return 0;
}
