// RUN: %cladclang %s -lstdc++ -I%S/../../include -oPointer.out 2>&1 | FileCheck %s
// RUN: ./Pointer.out | FileCheck -check-prefix=CHECK-EXEC %s
// CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

#include "../TestUtils.h"

double fn1(double i, double j) {
  double *p = &i;
  double **q = new double *;
  *q = new double;
  **q = j;
  return *p * **q;
}

// CHECK: double fn1_darg0(double i, double j) {
// CHECK-NEXT:     double _d_i = 1;
// CHECK-NEXT:     double _d_j = 0;
// CHECK-NEXT:     double *_d_p = &_d_i;
// CHECK-NEXT:     double *p = &i;
// CHECK-NEXT:     double **_d_q = new double *(/*implicit*/(double *)0);
// CHECK-NEXT:     double **q = new double *(/*implicit*/(double *)0);
// CHECK-NEXT:     *_d_q = new double(/*implicit*/(double)0);
// CHECK-NEXT:     *q = new double(/*implicit*/(double)0);
// CHECK-NEXT:     **_d_q = _d_j;
// CHECK-NEXT:     **q = j;
// CHECK-NEXT:     return *_d_p * **q + *p * **_d_q;
// CHECK-NEXT: }

double fn2(double i, double j) {
  return *(&i) * *(&(*(&j)));
}

// CHECK: double fn2_darg0(double i, double j) {
// CHECK-NEXT:     double _d_i = 1;
// CHECK-NEXT:     double _d_j = 0;
// CHECK-NEXT:     return *&_d_i * *&*&j + *&i * *&*&_d_j;
// CHECK-NEXT: }

double fn3(double i, double j) {
  double *p = new double[2];
  p[0] = i + j;
  p[1] = i*j;
  return p[0] + p[1];
}

// CHECK: double fn3_darg0(double i, double j) {
// CHECK-NEXT:     double _d_i = 1;
// CHECK-NEXT:     double _d_j = 0;
// CHECK-NEXT:     double *_d_p = new double [2](/*implicit*/(double [2])0);
// CHECK-NEXT:     double *p = new double [2](/*implicit*/(double [2])0);
// CHECK-NEXT:     _d_p[0] = _d_i + _d_j;
// CHECK-NEXT:     p[0] = i + j;
// CHECK-NEXT:     _d_p[1] = _d_i * j + i * _d_j;
// CHECK-NEXT:     p[1] = i * j;
// CHECK-NEXT:     return _d_p[0] + _d_p[1];
// CHECK-NEXT: }

double fn4(double i, double j) {
  double *p = new double(7*i + j);
  double q = *p + 9*i;
  delete p;
  return q;
}

// CHECK: double fn4_darg0(double i, double j) {
// CHECK-NEXT:     double _d_i = 1;
// CHECK-NEXT:     double _d_j = 0;
// CHECK-NEXT:     double *_d_p = new double(0 * i + 7 * _d_i + _d_j);
// CHECK-NEXT:     double *p = new double(7 * i + j);
// CHECK-NEXT:     double _d_q = *_d_p + 0 * i + 9 * _d_i;
// CHECK-NEXT:     double q = *p + 9 * i;
// CHECK-NEXT:     delete _d_p;
// CHECK-NEXT:     delete p;
// CHECK-NEXT:     return _d_q;
// CHECK-NEXT: }

int main() {
  INIT_DIFFERENTIATE(fn1, "i");
  INIT_DIFFERENTIATE(fn2, "i");
  INIT_DIFFERENTIATE(fn3, "i");
  INIT_DIFFERENTIATE(fn4, "i");

  TEST_DIFFERENTIATE(fn1, 3, 5);  // CHECK-EXEC: {5.00}
  TEST_DIFFERENTIATE(fn2, 3, 5);  // CHECK-EXEC: {5.00}
  TEST_DIFFERENTIATE(fn3, 3, 5);  // CHECK-EXEC: {6.00}
  TEST_DIFFERENTIATE(fn4, 3, 5);  // CHECK-EXEC: {16.00}
}
