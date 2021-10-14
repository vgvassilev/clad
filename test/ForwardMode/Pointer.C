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
// CHECK-NEXT:     double *_d_p = new double [2](/*implicit*/(double{{ ?}}[2])0);
// CHECK-NEXT:     double *p = new double [2](/*implicit*/(double{{ ?}}[2])0);
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

double fn5(double i, double j) {
  double arr[2] = {7*i, 9*i};
  *(arr + 1) += 11*i;
  int idx1 = 0, idx2 = 1;
  double *p = arr;
  p += 1;
  p = 0 + p;
  *p += 13*i;
  *(p-1) += 17*i;
  return *(arr+idx1) + *(arr+idx2);
}

// CHECK: double fn5_darg0(double i, double j) {
// CHECK-NEXT:     double _d_i = 1;
// CHECK-NEXT:     double _d_j = 0;
// CHECK-NEXT:     double _d_arr[2] = {0 * i + 7 * _d_i, 0 * i + 9 * _d_i};
// CHECK-NEXT:     double arr[2] = {7 * i, 9 * i};
// CHECK-NEXT:     *(_d_arr + 1) += 0 * i + 11 * _d_i;
// CHECK-NEXT:     *(arr + 1) += 11 * i;
// CHECK-NEXT:     int _d_idx1 = 0, _d_idx2 = 0;
// CHECK-NEXT:     int idx1 = 0, idx2 = 1;
// CHECK-NEXT:     double *_d_p = _d_arr;
// CHECK-NEXT:     double *p = arr;
// CHECK-NEXT:     _d_p += 1;
// CHECK-NEXT:     p += 1;
// CHECK-NEXT:     _d_p = 0 + _d_p;
// CHECK-NEXT:     p = 0 + p;
// CHECK-NEXT:     *_d_p += 0 * i + 13 * _d_i;
// CHECK-NEXT:     *p += 13 * i;
// CHECK-NEXT:     *(_d_p - 1) += 0 * i + 17 * _d_i;
// CHECK-NEXT:     *(p - 1) += 17 * i;
// CHECK-NEXT:     return *(_d_arr + idx1) + *(_d_arr + idx2);
// CHECK-NEXT: }

int main() {
  INIT_DIFFERENTIATE(fn1, "i");
  INIT_DIFFERENTIATE(fn2, "i");
  INIT_DIFFERENTIATE(fn3, "i");
  INIT_DIFFERENTIATE(fn4, "i");
  INIT_DIFFERENTIATE(fn5, "i");

  TEST_DIFFERENTIATE(fn1, 3, 5);  // CHECK-EXEC: {5.00}
  TEST_DIFFERENTIATE(fn2, 3, 5);  // CHECK-EXEC: {5.00}
  TEST_DIFFERENTIATE(fn3, 3, 5);  // CHECK-EXEC: {6.00}
  TEST_DIFFERENTIATE(fn4, 3, 5);  // CHECK-EXEC: {16.00}
  TEST_DIFFERENTIATE(fn5, 3, 5);  // CHECK-EXEC: {57.00}
}
