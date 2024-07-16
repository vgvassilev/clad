// RUN: %cladclang %s -I%S/../../include -oPointer.out 2>&1 | %filecheck %s
// RUN: ./Pointer.out | %filecheck_exec %s
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
// CHECK-NEXT:     double **_d_q = new double *;
// CHECK-NEXT:     double **q = new double *;
// CHECK-NEXT:     *_d_q = new double;
// CHECK-NEXT:     *q = new double;
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
// CHECK-NEXT:     double *_d_p = new double [2];
// CHECK-NEXT:     double *p = new double [2];
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

struct T {
  double i;
  int j;
};

double fn6 (double i) {
  T* t = new T{i};
  double res = t->i;
  delete t;
  return res;
}

// CHECK: double fn6_darg0(double i) {
// CHECK-NEXT:     double _d_i = 1;
// CHECK-NEXT:     T *_d_t = new T({_d_i, /*implicit*/(int)0});
// CHECK-NEXT:     T *t = new T({i, /*implicit*/(int)0});
// CHECK-NEXT:     double _d_res = _d_t->i;
// CHECK-NEXT:     double res = t->i;
// CHECK-NEXT:     delete _d_t;
// CHECK-NEXT:     delete t;
// CHECK-NEXT:     return _d_res;
// CHECK-NEXT: }

double fn7(double i) {
  double *p = (double*)malloc(8UL /*sizeof(double)*/);
  *p = i;
  T *t = (T*)calloc(1, sizeof(T));
  t->i = i;
  double res = *p + t->i;
  p = (double*)realloc(p, 2*sizeof(double));
  p[1] = 2*i;
  res += p[1];
  free(t);
  free(p);
  return res;
}

// CHECK: double fn7_darg0(double i) {
// CHECK-NEXT:     double _d_i = 1;
// CHECK-NEXT:     {{(clad::)?}}ValueAndPushforward<void *, void *> _t0 = clad::custom_derivatives::malloc_pushforward(8UL, 0UL);
// CHECK-NEXT:     double *_d_p = (double *)_t0.pushforward;
// CHECK-NEXT:     double *p = (double *)_t0.value;
// CHECK-NEXT:     *_d_p = _d_i;
// CHECK-NEXT:     *p = i;
// CHECK-NEXT:     {{(clad::)?}}ValueAndPushforward<void *, void *> _t1 = clad::custom_derivatives::calloc_pushforward(1, sizeof(T), 0, sizeof(T));
// CHECK-NEXT:     T *_d_t = (T *)_t1.pushforward;
// CHECK-NEXT:     T *t = (T *)_t1.value;
// CHECK-NEXT:     _d_t->i = _d_i;
// CHECK-NEXT:     t->i = i;
// CHECK-NEXT:     double _d_res = *_d_p + _d_t->i;
// CHECK-NEXT:     double res = *p + t->i;
// CHECK-NEXT:     unsigned {{(int|long)}} _t2 = sizeof(double);
// CHECK-NEXT:     {{(clad::)?}}ValueAndPushforward<void *, void *> _t3 = clad::custom_derivatives::realloc_pushforward(p, 2 * _t2, _d_p, 0 * _t2 + 2 * sizeof(double));
// CHECK-NEXT:     _d_p = (double *)_t3.pushforward;
// CHECK-NEXT:     p = (double *)_t3.value;
// CHECK-NEXT:     _d_p[1] = 0 * i + 2 * _d_i;
// CHECK-NEXT:     p[1] = 2 * i;
// CHECK-NEXT:     _d_res += _d_p[1];
// CHECK-NEXT:     res += p[1];
// CHECK-NEXT:     clad::custom_derivatives::free_pushforward(t, _d_t);
// CHECK-NEXT:     clad::custom_derivatives::free_pushforward(p, _d_p);
// CHECK-NEXT:     return _d_res;
// CHECK-NEXT: }

void* cling_runtime_internal_throwIfInvalidPointer(void *Sema, void *Expr, const void *Arg) {
  return const_cast<void*>(Arg);
}

double fn8(double* params) {
  double arr[] = {3.0};
  return params[0]*params[0] + *(double*)(cling_runtime_internal_throwIfInvalidPointer((void*)0UL, (void*)0UL, arr));
}

// CHECK: clad::ValueAndPushforward<void *, void *> cling_runtime_internal_throwIfInvalidPointer_pushforward(void *Sema, void *Expr, const void *Arg, void *_d_Sema, void *_d_Expr, const void *_d_Arg);

// CHECK: double fn8_darg0_0(double *params) {
// CHECK-NEXT:     double _d_arr[1] = {0.};
// CHECK-NEXT:     double arr[1] = {3.};
// CHECK-NEXT:     clad::ValueAndPushforward<void *, void *> _t0 = cling_runtime_internal_throwIfInvalidPointer_pushforward((void *)0UL, (void *)0UL, arr, (void *)0UL, (void *)0UL, _d_arr);
// CHECK-NEXT:     return 1. * params[0] + params[0] * 1. + *(double *)_t0.pushforward;
// CHECK-NEXT: }

double fn9(double* params, const double *constants) {
  double c0 = *constants;
  return params[0] * c0;
}

// CHECK: double fn9_darg0_0(double *params, const double *constants) {
// CHECK-NEXT:     double _d_c0 = 0;
// CHECK-NEXT:     double c0 = *constants;
// CHECK-NEXT:     return 1. * c0 + params[0] * _d_c0;
// CHECK-NEXT: }

double fn10(double *params, const double *constants) {
  double c0 = *(constants + 0);
  return params[0] * c0;
}

// CHECK: double fn10_darg0_0(double *params, const double *constants) {
// CHECK-NEXT:     double _d_c0 = 0;
// CHECK-NEXT:     double c0 = *(constants + 0);
// CHECK-NEXT:     return 1. * c0 + params[0] * _d_c0;
// CHECK-NEXT: }

int main() {
  INIT_DIFFERENTIATE(fn1, "i");
  INIT_DIFFERENTIATE(fn2, "i");
  INIT_DIFFERENTIATE(fn3, "i");
  INIT_DIFFERENTIATE(fn4, "i");
  INIT_DIFFERENTIATE(fn5, "i");
  INIT_DIFFERENTIATE(fn6, "i");
  INIT_DIFFERENTIATE(fn7, "i");

  TEST_DIFFERENTIATE(fn1, 3, 5);  // CHECK-EXEC: {5.00}
  TEST_DIFFERENTIATE(fn2, 3, 5);  // CHECK-EXEC: {5.00}
  TEST_DIFFERENTIATE(fn3, 3, 5);  // CHECK-EXEC: {6.00}
  TEST_DIFFERENTIATE(fn4, 3, 5);  // CHECK-EXEC: {16.00}
  TEST_DIFFERENTIATE(fn5, 3, 5);  // CHECK-EXEC: {57.00}
  TEST_DIFFERENTIATE(fn6, 3);     // CHECK-EXEC: {1.00}
  TEST_DIFFERENTIATE(fn7, 3);     // CHECK-EXEC: {4.00}

  double params[] = {3.0};
  auto fn8_dx = clad::differentiate(fn8, "params[0]");
  double d_param = fn8_dx.execute(params);
  printf("{%.2f}\n", d_param); // CHECK-EXEC: {6.00}

  double constants[] = {5.0};
  auto fn9_dx = clad::differentiate(fn9, "params[0]");
  d_param = fn9_dx.execute(params, constants);
  printf("{%.2f}\n", d_param); // CHECK-EXEC: {5.00}

  auto fn10_dx = clad::differentiate(fn10, "params[0]");
  d_param = fn10_dx.execute(params, constants);
  printf("{%.2f}\n", d_param); // CHECK-EXEC: {5.00}
}

// CHECK: clad::ValueAndPushforward<void *, void *> cling_runtime_internal_throwIfInvalidPointer_pushforward(void *Sema, void *Expr, const void *Arg, void *_d_Sema, void *_d_Expr, const void *_d_Arg) {
// CHECK-NEXT:     return {const_cast<void *>(Arg), const_cast<void *>(_d_Arg)};
// CHECK-NEXT: }
