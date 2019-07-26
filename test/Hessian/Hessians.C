// RUN: %cladclang %s -lm -I%S/../../include -oHessians.out 2>&1 | FileCheck %s
// RUN: ./Hessians.out | FileCheck -check-prefix=CHECK-EXEC %s

//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

double f_cubed_add1(double a, double b) {
  return a * a * a + b * b * b;
}

void f_cubed_add1_darg0_grad(double a, double b, double *_result);
//CHECK:void f_cubed_add1_darg0_grad(double a, double b, double *_result) {
//CHECK-NEXT:    double _d__d_a = 0;
//CHECK-NEXT:    double _d__d_b = 0;
//CHECK-NEXT:    double _t0;
//CHECK-NEXT:    double _t1;
//CHECK-NEXT:    double _d__t0 = 0;
//CHECK-NEXT:    double _t2;
//CHECK-NEXT:    double _t3;
//CHECK-NEXT:    double _d__t1 = 0;
//CHECK-NEXT:    double _t4;
//CHECK-NEXT:    double _t5;
//CHECK-NEXT:    double _t6;
//CHECK-NEXT:    double _t7;
//CHECK-NEXT:    double _t8;
//CHECK-NEXT:    double _t9;
//CHECK-NEXT:    double _t10;
//CHECK-NEXT:    double _t11;
//CHECK-NEXT:    double _t12;
//CHECK-NEXT:    double _t13;
//CHECK-NEXT:    double _t14;
//CHECK-NEXT:    double _t15;
//CHECK-NEXT:    double _t16;
//CHECK-NEXT:    double _t17;
//CHECK-NEXT:    double _t18;
//CHECK-NEXT:    double _t19;
//CHECK-NEXT:    double _d_a = 1;
//CHECK-NEXT:    double _d_b = 0;
//CHECK-NEXT:    _t1 = a;
//CHECK-NEXT:    _t0 = a;
//CHECK-NEXT:    double _t00 = _t1 * _t0;
//CHECK-NEXT:    _t3 = b;
//CHECK-NEXT:    _t2 = b;
//CHECK-NEXT:    double _t10 = _t3 * _t2;
//CHECK-NEXT:    _t6 = _d_a;
//CHECK-NEXT:    _t5 = a;
//CHECK-NEXT:    _t8 = a;
//CHECK-NEXT:    _t7 = _d_a;
//CHECK-NEXT:    _t9 = (_t6 * _t5 + _t8 * _t7);
//CHECK-NEXT:    _t4 = a;
//CHECK-NEXT:    _t11 = _t00;
//CHECK-NEXT:    _t10 = _d_a;
//CHECK-NEXT:    _t14 = _d_b;
//CHECK-NEXT:    _t13 = b;
//CHECK-NEXT:    _t16 = b;
//CHECK-NEXT:    _t15 = _d_b;
//CHECK-NEXT:    _t17 = (_t14 * _t13 + _t16 * _t15);
//CHECK-NEXT:    _t12 = b;
//CHECK-NEXT:    _t19 = _t10;
//CHECK-NEXT:    _t18 = _d_b;
//CHECK-NEXT:    double f_cubed_add1_darg0_return = _t9 * _t4 + _t11 * _t10 + _t17 * _t12 + _t19 * _t18;
//CHECK-NEXT:    goto _label0;
//CHECK-NEXT:  _label0:
//CHECK-NEXT:    {
//CHECK-NEXT:        double _r4 = 1 * _t4;
//CHECK-NEXT:        double _r5 = _r4 * _t5;
//CHECK-NEXT:        _d__d_a += _r5;
//CHECK-NEXT:        double _r6 = _t6 * _r4;
//CHECK-NEXT:        _result[0UL] += _r6;
//CHECK-NEXT:        double _r7 = _r4 * _t7;
//CHECK-NEXT:        _result[0UL] += _r7;
//CHECK-NEXT:        double _r8 = _t8 * _r4;
//CHECK-NEXT:        _d__d_a += _r8;
//CHECK-NEXT:        double _r9 = _t9 * 1;
//CHECK-NEXT:        _result[0UL] += _r9;
//CHECK-NEXT:        double _r10 = 1 * _t10;
//CHECK-NEXT:        _d__t0 += _r10;
//CHECK-NEXT:        double _r11 = _t11 * 1;
//CHECK-NEXT:        _d__d_a += _r11;
//CHECK-NEXT:        double _r12 = 1 * _t12;
//CHECK-NEXT:        double _r13 = _r12 * _t13;
//CHECK-NEXT:        _d__d_b += _r13;
//CHECK-NEXT:        double _r14 = _t14 * _r12;
//CHECK-NEXT:        _result[1UL] += _r14;
//CHECK-NEXT:        double _r15 = _r12 * _t15;
//CHECK-NEXT:        _result[1UL] += _r15;
//CHECK-NEXT:        double _r16 = _t16 * _r12;
//CHECK-NEXT:        _d__d_b += _r16;
//CHECK-NEXT:        double _r17 = _t17 * 1;
//CHECK-NEXT:        _result[1UL] += _r17;
//CHECK-NEXT:        double _r18 = 1 * _t18;
//CHECK-NEXT:        _d__t1 += _r18;
//CHECK-NEXT:        double _r19 = _t19 * 1;
//CHECK-NEXT:        _d__d_b += _r19;
//CHECK-NEXT:    }
//CHECK-NEXT:    {
//CHECK-NEXT:        double _r2 = _d__t1 * _t2;
//CHECK-NEXT:        _result[1UL] += _r2;
//CHECK-NEXT:        double _r3 = _t3 * _d__t1;
//CHECK-NEXT:        _result[1UL] += _r3;
//CHECK-NEXT:    }
//CHECK-NEXT:    {
//CHECK-NEXT:        double _r0 = _d__t0 * _t0;
//CHECK-NEXT:        _result[0UL] += _r0;
//CHECK-NEXT:        double _r1 = _t1 * _d__t0;
//CHECK-NEXT:        _result[0UL] += _r1;
//CHECK-NEXT:    }
//CHECK-NEXT:}


void f_cubed_add1_darg1_grad(double a, double b, double *_result);
//CHECK:void f_cubed_add1_darg1_grad(double a, double b, double *_result) {
//CHECK-NEXT:    double _d__d_a = 0;
//CHECK-NEXT:    double _d__d_b = 0;
//CHECK-NEXT:    double _t0;
//CHECK-NEXT:    double _t1;
//CHECK-NEXT:    double _d__t0 = 0;
//CHECK-NEXT:    double _t2;
//CHECK-NEXT:    double _t3;
//CHECK-NEXT:    double _d__t1 = 0;
//CHECK-NEXT:    double _t4;
//CHECK-NEXT:    double _t5;
//CHECK-NEXT:    double _t6;
//CHECK-NEXT:    double _t7;
//CHECK-NEXT:    double _t8;
//CHECK-NEXT:    double _t9;
//CHECK-NEXT:    double _t10;
//CHECK-NEXT:    double _t11;
//CHECK-NEXT:    double _t12;
//CHECK-NEXT:    double _t13;
//CHECK-NEXT:    double _t14;
//CHECK-NEXT:    double _t15;
//CHECK-NEXT:    double _t16;
//CHECK-NEXT:    double _t17;
//CHECK-NEXT:    double _t18;
//CHECK-NEXT:    double _t19;
//CHECK-NEXT:    double _d_a = 0;
//CHECK-NEXT:    double _d_b = 1;
//CHECK-NEXT:    _t1 = a;
//CHECK-NEXT:    _t0 = a;
//CHECK-NEXT:    double _t00 = _t1 * _t0;
//CHECK-NEXT:    _t3 = b;
//CHECK-NEXT:    _t2 = b;
//CHECK-NEXT:    double _t10 = _t3 * _t2;
//CHECK-NEXT:    _t6 = _d_a;
//CHECK-NEXT:    _t5 = a;
//CHECK-NEXT:    _t8 = a;
//CHECK-NEXT:    _t7 = _d_a;
//CHECK-NEXT:    _t9 = (_t6 * _t5 + _t8 * _t7);
//CHECK-NEXT:    _t4 = a;
//CHECK-NEXT:    _t11 = _t00;
//CHECK-NEXT:    _t10 = _d_a;
//CHECK-NEXT:    _t14 = _d_b;
//CHECK-NEXT:    _t13 = b;
//CHECK-NEXT:    _t16 = b;
//CHECK-NEXT:    _t15 = _d_b;
//CHECK-NEXT:    _t17 = (_t14 * _t13 + _t16 * _t15);
//CHECK-NEXT:    _t12 = b;
//CHECK-NEXT:    _t19 = _t10;
//CHECK-NEXT:    _t18 = _d_b;
//CHECK-NEXT:    double f_cubed_add1_darg1_return = _t9 * _t4 + _t11 * _t10 + _t17 * _t12 + _t19 * _t18;
//CHECK-NEXT:    goto _label0;
//CHECK-NEXT:  _label0:
//CHECK-NEXT:    {
//CHECK-NEXT:        double _r4 = 1 * _t4;
//CHECK-NEXT:        double _r5 = _r4 * _t5;
//CHECK-NEXT:        _d__d_a += _r5;
//CHECK-NEXT:        double _r6 = _t6 * _r4;
//CHECK-NEXT:        _result[0UL] += _r6;
//CHECK-NEXT:        double _r7 = _r4 * _t7;
//CHECK-NEXT:        _result[0UL] += _r7;
//CHECK-NEXT:        double _r8 = _t8 * _r4;
//CHECK-NEXT:        _d__d_a += _r8;
//CHECK-NEXT:        double _r9 = _t9 * 1;
//CHECK-NEXT:        _result[0UL] += _r9;
//CHECK-NEXT:        double _r10 = 1 * _t10;
//CHECK-NEXT:        _d__t0 += _r10;
//CHECK-NEXT:        double _r11 = _t11 * 1;
//CHECK-NEXT:        _d__d_a += _r11;
//CHECK-NEXT:        double _r12 = 1 * _t12;
//CHECK-NEXT:        double _r13 = _r12 * _t13;
//CHECK-NEXT:        _d__d_b += _r13;
//CHECK-NEXT:        double _r14 = _t14 * _r12;
//CHECK-NEXT:        _result[1UL] += _r14;
//CHECK-NEXT:        double _r15 = _r12 * _t15;
//CHECK-NEXT:        _result[1UL] += _r15;
//CHECK-NEXT:        double _r16 = _t16 * _r12;
//CHECK-NEXT:        _d__d_b += _r16;
//CHECK-NEXT:        double _r17 = _t17 * 1;
//CHECK-NEXT:        _result[1UL] += _r17;
//CHECK-NEXT:        double _r18 = 1 * _t18;
//CHECK-NEXT:        _d__t1 += _r18;
//CHECK-NEXT:        double _r19 = _t19 * 1;
//CHECK-NEXT:        _d__d_b += _r19;
//CHECK-NEXT:    }
//CHECK-NEXT:    {
//CHECK-NEXT:        double _r2 = _d__t1 * _t2;
//CHECK-NEXT:        _result[1UL] += _r2;
//CHECK-NEXT:        double _r3 = _t3 * _d__t1;
//CHECK-NEXT:        _result[1UL] += _r3;
//CHECK-NEXT:    }
//CHECK-NEXT:    {
//CHECK-NEXT:        double _r0 = _d__t0 * _t0;
//CHECK-NEXT:        _result[0UL] += _r0;
//CHECK-NEXT:        double _r1 = _t1 * _d__t0;
//CHECK-NEXT:        _result[0UL] += _r1;
//CHECK-NEXT:    }
//CHECK-NEXT:}



void f_cubed_add1_hessian(double a, double b, double *hessianMatrix);
//CHECK: void f_cubed_add1_hessian(double a, double b, double *hessianMatrix) {
//CHECK-NEXT:    f_cubed_add1_darg0_grad(a, b, &hessianMatrix[0UL]);
//CHECK-NEXT:    f_cubed_add1_darg1_grad(a, b, &hessianMatrix[2UL]);
//CHECK-NEXT: }

double f_suvat1(double u, double t) {
  return ((u * t) + ((0.5) * (9.81) * (t * t)));
}

void f_suvat1_darg0_grad(double u, double t, double *_result);
void f_suvat1_darg1_grad(double u, double t, double *_result);

void f_suvat1_hessian(double u, double t, double *hessianMatrix);
//CHECK:void f_suvat1_hessian(double u, double t, double *hessianMatrix) {
//CHECK-NEXT:    f_suvat1_darg0_grad(u, t, &hessianMatrix[0UL]);
//CHECK-NEXT:    f_suvat1_darg1_grad(u, t, &hessianMatrix[2UL]);
//CHECK-NEXT:}

double f_cond3(double x, double c) {
  if (c > 0) {
    return (x*x*x) + (c*c*c);
  }
  else {
    return (x*x*x) - (c*c*c);
  }
}

void f_cond3_darg0_grad(double x, double c, double *_result);
void f_cond3_darg1_grad(double x, double c, double *_result);

void f_cond3_hessian(double x, double c, double *hessianMatrix);
//CHECK:void f_cond3_hessian(double x, double c, double *hessianMatrix) {
//CHECK-NEXT:    f_cond3_darg0_grad(x, c, &hessianMatrix[0UL]);
//CHECK-NEXT:    f_cond3_darg1_grad(x, c, &hessianMatrix[2UL]);
//CHECK-NEXT:}

double f_power10(double x) {
  return x * x * x * x * x * x * x * x * x * x;
}

void f_power10_darg0_grad(double x, double *_result);

void f_power10_hessian(double x, double *hessianMatrix);
//CHECK: void f_power10_hessian(double x, double *hessianMatrix) {
//CHECK-NEXT:     f_power10_darg0_grad(x, &hessianMatrix[0UL]);
//CHECK-NEXT: }



#define TEST1(F, x) { \
  result[0] = 0;\
  auto h = clad::hessian(F);\
  h.execute(x, result);\
  printf("Result is = {%.2f}\n", result[0]); \
  F##_hessian(x, result);\
  F##_darg0_grad(x, result);\
}

#define TEST2(F, x, y) { \
  result[0] = 0; result[1] = 0;; result[2] = 0; result[3] = 0;\
  auto h = clad::hessian(F);\
  h.execute(x, y, result);\
  printf("Result is = {%.2f, %.2f, %.2f, %.2f}\n", result[0], result[1], result[2], result[3]); \
  F##_darg0_grad(x, y, result);\
  F##_darg1_grad(x, y, result);\
  F##_hessian(x, y, result);\
}

int main() {
  double result[10];
  TEST2(f_cubed_add1, 1, 2); // CHECK-EXEC: Result is = {6.00, 0.00, 0.00, 12.00}
  TEST2(f_suvat1, 1, 2); // CHECK-EXEC: Result is = {0.00, 1.00, 1.00, 9.81}
  TEST2(f_cond3, 5, -2); // CHECK-EXEC: Result is = {30.00, 0.00, 0.00, 12.00}
  TEST1(f_power10, 5); // CHECK-EXEC: Result is = {35156250.00}
}
