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
//CHECK:void f_suvat1_darg0_grad(double u, double t, double *_result) {
//CHECK-NEXT:    double _d__d_u = 0;
//CHECK-NEXT:    double _d__d_t = 0;
//CHECK-NEXT:    double _d__t0 = 0;
//CHECK-NEXT:    double _t0;
//CHECK-NEXT:    double _t1;
//CHECK-NEXT:    double _d__t1 = 0;
//CHECK-NEXT:    double _t2;
//CHECK-NEXT:    double _t3;
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
//CHECK-NEXT:    double _d_u = 1;
//CHECK-NEXT:    double _d_t = 0;
//CHECK-NEXT:    double _t00 = 0.5 * 9.8100000000000004;
//CHECK-NEXT:    _t1 = t;
//CHECK-NEXT:    _t0 = t;
//CHECK-NEXT:    double _t10 = (_t1 * _t0);
//CHECK-NEXT:    _t3 = _d_u;
//CHECK-NEXT:    _t2 = t;
//CHECK-NEXT:    _t5 = u;
//CHECK-NEXT:    _t4 = _d_t;
//CHECK-NEXT:    _t7 = (0. * 9.8100000000000004 + 0.5 * 0.);
//CHECK-NEXT:    _t6 = _t10;
//CHECK-NEXT:    _t9 = _t00;
//CHECK-NEXT:    _t11 = _d_t;
//CHECK-NEXT:    _t10 = t;
//CHECK-NEXT:    _t13 = t;
//CHECK-NEXT:    _t12 = _d_t;
//CHECK-NEXT:    _t8 = (_t11 * _t10 + _t13 * _t12);
//CHECK-NEXT:    double f_suvat1_darg0_return = ((_t3 * _t2 + _t5 * _t4) + (_t7 * _t6 + _t9 * _t8));
//CHECK-NEXT:    goto _label0;
//CHECK-NEXT:  _label0:
//CHECK-NEXT:    {
//CHECK-NEXT:        double _r3 = 1 * _t2;
//CHECK-NEXT:        _d__d_u += _r3;
//CHECK-NEXT:        double _r4 = _t3 * 1;
//CHECK-NEXT:        _result[1UL] += _r4;
//CHECK-NEXT:        double _r5 = 1 * _t4;
//CHECK-NEXT:        _result[0UL] += _r5;
//CHECK-NEXT:        double _r6 = _t5 * 1;
//CHECK-NEXT:        _d__d_t += _r6;
//CHECK-NEXT:        double _r7 = 1 * _t6;
//CHECK-NEXT:        double _r8 = _r7 * 9.8100000000000004;
//CHECK-NEXT:        double _r9 = _r7 * 0.;
//CHECK-NEXT:        double _r10 = _t7 * 1;
//CHECK-NEXT:        _d__t1 += _r10;
//CHECK-NEXT:        double _r11 = 1 * _t8;
//CHECK-NEXT:        _d__t0 += _r11;
//CHECK-NEXT:        double _r12 = _t9 * 1;
//CHECK-NEXT:        double _r13 = _r12 * _t10;
//CHECK-NEXT:        _d__d_t += _r13;
//CHECK-NEXT:        double _r14 = _t11 * _r12;
//CHECK-NEXT:        _result[1UL] += _r14;
//CHECK-NEXT:        double _r15 = _r12 * _t12;
//CHECK-NEXT:        _result[1UL] += _r15;
//CHECK-NEXT:        double _r16 = _t13 * _r12;
//CHECK-NEXT:        _d__d_t += _r16;
//CHECK-NEXT:    }
//CHECK-NEXT:    {
//CHECK-NEXT:        double _r1 = _d__t1 * _t0;
//CHECK-NEXT:        _result[1UL] += _r1;
//CHECK-NEXT:        double _r2 = _t1 * _d__t1;
//CHECK-NEXT:        _result[1UL] += _r2;
//CHECK-NEXT:    }
//CHECK-NEXT:    double _r0 = _d__t0 * 9.8100000000000004;
//CHECK-NEXT:}


void f_suvat1_darg1_grad(double u, double t, double *_result);
//CHECK:void f_suvat1_darg1_grad(double u, double t, double *_result) {
//CHECK-NEXT:    double _d__d_u = 0;
//CHECK-NEXT:    double _d__d_t = 0;
//CHECK-NEXT:    double _d__t0 = 0;
//CHECK-NEXT:    double _t0;
//CHECK-NEXT:    double _t1;
//CHECK-NEXT:    double _d__t1 = 0;
//CHECK-NEXT:    double _t2;
//CHECK-NEXT:    double _t3;
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
//CHECK-NEXT:    double _d_u = 0;
//CHECK-NEXT:    double _d_t = 1;
//CHECK-NEXT:    double _t00 = 0.5 * 9.8100000000000004;
//CHECK-NEXT:    _t1 = t;
//CHECK-NEXT:    _t0 = t;
//CHECK-NEXT:    double _t10 = (_t1 * _t0);
//CHECK-NEXT:    _t3 = _d_u;
//CHECK-NEXT:    _t2 = t;
//CHECK-NEXT:    _t5 = u;
//CHECK-NEXT:    _t4 = _d_t;
//CHECK-NEXT:    _t7 = (0. * 9.8100000000000004 + 0.5 * 0.);
//CHECK-NEXT:    _t6 = _t10;
//CHECK-NEXT:    _t9 = _t00;
//CHECK-NEXT:    _t11 = _d_t;
//CHECK-NEXT:    _t10 = t;
//CHECK-NEXT:    _t13 = t;
//CHECK-NEXT:    _t12 = _d_t;
//CHECK-NEXT:    _t8 = (_t11 * _t10 + _t13 * _t12);
//CHECK-NEXT:    double f_suvat1_darg1_return = ((_t3 * _t2 + _t5 * _t4) + (_t7 * _t6 + _t9 * _t8));
//CHECK-NEXT:    goto _label0;
//CHECK-NEXT:  _label0:
//CHECK-NEXT:    {
//CHECK-NEXT:        double _r3 = 1 * _t2;
//CHECK-NEXT:        _d__d_u += _r3;
//CHECK-NEXT:        double _r4 = _t3 * 1;
//CHECK-NEXT:        _result[1UL] += _r4;
//CHECK-NEXT:        double _r5 = 1 * _t4;
//CHECK-NEXT:        _result[0UL] += _r5;
//CHECK-NEXT:        double _r6 = _t5 * 1;
//CHECK-NEXT:        _d__d_t += _r6;
//CHECK-NEXT:        double _r7 = 1 * _t6;
//CHECK-NEXT:        double _r8 = _r7 * 9.8100000000000004;
//CHECK-NEXT:        double _r9 = _r7 * 0.;
//CHECK-NEXT:        double _r10 = _t7 * 1;
//CHECK-NEXT:        _d__t1 += _r10;
//CHECK-NEXT:        double _r11 = 1 * _t8;
//CHECK-NEXT:        _d__t0 += _r11;
//CHECK-NEXT:        double _r12 = _t9 * 1;
//CHECK-NEXT:        double _r13 = _r12 * _t10;
//CHECK-NEXT:        _d__d_t += _r13;
//CHECK-NEXT:        double _r14 = _t11 * _r12;
//CHECK-NEXT:        _result[1UL] += _r14;
//CHECK-NEXT:        double _r15 = _r12 * _t12;
//CHECK-NEXT:        _result[1UL] += _r15;
//CHECK-NEXT:        double _r16 = _t13 * _r12;
//CHECK-NEXT:        _d__d_t += _r16;
//CHECK-NEXT:    }
//CHECK-NEXT:    {
//CHECK-NEXT:        double _r1 = _d__t1 * _t0;
//CHECK-NEXT:        _result[1UL] += _r1;
//CHECK-NEXT:        double _r2 = _t1 * _d__t1;
//CHECK-NEXT:        _result[1UL] += _r2;
//CHECK-NEXT:    }
//CHECK-NEXT:    double _r0 = _d__t0 * 9.8100000000000004;
//CHECK-NEXT:}

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
//CHECK:void f_cond3_darg0_grad(double x, double c, double *_result) {
//CHECK-NEXT:    double _d__d_x = 0;
//CHECK-NEXT:    double _d__d_c = 0;
//CHECK-NEXT:    bool _cond0;
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
//CHECK-NEXT:    double _t20;
//CHECK-NEXT:    double _t21;
//CHECK-NEXT:    double _d__t2 = 0;
//CHECK-NEXT:    double _t22;
//CHECK-NEXT:    double _t23;
//CHECK-NEXT:    double _d__t3 = 0;
//CHECK-NEXT:    double _t24;
//CHECK-NEXT:    double _t25;
//CHECK-NEXT:    double _t26;
//CHECK-NEXT:    double _t27;
//CHECK-NEXT:    double _t28;
//CHECK-NEXT:    double _t29;
//CHECK-NEXT:    double _t30;
//CHECK-NEXT:    double _t31;
//CHECK-NEXT:    double _t32;
//CHECK-NEXT:    double _t33;
//CHECK-NEXT:    double _t34;
//CHECK-NEXT:    double _t35;
//CHECK-NEXT:    double _t36;
//CHECK-NEXT:    double _t37;
//CHECK-NEXT:    double _t38;
//CHECK-NEXT:    double _t39;
//CHECK-NEXT:    double _d_x = 1;
//CHECK-NEXT:    double _d_c = 0;
//CHECK-NEXT:    _cond0 = c > 0;
//CHECK-NEXT:    if (_cond0) {
//CHECK-NEXT:        _t1 = x;
//CHECK-NEXT:        _t0 = x;
//CHECK-NEXT:        double _t00 = _t1 * _t0;
//CHECK-NEXT:        _t3 = c;
//CHECK-NEXT:        _t2 = c;
//CHECK-NEXT:        double _t10 = _t3 * _t2;
//CHECK-NEXT:        _t6 = _d_x;
//CHECK-NEXT:        _t5 = x;
//CHECK-NEXT:        _t8 = x;
//CHECK-NEXT:        _t7 = _d_x;
//CHECK-NEXT:        _t9 = (_t6 * _t5 + _t8 * _t7);
//CHECK-NEXT:        _t4 = x;
//CHECK-NEXT:        _t11 = _t00;
//CHECK-NEXT:        _t10 = _d_x;
//CHECK-NEXT:        _t14 = _d_c;
//CHECK-NEXT:        _t13 = c;
//CHECK-NEXT:        _t16 = c;
//CHECK-NEXT:        _t15 = _d_c;
//CHECK-NEXT:        _t17 = (_t14 * _t13 + _t16 * _t15);
//CHECK-NEXT:        _t12 = c;
//CHECK-NEXT:        _t19 = _t10;
//CHECK-NEXT:        _t18 = _d_c;
//CHECK-NEXT:        double f_cond3_darg0_return = (_t9 * _t4 + _t11 * _t10) + (_t17 * _t12 + _t19 * _t18);
//CHECK-NEXT:        goto _label0;
//CHECK-NEXT:    } else {
//CHECK-NEXT:        _t21 = x;
//CHECK-NEXT:        _t20 = x;
//CHECK-NEXT:        double _t22 = _t21 * _t20;
//CHECK-NEXT:        _t23 = c;
//CHECK-NEXT:        _t22 = c;
//CHECK-NEXT:        double _t30 = _t23 * _t22;
//CHECK-NEXT:        _t26 = _d_x;
//CHECK-NEXT:        _t25 = x;
//CHECK-NEXT:        _t28 = x;
//CHECK-NEXT:        _t27 = _d_x;
//CHECK-NEXT:        _t29 = (_t26 * _t25 + _t28 * _t27);
//CHECK-NEXT:        _t24 = x;
//CHECK-NEXT:        _t31 = _t22;
//CHECK-NEXT:        _t30 = _d_x;
//CHECK-NEXT:        _t34 = _d_c;
//CHECK-NEXT:        _t33 = c;
//CHECK-NEXT:        _t36 = c;
//CHECK-NEXT:        _t35 = _d_c;
//CHECK-NEXT:        _t37 = (_t34 * _t33 + _t36 * _t35);
//CHECK-NEXT:        _t32 = c;
//CHECK-NEXT:        _t39 = _t30;
//CHECK-NEXT:        _t38 = _d_c;
//CHECK-NEXT:        double f_cond3_darg0_return = (_t29 * _t24 + _t31 * _t30) - (_t37 * _t32 + _t39 * _t38);
//CHECK-NEXT:        goto _label1;
//CHECK-NEXT:    }
//CHECK-NEXT:    if (_cond0) {
//CHECK-NEXT:      _label0:
//CHECK-NEXT:        {
//CHECK-NEXT:            double _r4 = 1 * _t4;
//CHECK-NEXT:            double _r5 = _r4 * _t5;
//CHECK-NEXT:            _d__d_x += _r5;
//CHECK-NEXT:            double _r6 = _t6 * _r4;
//CHECK-NEXT:            _result[0UL] += _r6;
//CHECK-NEXT:            double _r7 = _r4 * _t7;
//CHECK-NEXT:            _result[0UL] += _r7;
//CHECK-NEXT:            double _r8 = _t8 * _r4;
//CHECK-NEXT:            _d__d_x += _r8;
//CHECK-NEXT:            double _r9 = _t9 * 1;
//CHECK-NEXT:            _result[0UL] += _r9;
//CHECK-NEXT:            double _r10 = 1 * _t10;
//CHECK-NEXT:            _d__t0 += _r10;
//CHECK-NEXT:            double _r11 = _t11 * 1;
//CHECK-NEXT:            _d__d_x += _r11;
//CHECK-NEXT:            double _r12 = 1 * _t12;
//CHECK-NEXT:            double _r13 = _r12 * _t13;
//CHECK-NEXT:            _d__d_c += _r13;
//CHECK-NEXT:            double _r14 = _t14 * _r12;
//CHECK-NEXT:            _result[1UL] += _r14;
//CHECK-NEXT:            double _r15 = _r12 * _t15;
//CHECK-NEXT:            _result[1UL] += _r15;
//CHECK-NEXT:            double _r16 = _t16 * _r12;
//CHECK-NEXT:            _d__d_c += _r16;
//CHECK-NEXT:            double _r17 = _t17 * 1;
//CHECK-NEXT:            _result[1UL] += _r17;
//CHECK-NEXT:            double _r18 = 1 * _t18;
//CHECK-NEXT:            _d__t1 += _r18;
//CHECK-NEXT:            double _r19 = _t19 * 1;
//CHECK-NEXT:            _d__d_c += _r19;
//CHECK-NEXT:        }
//CHECK-NEXT:        {
//CHECK-NEXT:            double _r2 = _d__t1 * _t2;
//CHECK-NEXT:            _result[1UL] += _r2;
//CHECK-NEXT:            double _r3 = _t3 * _d__t1;
//CHECK-NEXT:            _result[1UL] += _r3;
//CHECK-NEXT:        }
//CHECK-NEXT:        {
//CHECK-NEXT:            double _r0 = _d__t0 * _t0;
//CHECK-NEXT:            _result[0UL] += _r0;
//CHECK-NEXT:            double _r1 = _t1 * _d__t0;
//CHECK-NEXT:            _result[0UL] += _r1;
//CHECK-NEXT:        }
//CHECK-NEXT:    } else {
//CHECK-NEXT:      _label1:
//CHECK-NEXT:        {
//CHECK-NEXT:            double _r24 = 1 * _t24;
//CHECK-NEXT:            double _r25 = _r24 * _t25;
//CHECK-NEXT:            _d__d_x += _r25;
//CHECK-NEXT:            double _r26 = _t26 * _r24;
//CHECK-NEXT:            _result[0UL] += _r26;
//CHECK-NEXT:            double _r27 = _r24 * _t27;
//CHECK-NEXT:            _result[0UL] += _r27;
//CHECK-NEXT:            double _r28 = _t28 * _r24;
//CHECK-NEXT:            _d__d_x += _r28;
//CHECK-NEXT:            double _r29 = _t29 * 1;
//CHECK-NEXT:            _result[0UL] += _r29;
//CHECK-NEXT:            double _r30 = 1 * _t30;
//CHECK-NEXT:            _d__t2 += _r30;
//CHECK-NEXT:            double _r31 = _t31 * 1;
//CHECK-NEXT:            _d__d_x += _r31;
//CHECK-NEXT:            double _r32 = -1 * _t32;
//CHECK-NEXT:            double _r33 = _r32 * _t33;
//CHECK-NEXT:            _d__d_c += _r33;
//CHECK-NEXT:            double _r34 = _t34 * _r32;
//CHECK-NEXT:            _result[1UL] += _r34;
//CHECK-NEXT:            double _r35 = _r32 * _t35;
//CHECK-NEXT:            _result[1UL] += _r35;
//CHECK-NEXT:            double _r36 = _t36 * _r32;
//CHECK-NEXT:            _d__d_c += _r36;
//CHECK-NEXT:            double _r37 = _t37 * -1;
//CHECK-NEXT:            _result[1UL] += _r37;
//CHECK-NEXT:            double _r38 = -1 * _t38;
//CHECK-NEXT:            _d__t3 += _r38;
//CHECK-NEXT:            double _r39 = _t39 * -1;
//CHECK-NEXT:            _d__d_c += _r39;
//CHECK-NEXT:        }
//CHECK-NEXT:        {
//CHECK-NEXT:            double _r22 = _d__t3 * _t22;
//CHECK-NEXT:            _result[1UL] += _r22;
//CHECK-NEXT:            double _r23 = _t23 * _d__t3;
//CHECK-NEXT:            _result[1UL] += _r23;
//CHECK-NEXT:        }
//CHECK-NEXT:        {
//CHECK-NEXT:            double _r20 = _d__t2 * _t20;
//CHECK-NEXT:            _result[0UL] += _r20;
//CHECK-NEXT:            double _r21 = _t21 * _d__t2;
//CHECK-NEXT:            _result[0UL] += _r21;
//CHECK-NEXT:        }
//CHECK-NEXT:    }
//CHECK-NEXT:}


void f_cond3_darg1_grad(double x, double c, double *_result);
//CHECK:void f_cond3_darg1_grad(double x, double c, double *_result) {
//CHECK-NEXT:    double _d__d_x = 0;
//CHECK-NEXT:    double _d__d_c = 0;
//CHECK-NEXT:    bool _cond0;
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
//CHECK-NEXT:    double _t20;
//CHECK-NEXT:    double _t21;
//CHECK-NEXT:    double _d__t2 = 0;
//CHECK-NEXT:    double _t22;
//CHECK-NEXT:    double _t23;
//CHECK-NEXT:    double _d__t3 = 0;
//CHECK-NEXT:    double _t24;
//CHECK-NEXT:    double _t25;
//CHECK-NEXT:    double _t26;
//CHECK-NEXT:    double _t27;
//CHECK-NEXT:    double _t28;
//CHECK-NEXT:    double _t29;
//CHECK-NEXT:    double _t30;
//CHECK-NEXT:    double _t31;
//CHECK-NEXT:    double _t32;
//CHECK-NEXT:    double _t33;
//CHECK-NEXT:    double _t34;
//CHECK-NEXT:    double _t35;
//CHECK-NEXT:    double _t36;
//CHECK-NEXT:    double _t37;
//CHECK-NEXT:    double _t38;
//CHECK-NEXT:    double _t39;
//CHECK-NEXT:    double _d_x = 0;
//CHECK-NEXT:    double _d_c = 1;
//CHECK-NEXT:    _cond0 = c > 0;
//CHECK-NEXT:    if (_cond0) {
//CHECK-NEXT:        _t1 = x;
//CHECK-NEXT:        _t0 = x;
//CHECK-NEXT:        double _t00 = _t1 * _t0;
//CHECK-NEXT:        _t3 = c;
//CHECK-NEXT:        _t2 = c;
//CHECK-NEXT:        double _t10 = _t3 * _t2;
//CHECK-NEXT:        _t6 = _d_x;
//CHECK-NEXT:        _t5 = x;
//CHECK-NEXT:        _t8 = x;
//CHECK-NEXT:        _t7 = _d_x;
//CHECK-NEXT:        _t9 = (_t6 * _t5 + _t8 * _t7);
//CHECK-NEXT:        _t4 = x;
//CHECK-NEXT:        _t11 = _t00;
//CHECK-NEXT:        _t10 = _d_x;
//CHECK-NEXT:        _t14 = _d_c;
//CHECK-NEXT:        _t13 = c;
//CHECK-NEXT:        _t16 = c;
//CHECK-NEXT:        _t15 = _d_c;
//CHECK-NEXT:        _t17 = (_t14 * _t13 + _t16 * _t15);
//CHECK-NEXT:        _t12 = c;
//CHECK-NEXT:        _t19 = _t10;
//CHECK-NEXT:        _t18 = _d_c;
//CHECK-NEXT:        double f_cond3_darg1_return = (_t9 * _t4 + _t11 * _t10) + (_t17 * _t12 + _t19 * _t18);
//CHECK-NEXT:        goto _label0;
//CHECK-NEXT:    } else {
//CHECK-NEXT:        _t21 = x;
//CHECK-NEXT:        _t20 = x;
//CHECK-NEXT:        double _t22 = _t21 * _t20;
//CHECK-NEXT:        _t23 = c;
//CHECK-NEXT:        _t22 = c;
//CHECK-NEXT:        double _t30 = _t23 * _t22;
//CHECK-NEXT:        _t26 = _d_x;
//CHECK-NEXT:        _t25 = x;
//CHECK-NEXT:        _t28 = x;
//CHECK-NEXT:        _t27 = _d_x;
//CHECK-NEXT:        _t29 = (_t26 * _t25 + _t28 * _t27);
//CHECK-NEXT:        _t24 = x;
//CHECK-NEXT:        _t31 = _t22;
//CHECK-NEXT:        _t30 = _d_x;
//CHECK-NEXT:        _t34 = _d_c;
//CHECK-NEXT:        _t33 = c;
//CHECK-NEXT:        _t36 = c;
//CHECK-NEXT:        _t35 = _d_c;
//CHECK-NEXT:        _t37 = (_t34 * _t33 + _t36 * _t35);
//CHECK-NEXT:        _t32 = c;
//CHECK-NEXT:        _t39 = _t30;
//CHECK-NEXT:        _t38 = _d_c;
//CHECK-NEXT:        double f_cond3_darg1_return = (_t29 * _t24 + _t31 * _t30) - (_t37 * _t32 + _t39 * _t38);
//CHECK-NEXT:        goto _label1;
//CHECK-NEXT:    }
//CHECK-NEXT:    if (_cond0) {
//CHECK-NEXT:      _label0:
//CHECK-NEXT:        {
//CHECK-NEXT:            double _r4 = 1 * _t4;
//CHECK-NEXT:            double _r5 = _r4 * _t5;
//CHECK-NEXT:            _d__d_x += _r5;
//CHECK-NEXT:            double _r6 = _t6 * _r4;
//CHECK-NEXT:            _result[0UL] += _r6;
//CHECK-NEXT:            double _r7 = _r4 * _t7;
//CHECK-NEXT:            _result[0UL] += _r7;
//CHECK-NEXT:            double _r8 = _t8 * _r4;
//CHECK-NEXT:            _d__d_x += _r8;
//CHECK-NEXT:            double _r9 = _t9 * 1;
//CHECK-NEXT:            _result[0UL] += _r9;
//CHECK-NEXT:            double _r10 = 1 * _t10;
//CHECK-NEXT:            _d__t0 += _r10;
//CHECK-NEXT:            double _r11 = _t11 * 1;
//CHECK-NEXT:            _d__d_x += _r11;
//CHECK-NEXT:            double _r12 = 1 * _t12;
//CHECK-NEXT:            double _r13 = _r12 * _t13;
//CHECK-NEXT:            _d__d_c += _r13;
//CHECK-NEXT:            double _r14 = _t14 * _r12;
//CHECK-NEXT:            _result[1UL] += _r14;
//CHECK-NEXT:            double _r15 = _r12 * _t15;
//CHECK-NEXT:            _result[1UL] += _r15;
//CHECK-NEXT:            double _r16 = _t16 * _r12;
//CHECK-NEXT:            _d__d_c += _r16;
//CHECK-NEXT:            double _r17 = _t17 * 1;
//CHECK-NEXT:            _result[1UL] += _r17;
//CHECK-NEXT:            double _r18 = 1 * _t18;
//CHECK-NEXT:            _d__t1 += _r18;
//CHECK-NEXT:            double _r19 = _t19 * 1;
//CHECK-NEXT:            _d__d_c += _r19;
//CHECK-NEXT:        }
//CHECK-NEXT:        {
//CHECK-NEXT:            double _r2 = _d__t1 * _t2;
//CHECK-NEXT:            _result[1UL] += _r2;
//CHECK-NEXT:            double _r3 = _t3 * _d__t1;
//CHECK-NEXT:            _result[1UL] += _r3;
//CHECK-NEXT:        }
//CHECK-NEXT:        {
//CHECK-NEXT:            double _r0 = _d__t0 * _t0;
//CHECK-NEXT:            _result[0UL] += _r0;
//CHECK-NEXT:            double _r1 = _t1 * _d__t0;
//CHECK-NEXT:            _result[0UL] += _r1;
//CHECK-NEXT:        }
//CHECK-NEXT:    } else {
//CHECK-NEXT:      _label1:
//CHECK-NEXT:        {
//CHECK-NEXT:            double _r24 = 1 * _t24;
//CHECK-NEXT:            double _r25 = _r24 * _t25;
//CHECK-NEXT:            _d__d_x += _r25;
//CHECK-NEXT:            double _r26 = _t26 * _r24;
//CHECK-NEXT:            _result[0UL] += _r26;
//CHECK-NEXT:            double _r27 = _r24 * _t27;
//CHECK-NEXT:            _result[0UL] += _r27;
//CHECK-NEXT:            double _r28 = _t28 * _r24;
//CHECK-NEXT:            _d__d_x += _r28;
//CHECK-NEXT:            double _r29 = _t29 * 1;
//CHECK-NEXT:            _result[0UL] += _r29;
//CHECK-NEXT:            double _r30 = 1 * _t30;
//CHECK-NEXT:            _d__t2 += _r30;
//CHECK-NEXT:            double _r31 = _t31 * 1;
//CHECK-NEXT:            _d__d_x += _r31;
//CHECK-NEXT:            double _r32 = -1 * _t32;
//CHECK-NEXT:            double _r33 = _r32 * _t33;
//CHECK-NEXT:            _d__d_c += _r33;
//CHECK-NEXT:            double _r34 = _t34 * _r32;
//CHECK-NEXT:            _result[1UL] += _r34;
//CHECK-NEXT:            double _r35 = _r32 * _t35;
//CHECK-NEXT:            _result[1UL] += _r35;
//CHECK-NEXT:            double _r36 = _t36 * _r32;
//CHECK-NEXT:            _d__d_c += _r36;
//CHECK-NEXT:            double _r37 = _t37 * -1;
//CHECK-NEXT:            _result[1UL] += _r37;
//CHECK-NEXT:            double _r38 = -1 * _t38;
//CHECK-NEXT:            _d__t3 += _r38;
//CHECK-NEXT:            double _r39 = _t39 * -1;
//CHECK-NEXT:            _d__d_c += _r39;
//CHECK-NEXT:        }
//CHECK-NEXT:        {
//CHECK-NEXT:            double _r22 = _d__t3 * _t22;
//CHECK-NEXT:            _result[1UL] += _r22;
//CHECK-NEXT:            double _r23 = _t23 * _d__t3;
//CHECK-NEXT:            _result[1UL] += _r23;
//CHECK-NEXT:        }
//CHECK-NEXT:        {
//CHECK-NEXT:            double _r20 = _d__t2 * _t20;
//CHECK-NEXT:            _result[0UL] += _r20;
//CHECK-NEXT:            double _r21 = _t21 * _d__t2;
//CHECK-NEXT:            _result[0UL] += _r21;
//CHECK-NEXT:        }
//CHECK-NEXT:    }
//CHECK-NEXT:}

void f_cond3_hessian(double x, double c, double *hessianMatrix);
//CHECK:void f_cond3_hessian(double x, double c, double *hessianMatrix) {
//CHECK-NEXT:    f_cond3_darg0_grad(x, c, &hessianMatrix[0UL]);
//CHECK-NEXT:    f_cond3_darg1_grad(x, c, &hessianMatrix[2UL]);
//CHECK-NEXT:}

double f_power10(double x) {
  return x * x * x * x * x * x * x * x * x * x;
}

void f_power10_darg0_grad(double x, double *_result);
//CHECK:void f_power10_darg0_grad(double x, double *_result) {
//CHECK-NEXT:    double _d__d_x = 0;
//CHECK-NEXT:    double _t0;
//CHECK-NEXT:    double _t1;
//CHECK-NEXT:    double _d__t0 = 0;
//CHECK-NEXT:    double _t2;
//CHECK-NEXT:    double _t3;
//CHECK-NEXT:    double _d__t1 = 0;
//CHECK-NEXT:    double _t4;
//CHECK-NEXT:    double _t5;
//CHECK-NEXT:    double _d__t2 = 0;
//CHECK-NEXT:    double _t6;
//CHECK-NEXT:    double _t7;
//CHECK-NEXT:    double _d__t3 = 0;
//CHECK-NEXT:    double _t8;
//CHECK-NEXT:    double _t9;
//CHECK-NEXT:    double _d__t4 = 0;
//CHECK-NEXT:    double _t10;
//CHECK-NEXT:    double _t11;
//CHECK-NEXT:    double _d__t5 = 0;
//CHECK-NEXT:    double _t12;
//CHECK-NEXT:    double _t13;
//CHECK-NEXT:    double _d__t6 = 0;
//CHECK-NEXT:    double _t14;
//CHECK-NEXT:    double _t15;
//CHECK-NEXT:    double _d__t7 = 0;
//CHECK-NEXT:    double _t16;
//CHECK-NEXT:    double _t17;
//CHECK-NEXT:    double _t18;
//CHECK-NEXT:    double _t19;
//CHECK-NEXT:    double _t20;
//CHECK-NEXT:    double _t21;
//CHECK-NEXT:    double _t22;
//CHECK-NEXT:    double _t23;
//CHECK-NEXT:    double _t24;
//CHECK-NEXT:    double _t25;
//CHECK-NEXT:    double _t26;
//CHECK-NEXT:    double _t27;
//CHECK-NEXT:    double _t28;
//CHECK-NEXT:    double _t29;
//CHECK-NEXT:    double _t30;
//CHECK-NEXT:    double _t31;
//CHECK-NEXT:    double _t32;
//CHECK-NEXT:    double _t33;
//CHECK-NEXT:    double _t34;
//CHECK-NEXT:    double _t35;
//CHECK-NEXT:    double _t36;
//CHECK-NEXT:    double _t37;
//CHECK-NEXT:    double _t38;
//CHECK-NEXT:    double _t39;
//CHECK-NEXT:    double _t40;
//CHECK-NEXT:    double _t41;
//CHECK-NEXT:    double _t42;
//CHECK-NEXT:    double _t43;
//CHECK-NEXT:    double _t44;
//CHECK-NEXT:    double _t45;
//CHECK-NEXT:    double _t46;
//CHECK-NEXT:    double _t47;
//CHECK-NEXT:    double _t48;
//CHECK-NEXT:    double _t49;
//CHECK-NEXT:    double _t50;
//CHECK-NEXT:    double _t51;
//CHECK-NEXT:    double _d_x = 1;
//CHECK-NEXT:    _t1 = x;
//CHECK-NEXT:    _t0 = x;
//CHECK-NEXT:    double _t00 = _t1 * _t0;
//CHECK-NEXT:    _t3 = _t00;
//CHECK-NEXT:    _t2 = x;
//CHECK-NEXT:    double _t10 = _t3 * _t2;
//CHECK-NEXT:    _t5 = _t10;
//CHECK-NEXT:    _t4 = x;
//CHECK-NEXT:    double _t20 = _t5 * _t4;
//CHECK-NEXT:    _t7 = _t20;
//CHECK-NEXT:    _t6 = x;
//CHECK-NEXT:    double _t30 = _t7 * _t6;
//CHECK-NEXT:    _t9 = _t30;
//CHECK-NEXT:    _t8 = x;
//CHECK-NEXT:    double _t40 = _t9 * _t8;
//CHECK-NEXT:    _t11 = _t40;
//CHECK-NEXT:    _t10 = x;
//CHECK-NEXT:    double _t50 = _t11 * _t10;
//CHECK-NEXT:    _t13 = _t50;
//CHECK-NEXT:    _t12 = x;
//CHECK-NEXT:    double _t60 = _t13 * _t12;
//CHECK-NEXT:    _t15 = _t60;
//CHECK-NEXT:    _t14 = x;
//CHECK-NEXT:    double _t70 = _t15 * _t14;
//CHECK-NEXT:    _t25 = _d_x;
//CHECK-NEXT:    _t24 = x;
//CHECK-NEXT:    _t27 = x;
//CHECK-NEXT:    _t26 = _d_x;
//CHECK-NEXT:    _t28 = (_t25 * _t24 + _t27 * _t26);
//CHECK-NEXT:    _t23 = x;
//CHECK-NEXT:    _t30 = _t00;
//CHECK-NEXT:    _t29 = _d_x;
//CHECK-NEXT:    _t31 = (_t28 * _t23 + _t30 * _t29);
//CHECK-NEXT:    _t22 = x;
//CHECK-NEXT:    _t33 = _t10;
//CHECK-NEXT:    _t32 = _d_x;
//CHECK-NEXT:    _t34 = (_t31 * _t22 + _t33 * _t32);
//CHECK-NEXT:    _t21 = x;
//CHECK-NEXT:    _t36 = _t20;
//CHECK-NEXT:    _t35 = _d_x;
//CHECK-NEXT:    _t37 = (_t34 * _t21 + _t36 * _t35);
//CHECK-NEXT:    _t20 = x;
//CHECK-NEXT:    _t39 = _t30;
//CHECK-NEXT:    _t38 = _d_x;
//CHECK-NEXT:    _t40 = (_t37 * _t20 + _t39 * _t38);
//CHECK-NEXT:    _t19 = x;
//CHECK-NEXT:    _t42 = _t40;
//CHECK-NEXT:    _t41 = _d_x;
//CHECK-NEXT:    _t43 = (_t40 * _t19 + _t42 * _t41);
//CHECK-NEXT:    _t18 = x;
//CHECK-NEXT:    _t45 = _t50;
//CHECK-NEXT:    _t44 = _d_x;
//CHECK-NEXT:    _t46 = (_t43 * _t18 + _t45 * _t44);
//CHECK-NEXT:    _t17 = x;
//CHECK-NEXT:    _t48 = _t60;
//CHECK-NEXT:    _t47 = _d_x;
//CHECK-NEXT:    _t49 = (_t46 * _t17 + _t48 * _t47);
//CHECK-NEXT:    _t16 = x;
//CHECK-NEXT:    _t51 = _t70;
//CHECK-NEXT:    _t50 = _d_x;
//CHECK-NEXT:    double f_power10_darg0_return = _t49 * _t16 + _t51 * _t50;
//CHECK-NEXT:    goto _label0;
//CHECK-NEXT:  _label0:
//CHECK-NEXT:    {
//CHECK-NEXT:        double _r16 = 1 * _t16;
//CHECK-NEXT:        double _r17 = _r16 * _t17;
//CHECK-NEXT:        double _r18 = _r17 * _t18;
//CHECK-NEXT:        double _r19 = _r18 * _t19;
//CHECK-NEXT:        double _r20 = _r19 * _t20;
//CHECK-NEXT:        double _r21 = _r20 * _t21;
//CHECK-NEXT:        double _r22 = _r21 * _t22;
//CHECK-NEXT:        double _r23 = _r22 * _t23;
//CHECK-NEXT:        double _r24 = _r23 * _t24;
//CHECK-NEXT:        _d__d_x += _r24;
//CHECK-NEXT:        double _r25 = _t25 * _r23;
//CHECK-NEXT:        _result[0UL] += _r25;
//CHECK-NEXT:        double _r26 = _r23 * _t26;
//CHECK-NEXT:        _result[0UL] += _r26;
//CHECK-NEXT:        double _r27 = _t27 * _r23;
//CHECK-NEXT:        _d__d_x += _r27;
//CHECK-NEXT:        double _r28 = _t28 * _r22;
//CHECK-NEXT:        _result[0UL] += _r28;
//CHECK-NEXT:        double _r29 = _r22 * _t29;
//CHECK-NEXT:        _d__t0 += _r29;
//CHECK-NEXT:        double _r30 = _t30 * _r22;
//CHECK-NEXT:        _d__d_x += _r30;
//CHECK-NEXT:        double _r31 = _t31 * _r21;
//CHECK-NEXT:        _result[0UL] += _r31;
//CHECK-NEXT:        double _r32 = _r21 * _t32;
//CHECK-NEXT:        _d__t1 += _r32;
//CHECK-NEXT:        double _r33 = _t33 * _r21;
//CHECK-NEXT:        _d__d_x += _r33;
//CHECK-NEXT:        double _r34 = _t34 * _r20;
//CHECK-NEXT:        _result[0UL] += _r34;
//CHECK-NEXT:        double _r35 = _r20 * _t35;
//CHECK-NEXT:        _d__t2 += _r35;
//CHECK-NEXT:        double _r36 = _t36 * _r20;
//CHECK-NEXT:        _d__d_x += _r36;
//CHECK-NEXT:        double _r37 = _t37 * _r19;
//CHECK-NEXT:        _result[0UL] += _r37;
//CHECK-NEXT:        double _r38 = _r19 * _t38;
//CHECK-NEXT:        _d__t3 += _r38;
//CHECK-NEXT:        double _r39 = _t39 * _r19;
//CHECK-NEXT:        _d__d_x += _r39;
//CHECK-NEXT:        double _r40 = _t40 * _r18;
//CHECK-NEXT:        _result[0UL] += _r40;
//CHECK-NEXT:        double _r41 = _r18 * _t41;
//CHECK-NEXT:        _d__t4 += _r41;
//CHECK-NEXT:        double _r42 = _t42 * _r18;
//CHECK-NEXT:        _d__d_x += _r42;
//CHECK-NEXT:        double _r43 = _t43 * _r17;
//CHECK-NEXT:        _result[0UL] += _r43;
//CHECK-NEXT:        double _r44 = _r17 * _t44;
//CHECK-NEXT:        _d__t5 += _r44;
//CHECK-NEXT:        double _r45 = _t45 * _r17;
//CHECK-NEXT:        _d__d_x += _r45;
//CHECK-NEXT:        double _r46 = _t46 * _r16;
//CHECK-NEXT:        _result[0UL] += _r46;
//CHECK-NEXT:        double _r47 = _r16 * _t47;
//CHECK-NEXT:        _d__t6 += _r47;
//CHECK-NEXT:        double _r48 = _t48 * _r16;
//CHECK-NEXT:        _d__d_x += _r48;
//CHECK-NEXT:        double _r49 = _t49 * 1;
//CHECK-NEXT:        _result[0UL] += _r49;
//CHECK-NEXT:        double _r50 = 1 * _t50;
//CHECK-NEXT:        _d__t7 += _r50;
//CHECK-NEXT:        double _r51 = _t51 * 1;
//CHECK-NEXT:        _d__d_x += _r51;
//CHECK-NEXT:    }
//CHECK-NEXT:    {
//CHECK-NEXT:        double _r14 = _d__t7 * _t14;
//CHECK-NEXT:        _d__t6 += _r14;
//CHECK-NEXT:        double _r15 = _t15 * _d__t7;
//CHECK-NEXT:        _result[0UL] += _r15;
//CHECK-NEXT:    }
//CHECK-NEXT:    {
//CHECK-NEXT:        double _r12 = _d__t6 * _t12;
//CHECK-NEXT:        _d__t5 += _r12;
//CHECK-NEXT:        double _r13 = _t13 * _d__t6;
//CHECK-NEXT:        _result[0UL] += _r13;
//CHECK-NEXT:    }
//CHECK-NEXT:    {
//CHECK-NEXT:        double _r10 = _d__t5 * _t10;
//CHECK-NEXT:        _d__t4 += _r10;
//CHECK-NEXT:        double _r11 = _t11 * _d__t5;
//CHECK-NEXT:        _result[0UL] += _r11;
//CHECK-NEXT:    }
//CHECK-NEXT:    {
//CHECK-NEXT:        double _r8 = _d__t4 * _t8;
//CHECK-NEXT:        _d__t3 += _r8;
//CHECK-NEXT:        double _r9 = _t9 * _d__t4;
//CHECK-NEXT:        _result[0UL] += _r9;
//CHECK-NEXT:    }
//CHECK-NEXT:    {
//CHECK-NEXT:        double _r6 = _d__t3 * _t6;
//CHECK-NEXT:        _d__t2 += _r6;
//CHECK-NEXT:        double _r7 = _t7 * _d__t3;
//CHECK-NEXT:        _result[0UL] += _r7;
//CHECK-NEXT:    }
//CHECK-NEXT:    {
//CHECK-NEXT:        double _r4 = _d__t2 * _t4;
//CHECK-NEXT:        _d__t1 += _r4;
//CHECK-NEXT:        double _r5 = _t5 * _d__t2;
//CHECK-NEXT:        _result[0UL] += _r5;
//CHECK-NEXT:    }
//CHECK-NEXT:    {
//CHECK-NEXT:        double _r2 = _d__t1 * _t2;
//CHECK-NEXT:        _d__t0 += _r2;
//CHECK-NEXT:        double _r3 = _t3 * _d__t1;
//CHECK-NEXT:        _result[0UL] += _r3;
//CHECK-NEXT:    }
//CHECK-NEXT:    {
//CHECK-NEXT:        double _r0 = _d__t0 * _t0;
//CHECK-NEXT:        _result[0UL] += _r0;
//CHECK-NEXT:        double _r1 = _t1 * _d__t0;
//CHECK-NEXT:        _result[0UL] += _r1;
//CHECK-NEXT:    }
//CHECK-NEXT:}

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
