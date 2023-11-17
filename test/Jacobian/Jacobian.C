// RUN: %cladclang %s -I%S/../../include -oJacobian.out 2>&1 | FileCheck %s
// RUN: ./Jacobian.out | FileCheck -check-prefix=CHECK-EXEC %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -enable-tbr %s -I%S/../../include -oJacobian.out
// RUN: ./Jacobian.out | FileCheck -check-prefix=CHECK-EXEC %s

//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"
#include <cmath>

void f_1(double a, double b, double c, double output[]) {
  output[0] = a * a * a;
  output[1] = a * a * a + b * b * b;
  output[2] = c * c * 10 - a * a;
}

void f_1_jac(double a, double b, double c, double output[], double *_result);
//CHECK:void f_1_jac(double a, double b, double c, double output[], double *jacobianMatrix) {
//CHECK-NEXT:  double _t0;
//CHECK-NEXT:  double _t1;
//CHECK-NEXT:  double _t2;
//CHECK-NEXT:  double _t3;
//CHECK-NEXT:  double _t4;
//CHECK-NEXT:  double _t5;
//CHECK-NEXT:  double _t6;
//CHECK-NEXT:  double _t7;
//CHECK-NEXT:  double _t8;
//CHECK-NEXT:  _t1 = a;
//CHECK-NEXT:  _t0 = a;
//CHECK-NEXT:  output[0] = a * a * a;
//CHECK-NEXT:  _t3 = a;
//CHECK-NEXT:  _t2 = a;
//CHECK-NEXT:  _t5 = b;
//CHECK-NEXT:  _t4 = b;
//CHECK-NEXT:  output[1] = a * a * a + b * b * b;
//CHECK-NEXT:  _t7 = c;
//CHECK-NEXT:  _t6 = 10;
//CHECK-NEXT:  _t8 = a;
//CHECK-NEXT:  output[2] = c * c * 10 - a * a;
//CHECK-NEXT:  {
//CHECK-NEXT:    double _r12 = 1 * _t6;
//CHECK-NEXT:    double _r13 = _r12 * _t7;
//CHECK-NEXT:    jacobianMatrix[8UL] += _r13;
//CHECK-NEXT:    double _r14 = c * _r12;
//CHECK-NEXT:    jacobianMatrix[8UL] += _r14;
//CHECK-NEXT:    double _r15 = c * _t7 * 1;
//CHECK-NEXT:    double _r16 = -1 * _t8;
//CHECK-NEXT:    jacobianMatrix[6UL] += _r16;
//CHECK-NEXT:    double _r17 = a * -1;
//CHECK-NEXT:    jacobianMatrix[6UL] += _r17;
//CHECK-NEXT:  }
//CHECK-NEXT:  {
//CHECK-NEXT:    double _r4 = 1 * _t2;
//CHECK-NEXT:    double _r5 = _r4 * _t3;
//CHECK-NEXT:    jacobianMatrix[3UL] += _r5;
//CHECK-NEXT:    double _r6 = a * _r4;
//CHECK-NEXT:    jacobianMatrix[3UL] += _r6;
//CHECK-NEXT:    double _r7 = a * _t3 * 1;
//CHECK-NEXT:    jacobianMatrix[3UL] += _r7;
//CHECK-NEXT:    double _r8 = 1 * _t4;
//CHECK-NEXT:    double _r9 = _r8 * _t5;
//CHECK-NEXT:    jacobianMatrix[4UL] += _r9;
//CHECK-NEXT:    double _r10 = b * _r8;
//CHECK-NEXT:    jacobianMatrix[4UL] += _r10;
//CHECK-NEXT:    double _r11 = b * _t5 * 1;
//CHECK-NEXT:    jacobianMatrix[4UL] += _r11;
//CHECK-NEXT:  }
//CHECK-NEXT:  {
//CHECK-NEXT:    double _r0 = 1 * _t0;
//CHECK-NEXT:    double _r1 = _r0 * _t1;
//CHECK-NEXT:    jacobianMatrix[0UL] += _r1;
//CHECK-NEXT:    double _r2 = a * _r0;
//CHECK-NEXT:    jacobianMatrix[0UL] += _r2;
//CHECK-NEXT:    double _r3 = a * _t1 * 1;
//CHECK-NEXT:    jacobianMatrix[0UL] += _r3;
//CHECK-NEXT:  }
//CHECK-NEXT:}


void f_3(double x, double y, double z, double *_result) {
  double constant = 42;

  _result[0] = sin(x) * constant;
  _result[1] = sin(y) * constant;
  _result[2] = sin(z) * constant;
}

void f_3_jac(double x, double y, double z, double *_result, double *jacobianMatrix);
//CHECK: void f_3_jac(double x, double y, double z, double *_result, double *jacobianMatrix) {
//CHECK-NEXT:  double _d_constant = 0;
//CHECK-NEXT:  double _t0;
//CHECK-NEXT:  double _t1;
//CHECK-NEXT:  double _t2;
//CHECK-NEXT:  double constant = 42;
//CHECK-NEXT:  _t0 = constant;
//CHECK-NEXT:  _result[0] = sin(x) * constant;
//CHECK-NEXT:  _t1 = constant;
//CHECK-NEXT:  _result[1] = sin(y) * constant;
//CHECK-NEXT:  _t2 = constant;
//CHECK-NEXT:  _result[2] = sin(z) * constant;
//CHECK-NEXT:  {
//CHECK-NEXT:    double _r6 = 1 * _t2;
//CHECK-NEXT:    double _r7 = _r6 * clad::custom_derivatives::sin_pushforward(z, 1.).pushforward;
//CHECK-NEXT:    jacobianMatrix[8UL] += _r7;
//CHECK-NEXT:    double _r8 = sin(z) * 1;
//CHECK-NEXT:  }
//CHECK-NEXT:  {
//CHECK-NEXT:    double _r3 = 1 * _t1;
//CHECK-NEXT:    double _r4 = _r3 * clad::custom_derivatives::sin_pushforward(y, 1.).pushforward;
//CHECK-NEXT:    jacobianMatrix[4UL] += _r4;
//CHECK-NEXT:    double _r5 = sin(y) * 1;
//CHECK-NEXT:  }
//CHECK-NEXT:  {
//CHECK-NEXT:    double _r0 = 1 * _t0;
//CHECK-NEXT:    double _r1 = _r0 * clad::custom_derivatives::sin_pushforward(x, 1.).pushforward;
//CHECK-NEXT:    jacobianMatrix[0UL] += _r1;
//CHECK-NEXT:    double _r2 = sin(x) * 1;
//CHECK-NEXT:  }
//CHECK-NEXT:}

double multiply(double x, double y) { return x * y; }
//CHECK: void multiply_pullback(double x, double y, double _d_y0, clad::array_ref<double> _d_x, clad::array_ref<double> _d_y) {
//CHECK-NEXT:    double _t0;
//CHECK-NEXT:    _t0 = y;
//CHECK-NEXT:    goto _label0;
//CHECK-NEXT:  _label0:
//CHECK-NEXT:    {
//CHECK-NEXT:        double _r0 = _d_y0 * _t0;
//CHECK-NEXT:        * _d_x += _r0;
//CHECK-NEXT:        double _r1 = x * _d_y0;
//CHECK-NEXT:        * _d_y += _r1;
//CHECK-NEXT:    }
//CHECK-NEXT:}

void f_4(double x, double y, double z, double *_result) {
  double constant = 42;

  _result[0] = multiply(x, y) * constant;
  _result[1] = multiply(y, z) * constant;
  _result[2] = multiply(z, x) * constant;
}

void f_4_jac(double x, double y, double z, double *_result, double *jacobianMatrix);
//CHECK: void f_4_jac(double x, double y, double z, double *_result, double *jacobianMatrix) {
//CHECK-NEXT:    double _d_constant = 0;
//CHECK-NEXT:    double _t0;
//CHECK-NEXT:    double _t1;
//CHECK-NEXT:    double _t2;
//CHECK-NEXT:    double constant = 42;
//CHECK-NEXT:    _t0 = constant;
//CHECK-NEXT:    _result[0] = multiply(x, y) * constant;
//CHECK-NEXT:    _t1 = constant;
//CHECK-NEXT:    _result[1] = multiply(y, z) * constant;
//CHECK-NEXT:    _t2 = constant;
//CHECK-NEXT:    _result[2] = multiply(z, x) * constant;
//CHECK-NEXT:    {
//CHECK-NEXT:        double _r8 = 1 * _t2;
//CHECK-NEXT:        double _jac4 = 0.;
//CHECK-NEXT:        double _jac5 = 0.;
//CHECK-NEXT:        multiply_pullback(z, x, _r8, &_jac4, &_jac5);
//CHECK-NEXT:        double _r9 = _jac4;
//CHECK-NEXT:        jacobianMatrix[8UL] += _r9;
//CHECK-NEXT:        double _r10 = _jac5;
//CHECK-NEXT:        jacobianMatrix[6UL] += _r10;
//CHECK-NEXT:        double _r11 = multiply(z, x) * 1;
//CHECK-NEXT:    }
//CHECK-NEXT:    {
//CHECK-NEXT:        double _r4 = 1 * _t1;
//CHECK-NEXT:        double _jac2 = 0.;
//CHECK-NEXT:        double _jac3 = 0.;
//CHECK-NEXT:        multiply_pullback(y, z, _r4, &_jac2, &_jac3);
//CHECK-NEXT:        double _r5 = _jac2;
//CHECK-NEXT:        jacobianMatrix[4UL] += _r5;
//CHECK-NEXT:        double _r6 = _jac3;
//CHECK-NEXT:        jacobianMatrix[5UL] += _r6;
//CHECK-NEXT:        double _r7 = multiply(y, z) * 1;
//CHECK-NEXT:    }
//CHECK-NEXT:    {
//CHECK-NEXT:        double _r0 = 1 * _t0;
//CHECK-NEXT:        double _jac0 = 0.;
//CHECK-NEXT:        double _jac1 = 0.;
//CHECK-NEXT:        multiply_pullback(x, y, _r0, &_jac0, &_jac1);
//CHECK-NEXT:        double _r1 = _jac0;
//CHECK-NEXT:        jacobianMatrix[0UL] += _r1;
//CHECK-NEXT:        double _r2 = _jac1;
//CHECK-NEXT:        jacobianMatrix[1UL] += _r2;
//CHECK-NEXT:        double _r3 = multiply(x, y) * 1;
//CHECK-NEXT:    }
//CHECK-NEXT:}

void f_1_jac_0(double a, double b, double c, double output[], double *jacobianMatrix);
// CHECK: void f_1_jac_0(double a, double b, double c, double output[], double *jacobianMatrix) {
// CHECK-NEXT:  double _d_b = 0;
// CHECK-NEXT:  double _d_c = 0;
// CHECK-NEXT:  double _t0;
// CHECK-NEXT:  double _t1;
// CHECK-NEXT:  double _t2;
// CHECK-NEXT:  double _t3;
// CHECK-NEXT:  double _t4;
// CHECK-NEXT:  double _t5;
// CHECK-NEXT:  double _t6;
// CHECK-NEXT:  double _t7;
// CHECK-NEXT:  double _t8;
// CHECK-NEXT:  _t1 = a;
// CHECK-NEXT:  _t0 = a;
// CHECK-NEXT:  output[0] = a * a * a;
// CHECK-NEXT:  _t3 = a;
// CHECK-NEXT:  _t2 = a;
// CHECK-NEXT:  _t5 = b;
// CHECK-NEXT:  _t4 = b;
// CHECK-NEXT:  output[1] = a * a * a + b * b * b;
// CHECK-NEXT:  _t7 = c;
// CHECK-NEXT:  _t6 = 10;
// CHECK-NEXT:  _t8 = a;
// CHECK-NEXT:  output[2] = c * c * 10 - a * a;
// CHECK-NEXT:  {
// CHECK-NEXT:    double _r12 = 1 * _t6;
// CHECK-NEXT:    double _r13 = _r12 * _t7;
// CHECK-NEXT:    double _r14 = c * _r12;
// CHECK-NEXT:    double _r15 = c * _t7 * 1;
// CHECK-NEXT:    double _r16 = -1 * _t8;
// CHECK-NEXT:    jacobianMatrix[2UL] += _r16;
// CHECK-NEXT:    double _r17 = a * -1;
// CHECK-NEXT:    jacobianMatrix[2UL] += _r17;
// CHECK-NEXT:  }
// CHECK-NEXT:  {
// CHECK-NEXT:    double _r4 = 1 * _t2;
// CHECK-NEXT:    double _r5 = _r4 * _t3;
// CHECK-NEXT:    jacobianMatrix[1UL] += _r5;
// CHECK-NEXT:    double _r6 = a * _r4;
// CHECK-NEXT:    jacobianMatrix[1UL] += _r6;
// CHECK-NEXT:    double _r7 = a * _t3 * 1;
// CHECK-NEXT:    jacobianMatrix[1UL] += _r7;
// CHECK-NEXT:    double _r8 = 1 * _t4;
// CHECK-NEXT:    double _r9 = _r8 * _t5;
// CHECK-NEXT:    double _r10 = b * _r8;
// CHECK-NEXT:    double _r11 = b * _t5 * 1;
// CHECK-NEXT:  }
// CHECK-NEXT:  {
// CHECK-NEXT:    double _r0 = 1 * _t0;
// CHECK-NEXT:    double _r1 = _r0 * _t1;
// CHECK-NEXT:    jacobianMatrix[0UL] += _r1;
// CHECK-NEXT:    double _r2 = a * _r0;
// CHECK-NEXT:    jacobianMatrix[0UL] += _r2;
// CHECK-NEXT:    double _r3 = a * _t1 * 1;
// CHECK-NEXT:    jacobianMatrix[0UL] += _r3;
// CHECK-NEXT:  }
// CHECK-NEXT:}

#define TEST(F, x, y, z) { \
  result[0] = 0; result[1] = 0; result[2] = 0;\
  result[3] = 0; result[4] = 0; result[5] = 0;\
  result[6] = 0; result[7] = 0; result[8] = 0;\
  outputarr[0] = 0; outputarr[1] = 1; outputarr[2] = 0;\
  auto j = clad::jacobian(F);\
  j.execute(x, y, z, outputarr, result);\
  printf("Result is = {%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f}\n",\
  result[0], result[1], result[2],\
  result[3], result[4], result[5],\
  result[6], result[7], result[8]);\
  F##_jac(x, y, z, outputarr, result);\
}

#define TEST_F_1_SINGLE_PARAM(x, y, z) { \
  result[0] = 0; result[1] = 0; result[2] = 0;\
  outputarr[0] = 0; outputarr[1] = 1; outputarr[2] = 0;\
  auto j = clad::jacobian(f_1,"a");\
  j.execute(x, y, z, outputarr, result);\
  printf("Result is = {%.2f, %.2f, %.2f}\n",\
  result[0], result[1], result[2]);\
}


int main() {
  double result[10];
  double outputarr[9];
  TEST(f_1, 1, 2, 3); // CHECK-EXEC: Result is = {3.00, 0.00, 0.00, 3.00, 12.00, 0.00, -2.00, 0.00, 60.00}
  TEST(f_3, 1, 2, 3); // CHECK-EXEC: Result is = {22.69, 0.00, 0.00, 0.00, -17.48, 0.00, 0.00, 0.00, -41.58}
  TEST(f_4, 1, 2, 3); // CHECK-EXEC: Result is = {84.00, 42.00, 0.00, 0.00, 126.00, 84.00, 126.00, 0.00, 42.00}
  TEST_F_1_SINGLE_PARAM(1, 2, 3); // CHECK-EXEC: Result is = {3.00, 3.00, -2.00}
}
