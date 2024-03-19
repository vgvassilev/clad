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
//CHECK-NEXT:  output[0] = a * a * a;
//CHECK-NEXT:  output[1] = a * a * a + b * b * b;
//CHECK-NEXT:  output[2] = c * c * 10 - a * a;
//CHECK-NEXT:  {
//CHECK-NEXT:    jacobianMatrix[{{8U|8UL}}] += 1 * 10 * c;
//CHECK-NEXT:    jacobianMatrix[{{8U|8UL}}] += c * 1 * 10;
//CHECK-NEXT:    jacobianMatrix[{{6U|6UL}}] += -1 * a;
//CHECK-NEXT:    jacobianMatrix[{{6U|6UL}}] += a * -1;
//CHECK-NEXT:  }
//CHECK-NEXT:  {
//CHECK-NEXT:    jacobianMatrix[{{3U|3UL}}] += 1 * a * a;
//CHECK-NEXT:    jacobianMatrix[{{3U|3UL}}] += a * 1 * a;
//CHECK-NEXT:    jacobianMatrix[{{3U|3UL}}] += a * a * 1;
//CHECK-NEXT:    jacobianMatrix[{{4U|4UL}}] += 1 * b * b;
//CHECK-NEXT:    jacobianMatrix[{{4U|4UL}}] += b * 1 * b;
//CHECK-NEXT:    jacobianMatrix[{{4U|4UL}}] += b * b * 1;
//CHECK-NEXT:  }
//CHECK-NEXT:  {
//CHECK-NEXT:    jacobianMatrix[{{0U|0UL}}] += 1 * a * a;
//CHECK-NEXT:    jacobianMatrix[{{0U|0UL}}] += a * 1 * a;
//CHECK-NEXT:    jacobianMatrix[{{0U|0UL}}] += a * a * 1;
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
//CHECK-NEXT:  _t0 = sin(x);
//CHECK-NEXT:  _result[0] = sin(x) * constant;
//CHECK-NEXT:  _t1 = sin(y);
//CHECK-NEXT:  _result[1] = sin(y) * constant;
//CHECK-NEXT:  _t2 = sin(z);
//CHECK-NEXT:  _result[2] = sin(z) * constant;
//CHECK-NEXT:  {
//CHECK-NEXT:    double _r2 = 0;
//CHECK-NEXT:    _r2 += 1 * constant * clad::custom_derivatives::sin_pushforward(z, 1.).pushforward;
//CHECK-NEXT:    jacobianMatrix[{{8U|8UL}}] += _r2;
//CHECK-NEXT:  }
//CHECK-NEXT:  {
//CHECK-NEXT:    double _r1 = 0;
//CHECK-NEXT:    _r1 += 1 * constant * clad::custom_derivatives::sin_pushforward(y, 1.).pushforward;
//CHECK-NEXT:    jacobianMatrix[{{4U|4UL}}] += _r1;
//CHECK-NEXT:  }
//CHECK-NEXT:  {
//CHECK-NEXT:    double _r0 = 0;
//CHECK-NEXT:    _r0 += 1 * constant * clad::custom_derivatives::sin_pushforward(x, 1.).pushforward;
//CHECK-NEXT:    jacobianMatrix[{{0U|0UL}}] += _r0;
//CHECK-NEXT:  }
//CHECK-NEXT:}

double multiply(double x, double y) { return x * y; }
//CHECK: void multiply_pullback(double x, double y, double _d_y0, double *_d_x, double *_d_y) {
//CHECK-NEXT:    goto _label0;
//CHECK-NEXT:  _label0:
//CHECK-NEXT:    {
//CHECK-NEXT:        *_d_x += _d_y0 * y;
//CHECK-NEXT:        *_d_y += x * _d_y0;
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
//CHECK-NEXT:    _t0 = multiply(x, y);
//CHECK-NEXT:    _result[0] = multiply(x, y) * constant;
//CHECK-NEXT:    _t1 = multiply(y, z);
//CHECK-NEXT:    _result[1] = multiply(y, z) * constant;
//CHECK-NEXT:    _t2 = multiply(z, x);
//CHECK-NEXT:    _result[2] = multiply(z, x) * constant;
//CHECK-NEXT:    {
//CHECK-NEXT:        double _r4 = 0;
//CHECK-NEXT:        double _r5 = 0;
//CHECK-NEXT:        multiply_pullback(z, x, 1 * constant, &_r4, &_r5);
//CHECK-NEXT:        jacobianMatrix[{{8U|8UL}}] += _r4;
//CHECK-NEXT:        jacobianMatrix[{{6U|6UL}}] += _r5;
//CHECK-NEXT:    }
//CHECK-NEXT:    {
//CHECK-NEXT:        double _r2 = 0;
//CHECK-NEXT:        double _r3 = 0;
//CHECK-NEXT:        multiply_pullback(y, z, 1 * constant, &_r2, &_r3);
//CHECK-NEXT:        jacobianMatrix[{{4U|4UL}}] += _r2;
//CHECK-NEXT:        jacobianMatrix[{{5U|5UL}}] += _r3;
//CHECK-NEXT:    }
//CHECK-NEXT:    {
//CHECK-NEXT:        double _r0 = 0;
//CHECK-NEXT:        double _r1 = 0;
//CHECK-NEXT:        multiply_pullback(x, y, 1 * constant, &_r0, &_r1);
//CHECK-NEXT:        jacobianMatrix[{{0U|0UL}}] += _r0;
//CHECK-NEXT:        jacobianMatrix[{{1U|1UL}}] += _r1;
//CHECK-NEXT:    }
//CHECK-NEXT:}

void f_1_jac_0(double a, double b, double c, double output[], double *jacobianMatrix);
// CHECK: void f_1_jac_0(double a, double b, double c, double output[], double *jacobianMatrix) {
// CHECK-NEXT:  double _d_b = 0;
// CHECK-NEXT:  double _d_c = 0;
// CHECK-NEXT:  output[0] = a * a * a;
// CHECK-NEXT:  output[1] = a * a * a + b * b * b;
// CHECK-NEXT:  output[2] = c * c * 10 - a * a;
// CHECK-NEXT:  {
// CHECK-NEXT:    jacobianMatrix[{{2U|2UL}}] += -1 * a;
// CHECK-NEXT:    jacobianMatrix[{{2U|2UL}}] += a * -1;
// CHECK-NEXT:  }
// CHECK-NEXT:  {
// CHECK-NEXT:    jacobianMatrix[{{1U|1UL}}] += 1 * a * a;
// CHECK-NEXT:    jacobianMatrix[{{1U|1UL}}] += a * 1 * a;
// CHECK-NEXT:    jacobianMatrix[{{1U|1UL}}] += a * a * 1;
// CHECK-NEXT:  }
// CHECK-NEXT:  {
// CHECK-NEXT:    jacobianMatrix[{{0U|0UL}}] += 1 * a * a;
// CHECK-NEXT:    jacobianMatrix[{{0U|0UL}}] += a * 1 * a;

// CHECK-NEXT:    jacobianMatrix[{{0U|0UL}}] += a * a * 1;
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
