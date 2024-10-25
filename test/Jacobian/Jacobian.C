// RUN: %cladclang %s -I%S/../../include -oJacobian.out 2>&1 | %filecheck %s
// RUN: ./Jacobian.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -enable-tbr %s -I%S/../../include -oJacobian.out
// RUN: ./Jacobian.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"
#include <cmath>

void f_1(double a, double b, double c, double output[]) {
  output[0] = a * a * a;
  output[1] = a * a * a + b * b * b;
  output[2] = c * c * 10 - a * a;
}

// CHECK: void f_1_jac(double a, double b, double c, double output[], clad::matrix<double> *_d_vector_output) {
// CHECK-NEXT:     unsigned long indepVarCount = _d_vector_output->rows() + {{3U|3UL|3ULL}};
// CHECK-NEXT:     clad::array<double> _d_vector_a = clad::one_hot_vector(indepVarCount, {{0U|0UL|0ULL}});
// CHECK-NEXT:     clad::array<double> _d_vector_b = clad::one_hot_vector(indepVarCount, {{1U|1UL|1ULL}});
// CHECK-NEXT:     clad::array<double> _d_vector_c = clad::one_hot_vector(indepVarCount, {{2U|2UL|2ULL}});
// CHECK-NEXT:     *_d_vector_output = clad::identity_matrix(_d_vector_output->rows(), indepVarCount, {{3U|3UL|3ULL}});
// CHECK-NEXT:     double _t0 = a * a;
// CHECK-NEXT:     *_d_vector_output[0] = (_d_vector_a * a + a * _d_vector_a) * a + _t0 * _d_vector_a;
// CHECK-NEXT:     output[0] = _t0 * a;
// CHECK-NEXT:     double _t1 = a * a;
// CHECK-NEXT:     double _t2 = b * b;
// CHECK-NEXT:     *_d_vector_output[1] = (_d_vector_a * a + a * _d_vector_a) * a + _t1 * _d_vector_a + (_d_vector_b * b + b * _d_vector_b) * b + _t2 * _d_vector_b;
// CHECK-NEXT:     output[1] = _t1 * a + _t2 * b;
// CHECK-NEXT:     double _t3 = c * c;
// CHECK-NEXT:     *_d_vector_output[2] = (_d_vector_c * c + c * _d_vector_c) * 10 + _t3 * (clad::zero_vector(indepVarCount)) - (_d_vector_a * a + a * _d_vector_a);
// CHECK-NEXT:     output[2] = _t3 * 10 - a * a;
// CHECK-NEXT: }

void f_3(double x, double y, double z, double *_result) {
  double constant = 42;

  _result[0] = sin(x) * constant;
  _result[1] = sin(y) * constant;
  _result[2] = sin(z) * constant;
}

// CHECK: void f_3_jac(double x, double y, double z, double *_result, clad::matrix<double> *_d_vector__result) {
// CHECK-NEXT:     unsigned long indepVarCount = _d_vector__result->rows() + {{3U|3UL|3ULL}};
// CHECK-NEXT:     clad::array<double> _d_vector_x = clad::one_hot_vector(indepVarCount, {{0U|0UL|0ULL}});
// CHECK-NEXT:     clad::array<double> _d_vector_y = clad::one_hot_vector(indepVarCount, {{1U|1UL|1ULL}});
// CHECK-NEXT:     clad::array<double> _d_vector_z = clad::one_hot_vector(indepVarCount, {{2U|2UL|2ULL}});
// CHECK-NEXT:     *_d_vector__result = clad::identity_matrix(_d_vector__result->rows(), indepVarCount, {{3U|3UL|3ULL}});
// CHECK-NEXT:     clad::array<double> _d_vector_constant(clad::zero_vector(indepVarCount)); 
// CHECK-NEXT:     double constant = 42;
// CHECK-NEXT:     {{.*}} _t0 = clad::custom_derivatives::sin_pushforward(x, _d_vector_x);
// CHECK-NEXT:     double &_t1 = _t0.value;
// CHECK-NEXT:     *_d_vector__result[0] = _t0.pushforward * constant + _t1 * _d_vector_constant;
// CHECK-NEXT:     _result[0] = _t1 * constant;
// CHECK-NEXT:     {{.*}} _t2 = clad::custom_derivatives::sin_pushforward(y, _d_vector_y);
// CHECK-NEXT:     double &_t3 = _t2.value;
// CHECK-NEXT:     *_d_vector__result[1] = _t2.pushforward * constant + _t3 * _d_vector_constant;
// CHECK-NEXT:     _result[1] = _t3 * constant;
// CHECK-NEXT:     {{.*}} _t4 = clad::custom_derivatives::sin_pushforward(z, _d_vector_z);
// CHECK-NEXT:     double &_t5 = _t4.value;
// CHECK-NEXT:     *_d_vector__result[2] = _t4.pushforward * constant + _t5 * _d_vector_constant;
// CHECK-NEXT:     _result[2] = _t5 * constant;
// CHECK-NEXT: }
double multiply(double x, double y) { return x * y; }
// CHECK: clad::ValueAndPushforward<double, clad::array<double> > multiply_vector_pushforward(double x, double y, clad::array<double> _d_x, clad::array<double> _d_y);

void f_4(double x, double y, double z, double *_result) {
  double constant = 42;

  _result[0] = multiply(x, y) * constant;
  _result[1] = multiply(y, z) * constant;
  _result[2] = multiply(z, x) * constant;
}

// CHECK: void f_4_jac(double x, double y, double z, double *_result, clad::matrix<double> *_d_vector__result) {
// CHECK-NEXT:     unsigned long indepVarCount = _d_vector__result->rows() + {{3U|3UL|3ULL}};
// CHECK-NEXT:     clad::array<double> _d_vector_x = clad::one_hot_vector(indepVarCount, {{0U|0UL|0ULL}});
// CHECK-NEXT:     clad::array<double> _d_vector_y = clad::one_hot_vector(indepVarCount, {{1U|1UL|1ULL}});
// CHECK-NEXT:     clad::array<double> _d_vector_z = clad::one_hot_vector(indepVarCount, {{2U|2UL|2ULL}});
// CHECK-NEXT:     *_d_vector__result = clad::identity_matrix(_d_vector__result->rows(), indepVarCount, {{3U|3UL|3ULL}});
// CHECK-NEXT:     clad::array<double> _d_vector_constant(clad::zero_vector(indepVarCount)); 
// CHECK-NEXT:     double constant = 42;
// CHECK-NEXT:     clad::ValueAndPushforward<double, clad::array<double> > _t0 = multiply_vector_pushforward(x, y, _d_vector_x, _d_vector_y);
// CHECK-NEXT:     double &_t1 = _t0.value;
// CHECK-NEXT:     *_d_vector__result[0] = _t0.pushforward * constant + _t1 * _d_vector_constant;
// CHECK-NEXT:     _result[0] = _t1 * constant;
// CHECK-NEXT:     clad::ValueAndPushforward<double, clad::array<double> > _t2 = multiply_vector_pushforward(y, z, _d_vector_y, _d_vector_z);
// CHECK-NEXT:     double &_t3 = _t2.value;
// CHECK-NEXT:     *_d_vector__result[1] = _t2.pushforward * constant + _t3 * _d_vector_constant;
// CHECK-NEXT:     _result[1] = _t3 * constant;
// CHECK-NEXT:     clad::ValueAndPushforward<double, clad::array<double> > _t4 = multiply_vector_pushforward(z, x, _d_vector_z, _d_vector_x);
// CHECK-NEXT:     double &_t5 = _t4.value;
// CHECK-NEXT:     *_d_vector__result[2] = _t4.pushforward * constant + _t5 * _d_vector_constant;
// CHECK-NEXT:     _result[2] = _t5 * constant;
// CHECK-NEXT: }

// CHECK: void f_1_jac_0(double a, double b, double c, double output[], clad::matrix<double> *_d_vector_output) {
// CHECK-NEXT:     unsigned long indepVarCount = _d_vector_output->rows() + {{3U|3UL|3ULL}};
// CHECK-NEXT:     clad::array<double> _d_vector_a = clad::one_hot_vector(indepVarCount, {{0U|0UL|0ULL}});
// CHECK-NEXT:     clad::array<double> _d_vector_b = clad::zero_vector(indepVarCount);
// CHECK-NEXT:     clad::array<double> _d_vector_c = clad::zero_vector(indepVarCount);
// CHECK-NEXT:     double _t0 = a * a;
// CHECK-NEXT:     *_d_vector_output[0] = (_d_vector_a * a + a * _d_vector_a) * a + _t0 * _d_vector_a;
// CHECK-NEXT:     output[0] = _t0 * a;
// CHECK-NEXT:     double _t1 = a * a;
// CHECK-NEXT:     double _t2 = b * b;
// CHECK-NEXT:     *_d_vector_output[1] = (_d_vector_a * a + a * _d_vector_a) * a + _t1 * _d_vector_a + (_d_vector_b * b + b * _d_vector_b) * b + _t2 * _d_vector_b;
// CHECK-NEXT:     output[1] = _t1 * a + _t2 * b;
// CHECK-NEXT:     double _t3 = c * c;
// CHECK-NEXT:     *_d_vector_output[2] = (_d_vector_c * c + c * _d_vector_c) * 10 + _t3 * (clad::zero_vector(indepVarCount)) - (_d_vector_a * a + a * _d_vector_a);
// CHECK-NEXT:     output[2] = _t3 * 10 - a * a;
// CHECK-NEXT: }

void f_5(float a, double output[]){
  output[1]=a;
  output[0]=a*a;  
}

// CHECK: void f_5_jac(float a, double output[], clad::matrix<double> *_d_vector_output) {
// CHECK-NEXT:     unsigned long indepVarCount = _d_vector_output->rows() + {{1U|1UL|1ULL}};
// CHECK-NEXT:     clad::array<float> _d_vector_a = clad::one_hot_vector(indepVarCount, {{0U|0UL|0ULL}});
// CHECK-NEXT:     *_d_vector_output = clad::identity_matrix(_d_vector_output->rows(), indepVarCount, {{1U|1UL|1ULL}});
// CHECK-NEXT:     *_d_vector_output[1] = _d_vector_a;
// CHECK-NEXT:     output[1] = a;
// CHECK-NEXT:     *_d_vector_output[0] = _d_vector_a * a + a * _d_vector_a;
// CHECK-NEXT:     output[0] = a * a;
// CHECK-NEXT: }

void f_6(double a[3], double b, double output[]) {
    output[0] = a[0] * a[0] * a[0];
    output[1] = a[0] * a[0] * a[0] + b * b * b;
    output[2] = 2 * (a[0] + b);
}

// CHECK: void f_6_jac(double a[3], double b, double output[], clad::matrix<double> *_d_vector_a, clad::matrix<double> *_d_vector_output) {
// CHECK-NEXT:     unsigned long indepVarCount = _d_vector_a->rows() + _d_vector_output->rows() + {{1U|1UL|1ULL}};
// CHECK-NEXT:     clad::array<double> _d_vector_b = clad::one_hot_vector(indepVarCount, _d_vector_a->rows());
// CHECK-NEXT:     *_d_vector_a = clad::identity_matrix(_d_vector_a->rows(), indepVarCount, {{0U|0UL|0ULL}});
// CHECK-NEXT:     *_d_vector_output = clad::identity_matrix(_d_vector_output->rows(), indepVarCount, _d_vector_a->rows() + {{1U|1UL|1ULL}});
// CHECK-NEXT:     double _t0 = a[0] * a[0];
// CHECK-NEXT:     *_d_vector_output[0] = ((*_d_vector_a[0]) * a[0] + a[0] * (*_d_vector_a[0])) * a[0] + _t0 * (*_d_vector_a[0]);
// CHECK-NEXT:     output[0] = _t0 * a[0];
// CHECK-NEXT:     double _t1 = a[0] * a[0];
// CHECK-NEXT:     double _t2 = b * b;
// CHECK-NEXT:     *_d_vector_output[1] = ((*_d_vector_a[0]) * a[0] + a[0] * (*_d_vector_a[0])) * a[0] + _t1 * (*_d_vector_a[0]) + (_d_vector_b * b + b * _d_vector_b) * b + _t2 * _d_vector_b;
// CHECK-NEXT:     output[1] = _t1 * a[0] + _t2 * b;
// CHECK-NEXT:     double _t3 = (a[0] + b);
// CHECK-NEXT:     *_d_vector_output[2] = (clad::zero_vector(indepVarCount)) * _t3 + 2 * (*_d_vector_a[0] + _d_vector_b);
// CHECK-NEXT:     output[2] = 2 * _t3;
// CHECK-NEXT: }

void f_7(double a, double& b, double&c) {
    b = 3 * a;
    c = -5 * b;
}

// CHECK: void f_7_jac(double a, double &b, double &c, clad::array<double> *_d_vector_b, clad::array<double> *_d_vector_c) {
// CHECK-NEXT:     unsigned long indepVarCount = {{3U|3UL|3ULL}};
// CHECK-NEXT:     clad::array<double> _d_vector_a = clad::one_hot_vector(indepVarCount, {{0U|0UL|0ULL}});
// CHECK-NEXT:     *_d_vector_b = clad::one_hot_vector(indepVarCount, {{1U|1UL|1ULL}});
// CHECK-NEXT:     *_d_vector_c = clad::one_hot_vector(indepVarCount, {{2U|2UL|2ULL}});
// CHECK-NEXT:     *_d_vector_b = (clad::zero_vector(indepVarCount)) * a + 3 * _d_vector_a;
// CHECK-NEXT:     b = 3 * a;
// CHECK-NEXT:     *_d_vector_c = - clad::zero_vector(indepVarCount) * b + -5 * *_d_vector_b;
// CHECK-NEXT:     c = -5 * b;
// CHECK-NEXT: }


#define TEST(F, ...) { \
  outputarr[0] = 0; outputarr[1] = 1; outputarr[2] = 0;\
  auto j = clad::jacobian(F);\
  j.execute(__VA_ARGS__, &result);\
  printf("Result is = {%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f}\n",\
  result[0][0], result[0][1], result[0][2],\
  result[1][0], result[1][1], result[1][2],\
  result[2][0], result[2][1], result[2][2]);\
}

#define TEST_F_1_SINGLE_PARAM(x, y, z) { \
  outputarr[0] = 0; outputarr[1] = 1; outputarr[2] = 0;\
  auto j = clad::jacobian(f_1,"a");\
  j.execute(x, y, z, outputarr, &result);\
  printf("Result is = {%.2f, %.2f, %.2f}\n",\
  result[0][0], result[1][0], result[2][0]);\
}


int main() {
  clad::matrix<double> result (3, 3);
  double outputarr[9];
  TEST(f_1, 1, 2, 3, outputarr); // CHECK-EXEC: Result is = {3.00, 0.00, 0.00, 3.00, 12.00, 0.00, -2.00, 0.00, 60.00}
  TEST(f_3, 1, 2, 3, outputarr); // CHECK-EXEC: Result is = {22.69, 0.00, 0.00, 0.00, -17.48, 0.00, 0.00, 0.00, -41.58}
  TEST(f_4, 1, 2, 3, outputarr); // CHECK-EXEC: Result is = {84.00, 42.00, 0.00, 0.00, 126.00, 84.00, 126.00, 0.00, 42.00}
  TEST_F_1_SINGLE_PARAM(1, 2, 3); // CHECK-EXEC: Result is = {3.00, 3.00, -2.00}

  auto df5 = clad::jacobian(f_5);
  result[0] = 0; result[1] = 0;
  df5.execute(3, outputarr, &result);
  printf("Result is = {%.2f, %.2f}\n", result[0][0], result[1][0]); // CHECK-EXEC: Result is = {6.00, 1.00}
  
  double a[] = {3, 1};
  clad::matrix<double> da (2, 5);
  TEST(f_6, a, 2, outputarr, &da); // CHECK-EXEC: Result is = {27.00, 0.00, 0.00, 27.00, 0.00, 12.00, 2.00, 0.00, 2.00}

  auto df7 = clad::jacobian(f_7);
  clad::array<double> db, dc;
  double a7=4, b=5, c=6;
  df7.execute(a7, b, c, &db, &dc);
  printf("Result is = {%.2f, %.2f}\n", db[0], dc[0]); // CHECK-EXEC: Result is = {3.00, -15.00}
  
}

// CHECK: clad::ValueAndPushforward<double, clad::array<double> > multiply_vector_pushforward(double x, double y, clad::array<double> _d_x, clad::array<double> _d_y) {
// CHECK-NEXT:     unsigned long indepVarCount = _d_y.size();
// CHECK-NEXT:     return {x * y, _d_x * y + x * _d_y};
// CHECK-NEXT: }
