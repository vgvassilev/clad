// RUN: %cladclang %s -I%S/../../include -oVectorMode.out 2>&1 | FileCheck %s
// RUN: ./VectorMode.out | FileCheck -check-prefix=CHECK-EXEC %s

//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

double f1(double x, double y) {
  return x*y*(x+y+1);
}

void f1_dvec(double x, double y, double *_d_x, double *_d_y);

// CHECK: void f1_dvec(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:   unsigned long indepVarCount = 2UL;
// CHECK-NEXT:   clad::array<double> _d_vector_x = clad::one_hot_vector(indepVarCount, 0UL);
// CHECK-NEXT:   clad::array<double> _d_vector_y = clad::one_hot_vector(indepVarCount, 1UL);
// CHECK-NEXT:   double _t0 = x * y;
// CHECK-NEXT:   double _t1 = (x + y + 1);
// CHECK-NEXT:   {
// CHECK-NEXT:     clad::array<double> _d_vector_return(clad::array<double>(indepVarCount, (_d_vector_x * y + x * _d_vector_y) * _t1 + _t0 * (_d_vector_x + _d_vector_y + 0)));
// CHECK-NEXT:     *_d_x = _d_vector_return[0UL];
// CHECK-NEXT:     *_d_y = _d_vector_return[1UL];
// CHECK-NEXT:     return;
// CHECK-NEXT:   }
// CHECK-NEXT: }

double f2(double x, double y) {
  // to test usage of local variables.
  double temp1 = x*y;
  double temp2 = x+y+1;
  return temp1*temp2;
}

void f2_dvec(double x, double y, double *_d_x, double *_d_y);

// CHECK: void f2_dvec(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:   unsigned long indepVarCount = 2UL;
// CHECK-NEXT:   clad::array<double> _d_vector_x = clad::one_hot_vector(indepVarCount, 0UL);
// CHECK-NEXT:   clad::array<double> _d_vector_y = clad::one_hot_vector(indepVarCount, 1UL);
// CHECK-NEXT:   clad::array<double> _d_vector_temp1(clad::array<double>(indepVarCount, _d_vector_x * y + x * _d_vector_y));
// CHECK-NEXT:   double temp1 = x * y;
// CHECK-NEXT:   clad::array<double> _d_vector_temp2(clad::array<double>(indepVarCount, _d_vector_x + _d_vector_y + 0));
// CHECK-NEXT:   double temp2 = x + y + 1;
// CHECK-NEXT:   {
// CHECK-NEXT:     clad::array<double> _d_vector_return(clad::array<double>(indepVarCount, _d_vector_temp1 * temp2 + temp1 * _d_vector_temp2));
// CHECK-NEXT:     *_d_x = _d_vector_return[0UL];
// CHECK-NEXT:     *_d_y = _d_vector_return[1UL];
// CHECK-NEXT:     return;
// CHECK-NEXT:   }
// CHECK-NEXT: }

double f3(double x, double y) {
  // x * (abs(y) + 1)
  if (y < 0) // to test if statements.
    y = -y;
  y += 1;
  return x*y;
}

void f3_dvec(double x, double y, double *_d_x, double *_d_y);

// CHECK: void f3_dvec(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:   unsigned long indepVarCount = 2UL;
// CHECK-NEXT:   clad::array<double> _d_vector_x = clad::one_hot_vector(indepVarCount, 0UL);
// CHECK-NEXT:   clad::array<double> _d_vector_y = clad::one_hot_vector(indepVarCount, 1UL);
// CHECK-NEXT:   if (y < 0) {
// CHECK-NEXT:     _d_vector_y = - _d_vector_y;
// CHECK-NEXT:     y = -y;
// CHECK-NEXT:   }
// CHECK-NEXT:   _d_vector_y += 0;
// CHECK-NEXT:   y += 1;
// CHECK-NEXT:   {
// CHECK-NEXT:     clad::array<double> _d_vector_return(clad::array<double>(indepVarCount, _d_vector_x * y + x * _d_vector_y));
// CHECK-NEXT:     *_d_x = _d_vector_return[0UL];
// CHECK-NEXT:     *_d_y = _d_vector_return[1UL];
// CHECK-NEXT:     return;
// CHECK-NEXT:   }
// CHECK-NEXT: }

double f4(double lower, double upper) {
  // integral of x^2 using reimann sum
  double sum = 0;
  double num_points = 10000;
  double interval = (upper - lower) / num_points;
  for (double x = lower; x <= upper; x += interval) {
    sum += x * x * interval;
  }
  return sum;
}

void f4_dvec(double lower, double upper, double *_d_lower, double *_d_upper);

// CHECK: void f4_dvec(double lower, double upper, double *_d_lower, double *_d_upper) {
// CHECK-NEXT:   unsigned long indepVarCount = 2UL;
// CHECK-NEXT:   clad::array<double> _d_vector_lower = clad::one_hot_vector(indepVarCount, 0UL);
// CHECK-NEXT:   clad::array<double> _d_vector_upper = clad::one_hot_vector(indepVarCount, 1UL);
// CHECK-NEXT:   clad::array<double> _d_vector_sum(clad::array<double>(indepVarCount, 0));
// CHECK-NEXT:   double sum = 0;
// CHECK-NEXT:   clad::array<double> _d_vector_num_points(clad::array<double>(indepVarCount, 0));
// CHECK-NEXT:   double num_points = 10000;
// CHECK-NEXT:   double _t0 = (upper - lower);
// CHECK-NEXT:   clad::array<double> _d_vector_interval(clad::array<double>(indepVarCount, ((_d_vector_upper - _d_vector_lower) * num_points - _t0 * _d_vector_num_points) / (num_points * num_points)));
// CHECK-NEXT:   double interval = _t0 / num_points;
// CHECK-NEXT:   {
// CHECK-NEXT:       clad::array<double> _d_vector_x(clad::array<double>(indepVarCount, _d_vector_lower));
// CHECK-NEXT:       for (double x = lower; x <= upper; (_d_vector_x += _d_vector_interval) , (x += interval)) {
// CHECK-NEXT:           double _t1 = x * x;
// CHECK-NEXT:           _d_vector_sum += (_d_vector_x * x + x * _d_vector_x) * interval + _t1 * _d_vector_interval;
// CHECK-NEXT:           sum += _t1 * interval;
// CHECK-NEXT:       }
// CHECK-NEXT:   }
// CHECK-NEXT:   {
// CHECK-NEXT:       clad::array<double> _d_vector_return(clad::array<double>(indepVarCount, _d_vector_sum));
// CHECK-NEXT:       *_d_lower = _d_vector_return[0UL];
// CHECK-NEXT:       *_d_upper = _d_vector_return[1UL];
// CHECK-NEXT:       return;
// CHECK-NEXT:   }
// CHECK-NEXT: }

double f5(double x, double y, double z) {
  return 1.0*x + 2.0*y + 3.0*z;
}

// all
// CHECK: void f5_dvec(double x, double y, double z, double *_d_x, double *_d_y, double *_d_z) {
// CHECK-NEXT:   unsigned long indepVarCount = 3UL;
// CHECK-NEXT:   clad::array<double> _d_vector_x = clad::one_hot_vector(indepVarCount, 0UL);
// CHECK-NEXT:   clad::array<double> _d_vector_y = clad::one_hot_vector(indepVarCount, 1UL);
// CHECK-NEXT:   clad::array<double> _d_vector_z = clad::one_hot_vector(indepVarCount, 2UL);
// CHECK-NEXT:   {
// CHECK-NEXT:     clad::array<double> _d_vector_return(clad::array<double>(indepVarCount, 0. * x + 1. * _d_vector_x + 0. * y + 2. * _d_vector_y + 0. * z + 3. * _d_vector_z));
// CHECK-NEXT:     *_d_x = _d_vector_return[0UL];
// CHECK-NEXT:     *_d_y = _d_vector_return[1UL];
// CHECK-NEXT:     *_d_z = _d_vector_return[2UL];
// CHECK-NEXT:     return;
// CHECK-NEXT:   }

// x, y
// CHECK: void f5_dvec_0_1(double x, double y, double z, double *_d_x, double *_d_y) {
// CHECK-NEXT:   unsigned long indepVarCount = 2UL;
// CHECK-NEXT:   clad::array<double> _d_vector_x = clad::one_hot_vector(indepVarCount, 0UL);
// CHECK-NEXT:   clad::array<double> _d_vector_y = clad::one_hot_vector(indepVarCount, 1UL);
// CHECK-NEXT:   clad::array<double> _d_vector_z = clad::zero_vector(indepVarCount);
// CHECK-NEXT:   {
// CHECK-NEXT:     clad::array<double> _d_vector_return(clad::array<double>(indepVarCount, 0. * x + 1. * _d_vector_x + 0. * y + 2. * _d_vector_y + 0. * z + 3. * _d_vector_z));
// CHECK-NEXT:     *_d_x = _d_vector_return[0UL];
// CHECK-NEXT:     *_d_y = _d_vector_return[1UL];
// CHECK-NEXT:     return;
// CHECK-NEXT:   }

// x, z
// CHECK: void f5_dvec_0_2(double x, double y, double z, double *_d_x, double *_d_z) {
// CHECK-NEXT:   unsigned long indepVarCount = 2UL;
// CHECK-NEXT:   clad::array<double> _d_vector_x = clad::one_hot_vector(indepVarCount, 0UL);
// CHECK-NEXT:   clad::array<double> _d_vector_y = clad::zero_vector(indepVarCount);
// CHECK-NEXT:   clad::array<double> _d_vector_z = clad::one_hot_vector(indepVarCount, 1UL);
// CHECK-NEXT:   {
// CHECK-NEXT:     clad::array<double> _d_vector_return(clad::array<double>(indepVarCount, 0. * x + 1. * _d_vector_x + 0. * y + 2. * _d_vector_y + 0. * z + 3. * _d_vector_z));
// CHECK-NEXT:     *_d_x = _d_vector_return[0UL];
// CHECK-NEXT:     *_d_z = _d_vector_return[1UL];
// CHECK-NEXT:     return;
// CHECK-NEXT:   }

// y, z
// CHECK: void f5_dvec_1_2(double x, double y, double z, double *_d_y, double *_d_z) {
// CHECK-NEXT:   unsigned long indepVarCount = 2UL;
// CHECK-NEXT:   clad::array<double> _d_vector_x = clad::zero_vector(indepVarCount);
// CHECK-NEXT:   clad::array<double> _d_vector_y = clad::one_hot_vector(indepVarCount, 0UL);
// CHECK-NEXT:   clad::array<double> _d_vector_z = clad::one_hot_vector(indepVarCount, 1UL);
// CHECK-NEXT:   {
// CHECK-NEXT:     clad::array<double> _d_vector_return(clad::array<double>(indepVarCount, 0. * x + 1. * _d_vector_x + 0. * y + 2. * _d_vector_y + 0. * z + 3. * _d_vector_z));
// CHECK-NEXT:     *_d_y = _d_vector_return[0UL];
// CHECK-NEXT:     *_d_z = _d_vector_return[1UL];
// CHECK-NEXT:     return;
// CHECK-NEXT:   }

// z
// CHECK: void f5_dvec_2(double x, double y, double z, double *_d_z) {
// CHECK-NEXT:   unsigned long indepVarCount = 1UL;
// CHECK-NEXT:   clad::array<double> _d_vector_x = clad::zero_vector(indepVarCount);
// CHECK-NEXT:   clad::array<double> _d_vector_y = clad::zero_vector(indepVarCount);
// CHECK-NEXT:   clad::array<double> _d_vector_z = clad::one_hot_vector(indepVarCount, 0UL);
// CHECK-NEXT:   {
// CHECK-NEXT:     clad::array<double> _d_vector_return(clad::array<double>(indepVarCount, 0. * x + 1. * _d_vector_x + 0. * y + 2. * _d_vector_y + 0. * z + 3. * _d_vector_z));
// CHECK-NEXT:     *_d_z = _d_vector_return[0UL];
// CHECK-NEXT:     return;
// CHECK-NEXT:   }


#define TEST(F, x, y)                                                          \
  {                                                                            \
    result[0] = 0;                                                             \
    result[1] = 0;                                                             \
    clad::differentiate<clad::opts::vector_mode>(F);                        \
    F##_dvec(x, y, &result[0], &result[1]);                                    \
    printf("Result is = {%.2f, %.2f}\n", result[0], result[1]);                \
  }

int main() {
  double result[3];

  TEST(f1, 3, 4); // CHECK-EXEC: Result is = {44.00, 36.00}
  TEST(f2, 3, 4); // CHECK-EXEC: Result is = {44.00, 36.00}
  TEST(f3, 3, -4); // CHECK-EXEC: Result is = {5.00, -3.00}
  TEST(f4, 1, 2); // CHECK-EXEC: Result is = {-1.00, 4.00}

  // Testing derivatives of partial parameters.
  auto f_dvec_x_y_z = clad::differentiate<clad::opts::vector_mode>(f5, "x, y, z");
  f_dvec_x_y_z.execute(1, 2, 3, &result[0], &result[1], &result[2]);
  printf("Result is = {%.2f, %.2f, %.2f}\n", result[0], result[1], result[2]); // CHECK-EXEC: Result is = {1.00, 2.00, 3.00}

  auto f_dvec_x_y = clad::differentiate<clad::opts::vector_mode>(f5, "x, y");
  f_dvec_x_y.execute(1, 2, 3, &result[0], &result[1]);
  printf("Result is = {%.2f, %.2f}\n", result[0], result[1]); // CHECK-EXEC: Result is = {1.00, 2.00}

  auto f_dvec_x_z = clad::differentiate<clad::opts::vector_mode>(f5, "x, z");
  f_dvec_x_z.execute(1, 2, 3, &result[0], &result[1]);
  printf("Result is = {%.2f, %.2f}\n", result[0], result[1]); // CHECK-EXEC: Result is = {1.00, 3.00}

  auto f_dvec_y_z = clad::differentiate<clad::opts::vector_mode>(f5, "y, z");
  f_dvec_y_z.execute(1, 2, 3, &result[0], &result[1]);
  printf("Result is = {%.2f, %.2f}\n", result[0], result[1]); // CHECK-EXEC: Result is = {2.00, 3.00}

  auto f_dvec_y_x = clad::differentiate<clad::opts::vector_mode>(f5, "y, x");
  f_dvec_y_x.execute(1, 2, 3, &result[0], &result[1]);

  auto f_dvec_z = clad::differentiate<clad::opts::vector_mode>(f5, "z");
  f_dvec_z.execute(1, 2, 3, &result[0]);
  printf("Result is = {%.2f}\n", result[0]); // CHECK-EXEC: Result is = {3.00}
}
