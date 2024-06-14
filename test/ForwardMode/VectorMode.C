// RUN: %cladclang %s -I%S/../../include -oVectorMode.out 2>&1 | %filecheck %s
// RUN: ./VectorMode.out | %filecheck_exec %s

// XFAIL: asserts

//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

double f1(double x, double y) {
  return x*y*(x+y+1);
}

void f1_dvec(double x, double y, double *_d_x, double *_d_y);

// CHECK: void f1_dvec(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:   unsigned {{int|long}} indepVarCount = {{2U|2UL}};
// CHECK-NEXT:   clad::array<double> _d_vector_x = clad::one_hot_vector(indepVarCount, {{0U|0UL}});
// CHECK-NEXT:   clad::array<double> _d_vector_y = clad::one_hot_vector(indepVarCount, {{1U|1UL}});
// CHECK-NEXT:   double _t0 = x * y;
// CHECK-NEXT:   double _t1 = (x + y + 1);
// CHECK-NEXT:   {
// CHECK-NEXT:     clad::array<double> _d_vector_return(clad::array<double>(indepVarCount, (_d_vector_x * y + x * _d_vector_y) * _t1 + _t0 * (_d_vector_x + _d_vector_y + 0)));
// CHECK-NEXT:     *_d_x = _d_vector_return[{{0U|0UL}}];
// CHECK-NEXT:     *_d_y = _d_vector_return[{{1U|1UL}}];
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
// CHECK-NEXT:   unsigned {{int|long}} indepVarCount = {{2U|2UL}};
// CHECK-NEXT:   clad::array<double> _d_vector_x = clad::one_hot_vector(indepVarCount, {{0U|0UL}});
// CHECK-NEXT:   clad::array<double> _d_vector_y = clad::one_hot_vector(indepVarCount, {{1U|1UL}});
// CHECK-NEXT:   clad::array<double> _d_vector_temp1(clad::array<double>(indepVarCount, _d_vector_x * y + x * _d_vector_y));
// CHECK-NEXT:   double temp1 = x * y;
// CHECK-NEXT:   clad::array<double> _d_vector_temp2(clad::array<double>(indepVarCount, _d_vector_x + _d_vector_y + 0));
// CHECK-NEXT:   double temp2 = x + y + 1;
// CHECK-NEXT:   {
// CHECK-NEXT:     clad::array<double> _d_vector_return(clad::array<double>(indepVarCount, _d_vector_temp1 * temp2 + temp1 * _d_vector_temp2));
// CHECK-NEXT:     *_d_x = _d_vector_return[{{0U|0UL}}];
// CHECK-NEXT:     *_d_y = _d_vector_return[{{1U|1UL}}];
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
// CHECK-NEXT:   unsigned {{int|long}} indepVarCount = {{2U|2UL}};
// CHECK-NEXT:   clad::array<double> _d_vector_x = clad::one_hot_vector(indepVarCount, {{0U|0UL}});
// CHECK-NEXT:   clad::array<double> _d_vector_y = clad::one_hot_vector(indepVarCount, {{1U|1UL}});
// CHECK-NEXT:   if (y < 0) {
// CHECK-NEXT:     _d_vector_y = - _d_vector_y;
// CHECK-NEXT:     y = -y;
// CHECK-NEXT:   }
// CHECK-NEXT:   _d_vector_y += 0;
// CHECK-NEXT:   y += 1;
// CHECK-NEXT:   {
// CHECK-NEXT:     clad::array<double> _d_vector_return(clad::array<double>(indepVarCount, _d_vector_x * y + x * _d_vector_y));
// CHECK-NEXT:     *_d_x = _d_vector_return[{{0U|0UL}}];
// CHECK-NEXT:     *_d_y = _d_vector_return[{{1U|1UL}}];
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
// CHECK-NEXT:   unsigned {{int|long}} indepVarCount = {{2U|2UL}};
// CHECK-NEXT:   clad::array<double> _d_vector_lower = clad::one_hot_vector(indepVarCount, {{0U|0UL}});
// CHECK-NEXT:   clad::array<double> _d_vector_upper = clad::one_hot_vector(indepVarCount, {{1U|1UL}});
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
// CHECK-NEXT:       *_d_lower = _d_vector_return[{{0U|0UL}}];
// CHECK-NEXT:       *_d_upper = _d_vector_return[{{1U|1UL}}];
// CHECK-NEXT:       return;
// CHECK-NEXT:   }
// CHECK-NEXT: }

double f5(double x, double y, double z) {
  return 1.0*x + 2.0*y + 3.0*z;
}

// all
// CHECK: void f5_dvec(double x, double y, double z, double *_d_x, double *_d_y, double *_d_z) {
// CHECK-NEXT:   unsigned {{int|long}} indepVarCount = {{3U|3UL}};
// CHECK-NEXT:   clad::array<double> _d_vector_x = clad::one_hot_vector(indepVarCount, {{0U|0UL}});
// CHECK-NEXT:   clad::array<double> _d_vector_y = clad::one_hot_vector(indepVarCount, {{1U|1UL}});
// CHECK-NEXT:   clad::array<double> _d_vector_z = clad::one_hot_vector(indepVarCount, {{2U|2UL}});
// CHECK-NEXT:   {
// CHECK-NEXT:     clad::array<double> _d_vector_return(clad::array<double>(indepVarCount, 0. * x + 1. * _d_vector_x + 0. * y + 2. * _d_vector_y + 0. * z + 3. * _d_vector_z));
// CHECK-NEXT:     *_d_x = _d_vector_return[{{0U|0UL}}];
// CHECK-NEXT:     *_d_y = _d_vector_return[{{1U|1UL}}];
// CHECK-NEXT:     *_d_z = _d_vector_return[{{2U|2UL}}];
// CHECK-NEXT:     return;
// CHECK-NEXT:   }
// CHECK-NEXT: }

// x, y
// CHECK: void f5_dvec_0_1(double x, double y, double z, double *_d_x, double *_d_y) {
// CHECK-NEXT:   unsigned {{int|long}} indepVarCount = {{2U|2UL}};
// CHECK-NEXT:   clad::array<double> _d_vector_x = clad::one_hot_vector(indepVarCount, {{0U|0UL}});
// CHECK-NEXT:   clad::array<double> _d_vector_y = clad::one_hot_vector(indepVarCount, {{1U|1UL}});
// CHECK-NEXT:   clad::array<double> _d_vector_z = clad::zero_vector(indepVarCount);
// CHECK-NEXT:   {
// CHECK-NEXT:     clad::array<double> _d_vector_return(clad::array<double>(indepVarCount, 0. * x + 1. * _d_vector_x + 0. * y + 2. * _d_vector_y + 0. * z + 3. * _d_vector_z));
// CHECK-NEXT:     *_d_x = _d_vector_return[{{0U|0UL}}];
// CHECK-NEXT:     *_d_y = _d_vector_return[{{1U|1UL}}];
// CHECK-NEXT:     return;
// CHECK-NEXT:   }
// CHECK-NEXT: }

// x, z
// CHECK: void f5_dvec_0_2(double x, double y, double z, double *_d_x, double *_d_z) {
// CHECK-NEXT:   unsigned {{int|long}} indepVarCount = {{2U|2UL}};
// CHECK-NEXT:   clad::array<double> _d_vector_x = clad::one_hot_vector(indepVarCount, {{0U|0UL}});
// CHECK-NEXT:   clad::array<double> _d_vector_y = clad::zero_vector(indepVarCount);
// CHECK-NEXT:   clad::array<double> _d_vector_z = clad::one_hot_vector(indepVarCount, {{1U|1UL}});
// CHECK-NEXT:   {
// CHECK-NEXT:     clad::array<double> _d_vector_return(clad::array<double>(indepVarCount, 0. * x + 1. * _d_vector_x + 0. * y + 2. * _d_vector_y + 0. * z + 3. * _d_vector_z));
// CHECK-NEXT:     *_d_x = _d_vector_return[{{0U|0UL}}];
// CHECK-NEXT:     *_d_z = _d_vector_return[{{1U|1UL}}];
// CHECK-NEXT:     return;
// CHECK-NEXT:   }
// CHECK-NEXT: }

// y, z
// CHECK: void f5_dvec_1_2(double x, double y, double z, double *_d_y, double *_d_z) {
// CHECK-NEXT:   unsigned {{int|long}} indepVarCount = {{2U|2UL}};
// CHECK-NEXT:   clad::array<double> _d_vector_x = clad::zero_vector(indepVarCount);
// CHECK-NEXT:   clad::array<double> _d_vector_y = clad::one_hot_vector(indepVarCount, {{0U|0UL}});
// CHECK-NEXT:   clad::array<double> _d_vector_z = clad::one_hot_vector(indepVarCount, {{1U|1UL}});
// CHECK-NEXT:   {
// CHECK-NEXT:     clad::array<double> _d_vector_return(clad::array<double>(indepVarCount, 0. * x + 1. * _d_vector_x + 0. * y + 2. * _d_vector_y + 0. * z + 3. * _d_vector_z));
// CHECK-NEXT:     *_d_y = _d_vector_return[{{0U|0UL}}];
// CHECK-NEXT:     *_d_z = _d_vector_return[{{1U|1UL}}];
// CHECK-NEXT:     return;
// CHECK-NEXT:   }
// CHECK-NEXT: }

// z
// CHECK: void f5_dvec_2(double x, double y, double z, double *_d_z) {
// CHECK-NEXT:   unsigned {{int|long}} indepVarCount = {{1U|1UL}};
// CHECK-NEXT:   clad::array<double> _d_vector_x = clad::zero_vector(indepVarCount);
// CHECK-NEXT:   clad::array<double> _d_vector_y = clad::zero_vector(indepVarCount);
// CHECK-NEXT:   clad::array<double> _d_vector_z = clad::one_hot_vector(indepVarCount, {{0U|0UL}});
// CHECK-NEXT:   {
// CHECK-NEXT:     clad::array<double> _d_vector_return(clad::array<double>(indepVarCount, 0. * x + 1. * _d_vector_x + 0. * y + 2. * _d_vector_y + 0. * z + 3. * _d_vector_z));
// CHECK-NEXT:     *_d_z = _d_vector_return[{{0U|0UL}}];
// CHECK-NEXT:     return;
// CHECK-NEXT:   }
// CHECK-NEXT: }

double square(const double& x) {
  double z = x*x;
  return z;
}

// CHECK: clad::ValueAndPushforward<double, clad::array<double> > square_vector_pushforward(const double &x, const clad::array<double> &_d_x);

double f6(double x, double y) {
  return square(x) + square(y);
}

// CHECK: void f6_dvec(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:    unsigned {{int|long}} indepVarCount = {{2U|2UL}};
// CHECK-NEXT:    clad::array<double> _d_vector_x = clad::one_hot_vector(indepVarCount, {{0U|0UL}});
// CHECK-NEXT:    clad::array<double> _d_vector_y = clad::one_hot_vector(indepVarCount, {{1U|1UL}});
// CHECK-NEXT:    clad::ValueAndPushforward<double, clad::array<double> > _t0 = square_vector_pushforward(x, _d_vector_x);
// CHECK-NEXT:    clad::ValueAndPushforward<double, clad::array<double> > _t1 = square_vector_pushforward(y, _d_vector_y);
// CHECK-NEXT:    {
// CHECK-NEXT:        clad::array<double> _d_vector_return(clad::array<double>(indepVarCount, _t0.pushforward + _t1.pushforward));
// CHECK-NEXT:        *_d_x = _d_vector_return[{{0U|0UL}}];
// CHECK-NEXT:        *_d_y = _d_vector_return[{{1U|1UL}}];
// CHECK-NEXT:        return;
// CHECK-NEXT:    }
// CHECK-NEXT: }

double weighted_array_squared_sum(const double* arr, double w, int n) {
  double sum = 0;
  for (int i = 0; i < n; ++i) {
    sum += w * square(arr[i]);
  }
  return sum;
}

double f7(const double* arr, double w, int n) {
  return weighted_array_squared_sum(arr, w, n);
}
// CHECK: clad::ValueAndPushforward<double, clad::array<double> > weighted_array_squared_sum_vector_pushforward(const double *arr, double w, int n, clad::matrix<double> &_d_arr, clad::array<double> _d_w, clad::array<int> _d_n);

// CHECK: void f7_dvec_0_1(const double *arr, double w, int n, clad::array_ref<double> _d_arr, double *_d_w) {
// CHECK-NEXT:    unsigned {{int|long}} indepVarCount = _d_arr.size() + {{1U|1UL}};
// CHECK-NEXT:    clad::matrix<double> _d_vector_arr = clad::identity_matrix(_d_arr.size(), indepVarCount, {{0U|0UL}});
// CHECK-NEXT:    clad::array<double> _d_vector_w = clad::one_hot_vector(indepVarCount, _d_arr.size());
// CHECK-NEXT:    clad::array<int> _d_vector_n = clad::zero_vector(indepVarCount);
// CHECK-NEXT:    clad::ValueAndPushforward<double, clad::array<double> > _t0 = weighted_array_squared_sum_vector_pushforward(arr, w, n, _d_vector_arr, _d_vector_w, _d_vector_n);
// CHECK-NEXT:    {
// CHECK-NEXT:        clad::array<double> _d_vector_return(clad::array<double>(indepVarCount, _t0.pushforward));
// CHECK-NEXT:        _d_arr = _d_vector_return.slice({{0U|0UL}}, _d_arr.size());
// CHECK-NEXT:        *_d_w = _d_vector_return[_d_arr.size()];
// CHECK-NEXT:        return;
// CHECK-NEXT:    }
// CHECK-NEXT: }

void sum_ref(double& res, int n, const double* arr) {
  for(int i=0; i<n; ++i) {
    res += arr[i];
  }
  return;
}

// CHECK: void sum_ref_vector_pushforward(double &res, int n, const double *arr, clad::array<double> &_d_res, clad::array<int> _d_n, clad::matrix<double> &_d_arr);

double f8(int n, const double* arr) {
  double res = 0;
  sum_ref(res, n, arr);
  return res;
}

// CHECK: void f8_dvec_1(int n, const double *arr, clad::array_ref<double> _d_arr) {
// CHECK-NEXT:     unsigned {{int|long}} indepVarCount = _d_arr.size();
// CHECK-NEXT:     clad::array<int> _d_vector_n = clad::zero_vector(indepVarCount);
// CHECK-NEXT:     clad::matrix<double> _d_vector_arr = clad::identity_matrix(_d_arr.size(), indepVarCount, {{0U|0UL}});
// CHECK-NEXT:     clad::array<double> _d_vector_res(clad::array<double>(indepVarCount, 0));
// CHECK-NEXT:     double res = 0;
// CHECK-NEXT:     sum_ref_vector_pushforward(res, n, arr, _d_vector_res, _d_vector_n, _d_vector_arr);
// CHECK-NEXT:     {
// CHECK-NEXT:         clad::array<double> _d_vector_return(clad::array<double>(indepVarCount, _d_vector_res));
// CHECK-NEXT:         _d_arr = _d_vector_return.slice({{0U|0UL}}, _d_arr.size());
// CHECK-NEXT:         return;
// CHECK-NEXT:     }
// CHECK-NEXT: }

namespace clad {
  namespace custom_derivatives{
    void f9_dvec(double x, double y, double *d_x, double *d_y) {
      *d_x += 1;
      *d_y += 1;
    }
  }
}

double f9(double x, double y) {
  return x + y;
}

// CHECK: void f9_dvec(double x, double y, double *d_x, double *d_y) {
// CHECK-NEXT:   *d_x += 1;
// CHECK-NEXT:   *d_y += 1;
// CHECK-NEXT: }

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

  // Testing derivatives of function calls.
  auto f6_dvec = clad::differentiate<clad::opts::vector_mode>(f6);
  f6_dvec.execute(1, 2, &result[0], &result[1]);
  printf("Result is = {%.2f, %.2f}\n", result[0], result[1]); // CHECK-EXEC: Result is = {2.00, 4.00}

  // Testing derivatives of function calls with array parameters.
  auto f7_dvec = clad::differentiate<clad::opts::vector_mode>(f7, "arr,w");
  double arr[3] = {1, 2, 3};
  double w = 2, dw = 0;
  double darr[3] = {0, 0, 0};
  clad::array_ref<double> darr_ref(darr, 3);
  f7_dvec.execute(arr, 2, 3, darr_ref, &dw);
  printf("Result is = {%.2f, %.2f, %.2f, %.2f}\n", darr[0], darr[1], darr[2], dw); // CHECK-EXEC: Result is = {4.00, 8.00, 12.00, 14.00}

  // Testing derivatives of function calls with array and reference parameters.
  auto f8_dvec = clad::differentiate<clad::opts::vector_mode>(f8, "arr");
  double arr2[3] = {1, 2, 3};
  double darr2[3] = {0, 0, 0};
  clad::array_ref<double> darr2_ref(darr2, 3);
  f8_dvec.execute(3, arr2, darr2_ref);
  printf("Result is = {%.2f, %.2f, %.2f}\n", darr2[0], darr2[1], darr2[2]); // CHECK-EXEC: Result is = {1.00, 1.00, 1.00}

  auto f9_dvec = clad::differentiate<clad::opts::vector_mode>(f9);
  double dx = 0, dy = 0;
  f9_dvec.execute(1, 2, &dx, &dy);
  printf("Result is = {%.2f, %.2f}\n", dx, dy); // CHECK-EXEC: Result is = {1.00, 1.00}

// CHECK: clad::ValueAndPushforward<double, clad::array<double> > square_vector_pushforward(const double &x, const clad::array<double> &_d_x) {
// CHECK-NEXT:    unsigned long indepVarCount = _d_x.size();
// CHECK-NEXT:    clad::array<double> _d_vector_z(clad::array<double>(indepVarCount, _d_x * x + x * _d_x));
// CHECK-NEXT:    double z = x * x;
// CHECK-NEXT:    return {z, _d_vector_z};
// CHECK-NEXT: }

// CHECK: clad::ValueAndPushforward<double, clad::array<double> > weighted_array_squared_sum_vector_pushforward(const double *arr, double w, int n, clad::matrix<double> &_d_arr, clad::array<double> _d_w, clad::array<int> _d_n) {
// CHECK-NEXT:    unsigned long indepVarCount = _d_n.size();
// CHECK-NEXT:    clad::array<double> _d_vector_sum(clad::array<double>(indepVarCount, 0));
// CHECK-NEXT:    double sum = 0;
// CHECK-NEXT:    {
// CHECK-NEXT:        clad::array<int> _d_vector_i(clad::array<int>(indepVarCount, 0));
// CHECK-NEXT:        for (int i = 0; i < n; ++i) {
// CHECK-NEXT:            clad::ValueAndPushforward<double, clad::array<double> > _t0 = square_vector_pushforward(arr[i], _d_arr[i]);
// CHECK-NEXT:            double &_t1 = _t0.value;
// CHECK-NEXT:            _d_vector_sum += _d_w * _t1 + w * _t0.pushforward;
// CHECK-NEXT:            sum += w * _t1;
// CHECK-NEXT:        }
// CHECK-NEXT:    }
// CHECK-NEXT:    return {sum, _d_vector_sum};
// CHECK-NEXT: }

// CHECK: void sum_ref_vector_pushforward(double &res, int n, const double *arr, clad::array<double> &_d_res, clad::array<int> _d_n, clad::matrix<double> &_d_arr) {
// CHECK-NEXT:    unsigned long indepVarCount = _d_arr[0].size();
// CHECK-NEXT:    {
// CHECK-NEXT:        clad::array<int> _d_vector_i(clad::array<int>(indepVarCount, 0));
// CHECK-NEXT:        for (int i = 0; i < n; ++i) {
// CHECK-NEXT:            _d_res += _d_arr[i];
// CHECK-NEXT:            res += arr[i];
// CHECK-NEXT:        }
// CHECK-NEXT:    }
// CHECK-NEXT: }
}
