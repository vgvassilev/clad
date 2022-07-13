// RUN: %cladclang %s -I%S/../../include -oFunctionCallsWithResults.out 
// RUN: ./FunctionCallsWithResults.out | FileCheck -check-prefix=CHECK-EXEC %s

#include "clad/Differentiator/Differentiator.h"

int printf(const char* fmt, ...);

float custom_fn(float x) {
  return x;
}

int custom_fn(int x) {
  return x;
}

float overloaded(float x) {
  return x;
}

float overloaded() {
  return 3;
}

namespace clad {
namespace custom_derivatives {
clad::ValueAndPushforward<int, int> custom_fn_pushforward(int x, int d_x) {
  return {custom_fn(x), x * d_x};
}

clad::ValueAndPushforward<float, float> custom_fn_pushforward(float x,
                                                              float d_x) {
  return {custom_fn(x), d_x};
}

clad::ValueAndPushforward<int, int> custom_fn_pushforward() { return {0, 5}; }

clad::ValueAndPushforward<float, float> overloaded_pushforward(float x,
                                                               float d_x) {
  printf("A was called.\n");
  return {overloaded(x), d_x};
}

clad::ValueAndPushforward<float, float> overloaded_pushforward() {
  float x = 2;
  printf("A was called.\n");
  return {overloaded(), x * x};
}
} // namespace custom_derivatives
} // namespace clad

float test_1(float x) {
  return overloaded(x) + custom_fn(x);
}

// CHECK: float test_1_darg0(float x) {
// CHECK-NEXT: float _d_x = 1;
// CHECK-NEXT: clad::ValueAndPushforward<float, float> _t0 = clad::custom_derivatives::overloaded_pushforward(x, _d_x);
// CHECK-NEXT: clad::ValueAndPushforward<float, float> _t1 = clad::custom_derivatives::custom_fn_pushforward(x, _d_x);
// CHECK-NEXT: return _t0.pushforward + _t1.pushforward;
// CHECK-NEXT: }

float test_2(float x) {
  return overloaded(x) + custom_fn(x);
}

// CHECK: float test_2_darg0(float x) {
// CHECK-NEXT: float _d_x = 1;
// CHECK-NEXT: clad::ValueAndPushforward<float, float> _t0 = clad::custom_derivatives::overloaded_pushforward(x, _d_x);
// CHECK-NEXT: clad::ValueAndPushforward<float, float> _t1 = clad::custom_derivatives::custom_fn_pushforward(x, _d_x);
// CHECK-NEXT: return _t0.pushforward + _t1.pushforward;
// CHECK-NEXT: }

float test_4(float x) {
  return overloaded();
}

// CHECK: float test_4_darg0(float x) {
// CHECK-NEXT: float _d_x = 1;
// CHECK-NEXT: clad::ValueAndPushforward<float, float> _t0 = clad::custom_derivatives::overloaded_pushforward();
// CHECK-NEXT: return _t0.pushforward;
// CHECK-NEXT: }

double sum_of_squares(double u, double v) {
  return u*u + v*v;
}

// CHECK: clad::ValueAndPushforward<double, double> sum_of_squares_pushforward(double u, double v, double _d_u, double _d_v) {
// CHECK-NEXT:     return {u * u + v * v, _d_u * u + u * _d_u + _d_v * v + v * _d_v};
// CHECK-NEXT: }

double fn1(double i, double j) {
  double res = sum_of_squares(i, j);
  res += sum_of_squares(j, i);
  return res;
}

// CHECK: double fn1_darg0(double i, double j) {
// CHECK-NEXT:     double _d_i = 1;
// CHECK-NEXT:     double _d_j = 0;
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t0 = sum_of_squares_pushforward(i, j, _d_i, _d_j);
// CHECK-NEXT:     double _d_res = _t0.pushforward;
// CHECK-NEXT:     double res = _t0.value;
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t1 = sum_of_squares_pushforward(j, i, _d_j, _d_i);
// CHECK-NEXT:     _d_res += _t1.pushforward;
// CHECK-NEXT:     res += _t1.value;
// CHECK-NEXT:     return _d_res;
// CHECK-NEXT: }

// CHECK: clad::ValueAndPushforward<double, double> fn1_pushforward(double i, double j, double _d_i, double _d_j) {
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t0 = sum_of_squares_pushforward(i, j, _d_i, _d_j);
// CHECK-NEXT:     double _d_res = _t0.pushforward;
// CHECK-NEXT:     double res = _t0.value;
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t1 = sum_of_squares_pushforward(j, i, _d_j, _d_i);
// CHECK-NEXT:     _d_res += _t1.pushforward;
// CHECK-NEXT:     res += _t1.value;
// CHECK-NEXT:     return {res, _d_res};
// CHECK-NEXT: }

double sum_of_pairwise_product(double u, double v, double w) {
  return u*v + v*w + w*u;
}

// CHECK: clad::ValueAndPushforward<double, double> sum_of_pairwise_product_pushforward(double u, double v, double w, double _d_u, double _d_v, double _d_w) {
// CHECK-NEXT:     return {u * v + v * w + w * u, _d_u * v + u * _d_v + _d_v * w + v * _d_w + _d_w * u + w * _d_u};
// CHECK-NEXT: }

double fn2(double i, double j) {
  double res = fn1(i, j);
  res += sum_of_pairwise_product(res, i, j);
  return res;
}

// CHECK: double fn2_darg0(double i, double j) {
// CHECK-NEXT:     double _d_i = 1;
// CHECK-NEXT:     double _d_j = 0;
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t0 = fn1_pushforward(i, j, _d_i, _d_j);
// CHECK-NEXT:     double _d_res = _t0.pushforward;
// CHECK-NEXT:     double res = _t0.value;
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t1 = sum_of_pairwise_product_pushforward(res, i, j, _d_res, _d_i, _d_j);
// CHECK-NEXT:     _d_res += _t1.pushforward;
// CHECK-NEXT:     res += _t1.value;
// CHECK-NEXT:     return _d_res;
// CHECK-NEXT: }

void square_inplace(double& u) {
  u = u*u;
}  

// CHECK: void square_inplace_pushforward(double &u, double &_d_u) {
// CHECK-NEXT:     _d_u = _d_u * u + u * _d_u;
// CHECK-NEXT:     u = u * u;
// CHECK-NEXT: }

double fn3(double i, double j) {
  square_inplace(i);
  double res = i;
  return res;
}

// CHECK: double fn3_darg0(double i, double j) {
// CHECK-NEXT:     double _d_i = 1;
// CHECK-NEXT:     double _d_j = 0;
// CHECK-NEXT:     square_inplace_pushforward(i, _d_i);
// CHECK-NEXT:     double _d_res = _d_i;
// CHECK-NEXT:     double res = i;
// CHECK-NEXT:     return _d_res;
// CHECK-NEXT: }

double nonRealParamFn(const char* a, const char* b = nullptr) {
  return 1;
}

// CHECK: clad::ValueAndPushforward<double, double> nonRealParamFn_pushforward(const char *a, const char *b, const char *_d_a, const char *_d_b) {
// CHECK-NEXT:     return {1, 0};
// CHECK-NEXT: }

double fn4(double i, double j) {
  double res = nonRealParamFn(0, 0);
  res += i;
  return res;
}

// CHECK: double fn4_darg0(double i, double j) {
// CHECK-NEXT:     double _d_i = 1;
// CHECK-NEXT:     double _d_j = 0;
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t0 = nonRealParamFn_pushforward(0, 0, 0, 0);
// CHECK-NEXT:     double _d_res = _t0.pushforward;
// CHECK-NEXT:     double res = _t0.value;
// CHECK-NEXT:     _d_res += _d_i;
// CHECK-NEXT:     res += i;
// CHECK-NEXT:     return _d_res;
// CHECK-NEXT: }

double fn5(double i, double j) {
  if (i < 2) {
    return j;
  }
  return fn5(0, i);
}

// CHECK: clad::ValueAndPushforward<double, double> fn5_pushforward(double i, double j, double _d_i, double _d_j) {
// CHECK-NEXT:     if (i < 2) {
// CHECK-NEXT:         return {j, _d_j};
// CHECK-NEXT:     }
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t0 = fn5_pushforward(0, i, 0, _d_i);
// CHECK-NEXT:     return {_t0.value, _t0.pushforward};
// CHECK-NEXT: }

// CHECK: double fn5_darg0(double i, double j) {
// CHECK-NEXT:     double _d_i = 1;
// CHECK-NEXT:     double _d_j = 0;
// CHECK-NEXT:     if (i < 2) {
// CHECK-NEXT:         return _d_j;
// CHECK-NEXT:     }
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t0 = fn5_pushforward(0, i, 0, _d_i);
// CHECK-NEXT:     return _t0.pushforward;
// CHECK-NEXT: }

double fn6(double i, double j, double k) {
  if (i < 0.5)
    return 0;
  return i+j+k + fn6(i-1, j-1, k-1);
}

// CHECK: clad::ValueAndPushforward<double, double> fn6_pushforward(double i, double j, double k, double _d_i, double _d_j, double _d_k) {
// CHECK-NEXT:     if (i < 0.5)
// CHECK-NEXT:         return {0, 0};
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t0 = fn6_pushforward(i - 1, j - 1, k - 1, _d_i - 0, _d_j - 0, _d_k - 0);
// CHECK-NEXT:     return {i + j + k + _t0.value, _d_i + _d_j + _d_k + _t0.pushforward};
// CHECK-NEXT: }

// CHECK: double fn6_darg0(double i, double j, double k) {
// CHECK-NEXT:     double _d_i = 1;
// CHECK-NEXT:     double _d_j = 0;
// CHECK-NEXT:     double _d_k = 0;
// CHECK-NEXT:     if (i < 0.5)
// CHECK-NEXT:         return 0;
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t0 = fn6_pushforward(i - 1, j - 1, k - 1, _d_i - 0, _d_j - 0, _d_k - 0);
// CHECK-NEXT:     return _d_i + _d_j + _d_k + _t0.pushforward;
// CHECK-NEXT: }

double helperFn(double&& i, double&& j) {
  return i+j;
}

// CHECK: clad::ValueAndPushforward<double, double> helperFn_pushforward(double &&i, double &&j, double &&_d_i, double &&_d_j) {
// CHECK-NEXT:     return {i + j, _d_i + _d_j};
// CHECK-NEXT: }

double fn7(double i, double j) {
  return helperFn(helperFn(7*i, 9*j), i+j);
}

// CHECK: double fn7_darg0(double i, double j) {
// CHECK-NEXT:     double _d_i = 1;
// CHECK-NEXT:     double _d_j = 0;
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t0 = helperFn_pushforward(7 * i, 9 * j, 0 * i + 7 * _d_i, 0 * j + 9 * _d_j);
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t1 = helperFn_pushforward(static_cast<double &&>(_t0.value), i + j, static_cast<double &&>(_t0.pushforward), _d_i + _d_j);
// CHECK-NEXT:     return _t1.pushforward;
// CHECK-NEXT: }

void modifyArr(double* arr, int n, double val) {
  for (int i=0; i<n; ++i)
    arr[i] = val;
}

// CHECK: void modifyArr_pushforward(double *arr, int n, double val, double *_d_arr, int _d_n, double _d_val) {
// CHECK-NEXT:     {
// CHECK-NEXT:         int _d_i = 0;
// CHECK-NEXT:         for (int i = 0; i < n; ++i) {
// CHECK-NEXT:             _d_arr[i] = _d_val;
// CHECK-NEXT:             arr[i] = val;
// CHECK-NEXT:         }
// CHECK-NEXT:     }
// CHECK-NEXT: }

double sum(double* arr, int n) {
  double val = 0;
  for (int i=0; i<n; ++i)
    val += arr[i];
  return val;
}

// CHECK: clad::ValueAndPushforward<double, double> sum_pushforward(double *arr, int n, double *_d_arr, int _d_n) {
// CHECK-NEXT:     double _d_val = 0;
// CHECK-NEXT:     double val = 0;
// CHECK-NEXT:     {
// CHECK-NEXT:         int _d_i = 0;
// CHECK-NEXT:         for (int i = 0; i < n; ++i) {
// CHECK-NEXT:             _d_val += _d_arr[i];
// CHECK-NEXT:             val += arr[i];
// CHECK-NEXT:         }
// CHECK-NEXT:     }
// CHECK-NEXT:     return {val, _d_val};
// CHECK-NEXT: }

double fn8(double i, double j) {
  double arr[5] = {};
  modifyArr(arr, 5, i*j);
  return sum(arr, 5);
}

// CHECK: double fn8_darg0(double i, double j) {
// CHECK-NEXT:     double _d_i = 1;
// CHECK-NEXT:     double _d_j = 0;
// CHECK-NEXT:     double _d_arr[5] = {};
// CHECK-NEXT:     double arr[5] = {};
// CHECK-NEXT:     modifyArr_pushforward(arr, 5, i * j, _d_arr, 0, _d_i * j + i * _d_j);
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t0 = sum_pushforward(arr, 5, _d_arr, 0);
// CHECK-NEXT:     return _t0.pushforward;
// CHECK-NEXT: }

float test_1_darg0(float x);
float test_2_darg0(float x);
float test_4_darg0(float x);

#define INIT(fn, ...)\
  auto d_##fn = clad::differentiate(fn, __VA_ARGS__);

#define TEST(fn, ...)\
  printf("{%.2f}\n", d_##fn.execute(__VA_ARGS__));

int main () {
  clad::differentiate(test_1, 0);
  printf("Result is = %f\n", test_1_darg0(1.1)); // CHECK-EXEC: Result is = 2

  clad::differentiate(test_2, 0);
  printf("Result is = %f\n", test_2_darg0(1.0)); // CHECK-EXEC: Result is = 2

  clad::differentiate(test_4, 0);
  printf("Result is = %f\n", test_4_darg0(1.0)); // CHECK-EXEC: Result is = 4

  INIT(fn1, "i");
  INIT(fn2, "i");
  INIT(fn3, "i");
  INIT(fn4, "i");
  INIT(fn5, "i");
  INIT(fn6, "i");
  INIT(fn7, "i");
  INIT(fn8, "i");

  TEST(fn1, 3, 5);    // CHECK-EXEC: {12.00}
  TEST(fn2, 3, 5);    // CHECK-EXEC: {181.00}
  TEST(fn3, 3, 5);    // CHECK-EXEC: {6.00}
  TEST(fn4, 3, 5);    // CHECK-EXEC: {1.00}
  TEST(fn5, 3, 5);    // CHECK-EXEC: {1.00}
  TEST(fn6, 3, 5, 7); // CHECK-EXEC: {3.00}
  TEST(fn7, 3, 5);    // CHECK-EXEC: {8.00}
  TEST(fn8, 3, 5);    // CHECK-EXEC: {25.00}
  return 0;
}
