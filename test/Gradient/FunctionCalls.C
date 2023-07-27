// RUN: %cladnumdiffclang %s  -I%S/../../include -oFunctionCalls.out 2>&1 | FileCheck %s
// RUN: ./FunctionCalls.out | FileCheck -check-prefix=CHECK-EXEC %s

//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

namespace A {
  template <typename T> T constantFn(T i) { return 3; }
  // CHECK: void constantFn_pullback(float i, float _d_y, clad::array_ref<float> _d_i) {
  // CHECK-NEXT:     ;
  // CHECK-NEXT: }
} // namespace A

double constantFn(double i) {
  return 5;
}

double fn1(float i) {
  float res = A::constantFn(i);
  double a = res*i;
  return a;
}

// CHECK: void fn1_grad(float i, clad::array_ref<float> _d_i) {
// CHECK-NEXT:     float _t0;
// CHECK-NEXT:     float _d_res = 0;
// CHECK-NEXT:     float _t1;
// CHECK-NEXT:     float _t2;
// CHECK-NEXT:     double _d_a = 0;
// CHECK-NEXT:     _t0 = i;
// CHECK-NEXT:     float res = A::constantFn(_t0);
// CHECK-NEXT:     _t2 = res;
// CHECK-NEXT:     _t1 = i;
// CHECK-NEXT:     double a = _t2 * _t1;
// CHECK-NEXT:     _d_a += 1;
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r1 = _d_a * _t1;
// CHECK-NEXT:         _d_res += _r1;
// CHECK-NEXT:         double _r2 = _t2 * _d_a;
// CHECK-NEXT:         * _d_i += _r2;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         float _grad0 = 0.F;
// CHECK-NEXT:         constantFn_pullback(_t0, _d_res, &_grad0);
// CHECK-NEXT:         float _r0 = _grad0;
// CHECK-NEXT:         * _d_i += _r0;
// CHECK-NEXT:     }
// CHECK-NEXT: }

double modify1(double& i, double& j) {
  i += j;
  j += j;
  double res = i + j;
  return res;
}

// CHECK: void modify1_pullback(double &i, double &j, double _d_y, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j) {
// CHECK-NEXT:     double _d_res = 0;
// CHECK-NEXT:     i += j;
// CHECK-NEXT:     j += j;
// CHECK-NEXT:     double res = i + j;
// CHECK-NEXT:     _d_res += _d_y;
// CHECK-NEXT:     {
// CHECK-NEXT:         * _d_i += _d_res;
// CHECK-NEXT:         * _d_j += _d_res;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r_d1 = * _d_j;
// CHECK-NEXT:         * _d_j += _r_d1;
// CHECK-NEXT:         * _d_j += _r_d1;
// CHECK-NEXT:         * _d_j -= _r_d1;
// CHECK-NEXT:         * _d_j;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r_d0 = * _d_i;
// CHECK-NEXT:         * _d_i += _r_d0;
// CHECK-NEXT:         * _d_j += _r_d0;
// CHECK-NEXT:         * _d_i -= _r_d0;
// CHECK-NEXT:         * _d_i;
// CHECK-NEXT:     }
// CHECK-NEXT: }

double fn2(double i, double j) {
  double temp = 0;
  temp = modify1(i, j);
  temp = modify1(i, j);
  return i;
}

// CHECK: void fn2_grad(double i, double j, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j) {
// CHECK-NEXT:     double _d_temp = 0;
// CHECK-NEXT:     double _t0;
// CHECK-NEXT:     double _t1;
// CHECK-NEXT:     double _t2;
// CHECK-NEXT:     double _t3;
// CHECK-NEXT:     double temp = 0;
// CHECK-NEXT:     _t0 = i;
// CHECK-NEXT:     _t1 = j;
// CHECK-NEXT:     temp = modify1(i, j);
// CHECK-NEXT:     _t2 = i;
// CHECK-NEXT:     _t3 = j;
// CHECK-NEXT:     temp = modify1(i, j);
// CHECK-NEXT:     * _d_i += 1;
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r_d1 = _d_temp;
// CHECK-NEXT:         modify1_pullback(_t2, _t3, _r_d1, &* _d_i, &* _d_j);
// CHECK-NEXT:         double _r2 = * _d_i;
// CHECK-NEXT:         double _r3 = * _d_j;
// CHECK-NEXT:         _d_temp -= _r_d1;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r_d0 = _d_temp;
// CHECK-NEXT:         modify1_pullback(_t0, _t1, _r_d0, &* _d_i, &* _d_j);
// CHECK-NEXT:         double _r0 = * _d_i;
// CHECK-NEXT:         double _r1 = * _d_j;
// CHECK-NEXT:         _d_temp -= _r_d0;
// CHECK-NEXT:     }
// CHECK-NEXT: }

void update1(double& i, double& j) {
  i += j;
  j += j;
}

// CHECK: void update1_pullback(double &i, double &j, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j) {
// CHECK-NEXT:     i += j;
// CHECK-NEXT:     j += j;
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r_d1 = * _d_j;
// CHECK-NEXT:         * _d_j += _r_d1;
// CHECK-NEXT:         * _d_j += _r_d1;
// CHECK-NEXT:         * _d_j -= _r_d1;
// CHECK-NEXT:         * _d_j;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r_d0 = * _d_i;
// CHECK-NEXT:         * _d_i += _r_d0;
// CHECK-NEXT:         * _d_j += _r_d0;
// CHECK-NEXT:         * _d_i -= _r_d0;
// CHECK-NEXT:         * _d_i;
// CHECK-NEXT:     }
// CHECK-NEXT: }

double fn3(double i, double j) {
  update1(i, j);
  update1(i, j);
  return i;
}

// CHECK: void fn3_grad(double i, double j, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j) {
// CHECK-NEXT:     double _t0;
// CHECK-NEXT:     double _t1;
// CHECK-NEXT:     double _t2;
// CHECK-NEXT:     double _t3;
// CHECK-NEXT:     _t0 = i;
// CHECK-NEXT:     _t1 = j;
// CHECK-NEXT:     update1(i, j);
// CHECK-NEXT:     _t2 = i;
// CHECK-NEXT:     _t3 = j;
// CHECK-NEXT:     update1(i, j);
// CHECK-NEXT:     * _d_i += 1;
// CHECK-NEXT:     {
// CHECK-NEXT:         update1_pullback(_t2, _t3, &* _d_i, &* _d_j);
// CHECK-NEXT:         double _r2 = * _d_i;
// CHECK-NEXT:         double _r3 = * _d_j;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         update1_pullback(_t0, _t1, &* _d_i, &* _d_j);
// CHECK-NEXT:         double _r0 = * _d_i;
// CHECK-NEXT:         double _r1 = * _d_j;
// CHECK-NEXT:     }
// CHECK-NEXT: }

float sum(double* arr, int n) {
  float res = 0;
  for (int i=0; i<n; ++i)
    res += arr[i];
  arr[0] += 10*arr[0];
  return res;
}

// CHECK: void sum_pullback(double *arr, int n, float _d_y, clad::array_ref<double> _d_arr, clad::array_ref<int> _d_n) {
// CHECK-NEXT:     float _d_res = 0;
// CHECK-NEXT:     unsigned long _t0;
// CHECK-NEXT:     int _d_i = 0;
// CHECK-NEXT:     clad::tape<int> _t1 = {};
// CHECK-NEXT:     double _t3;
// CHECK-NEXT:     float res = 0;
// CHECK-NEXT:     _t0 = 0;
// CHECK-NEXT:     for (int i = 0; i < n; ++i) {
// CHECK-NEXT:         _t0++;
// CHECK-NEXT:         res += arr[clad::push(_t1, i)];
// CHECK-NEXT:     }
// CHECK-NEXT:     _t3 = arr[0];
// CHECK-NEXT:     arr[0] += 10 * _t3;
// CHECK-NEXT:     _d_res += _d_y;
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r_d1 = _d_arr[0];
// CHECK-NEXT:         _d_arr[0] += _r_d1;
// CHECK-NEXT:         double _r0 = _r_d1 * _t3;
// CHECK-NEXT:         double _r1 = 10 * _r_d1;
// CHECK-NEXT:         _d_arr[0] += _r1;
// CHECK-NEXT:         _d_arr[0] -= _r_d1;
// CHECK-NEXT:         _d_arr[0];
// CHECK-NEXT:     }
// CHECK-NEXT:     for (; _t0; _t0--) {
// CHECK-NEXT:         float _r_d0 = _d_res;
// CHECK-NEXT:         _d_res += _r_d0;
// CHECK-NEXT:         int _t2 = clad::pop(_t1);
// CHECK-NEXT:         _d_arr[_t2] += _r_d0;
// CHECK-NEXT:         _d_res -= _r_d0;
// CHECK-NEXT:     }
// CHECK-NEXT: }

void twice(double& d) {
  d = 2*d;
}

// CHECK: void twice_pullback(double &d, clad::array_ref<double> _d_d) {
// CHECK-NEXT:     double _t0;
// CHECK-NEXT:     _t0 = d;
// CHECK-NEXT:     d = 2 * _t0;
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r_d0 = * _d_d;
// CHECK-NEXT:         double _r0 = _r_d0 * _t0;
// CHECK-NEXT:         double _r1 = 2 * _r_d0;
// CHECK-NEXT:         * _d_d += _r1;
// CHECK-NEXT:         * _d_d -= _r_d0;
// CHECK-NEXT:         * _d_d;
// CHECK-NEXT:     }
// CHECK-NEXT: }

double fn4(double* arr, int n) {
  double res = 0;
  res += sum(arr, n);
  for (int i=0; i<n; ++i) {
    twice(arr[i]);
    res += arr[i];
  }
  return res;
}

// CHECK: void fn4_grad(double *arr, int n, clad::array_ref<double> _d_arr, clad::array_ref<int> _d_n) {
// CHECK-NEXT:     double _d_res = 0;
// CHECK-NEXT:     double *_t0;
// CHECK-NEXT:     int _t1;
// CHECK-NEXT:     unsigned long _t2;
// CHECK-NEXT:     int _d_i = 0;
// CHECK-NEXT:     clad::tape<int> _t3 = {};
// CHECK-NEXT:     clad::tape<double> _t5 = {};
// CHECK-NEXT:     clad::tape<int> _t6 = {};
// CHECK-NEXT:     clad::tape<int> _t8 = {};
// CHECK-NEXT:     double res = 0;
// CHECK-NEXT:     _t0 = arr;
// CHECK-NEXT:     _t1 = n;
// CHECK-NEXT:     res += sum(arr, _t1);
// CHECK-NEXT:     _t2 = 0;
// CHECK-NEXT:     for (int i = 0; i < n; ++i) {
// CHECK-NEXT:         _t2++;
// CHECK-NEXT:         clad::push(_t5, arr[clad::push(_t3, i)]);
// CHECK-NEXT:         twice(arr[clad::push(_t6, i)]);
// CHECK-NEXT:         res += arr[clad::push(_t8, i)];
// CHECK-NEXT:     }
// CHECK-NEXT:     _d_res += 1;
// CHECK-NEXT:     for (; _t2; _t2--) {
// CHECK-NEXT:         {
// CHECK-NEXT:             int _t7 = clad::pop(_t6);
// CHECK-NEXT:             int _t4 = clad::pop(_t3);
// CHECK-NEXT:             double _r_d1 = _d_res;
// CHECK-NEXT:             _d_res += _r_d1;
// CHECK-NEXT:             int _t9 = clad::pop(_t8);
// CHECK-NEXT:             _d_arr[_t9] += _r_d1;
// CHECK-NEXT:             _d_res -= _r_d1;
// CHECK-NEXT:         }
// CHECK-NEXT:         {
// CHECK-NEXT:             double _r3 = clad::pop(_t5);
// CHECK-NEXT:             twice_pullback(_r3, &_d_arr[_t4]);
// CHECK-NEXT:             double _r2 = _d_arr[_t4];
// CHECK-NEXT:         }
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r_d0 = _d_res;
// CHECK-NEXT:         _d_res += _r_d0;
// CHECK-NEXT:         int _grad1 = 0;
// CHECK-NEXT:         sum_pullback(_t0, _t1, _r_d0, _d_arr, &_grad1);
// CHECK-NEXT:         clad::array<double> _r0(_d_arr);
// CHECK-NEXT:         int _r1 = _grad1;
// CHECK-NEXT:         * _d_n += _r1;
// CHECK-NEXT:         _d_res -= _r_d0;
// CHECK-NEXT:     }
// CHECK-NEXT: }

double modify2(double* arr) {
    arr[0] = 5*arr[0] + arr[1];
    return 1;
}

// CHECK: void modify2_pullback(double *arr, double _d_y, clad::array_ref<double> _d_arr) {
// CHECK-NEXT:     double _t0;
// CHECK-NEXT:     _t0 = arr[0];
// CHECK-NEXT:     arr[0] = 5 * _t0 + arr[1];
// CHECK-NEXT:     ;
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r_d0 = _d_arr[0];
// CHECK-NEXT:         double _r0 = _r_d0 * _t0;
// CHECK-NEXT:         double _r1 = 5 * _r_d0;
// CHECK-NEXT:         _d_arr[0] += _r1;
// CHECK-NEXT:         _d_arr[1] += _r_d0;
// CHECK-NEXT:         _d_arr[0] -= _r_d0;
// CHECK-NEXT:         _d_arr[0];
// CHECK-NEXT:     }
// CHECK-NEXT: }

double fn5(double* arr, int n) {
    double temp = modify2(arr);
    return arr[0];
}

// CHECK: void fn5_grad(double *arr, int n, clad::array_ref<double> _d_arr, clad::array_ref<int> _d_n) {
// CHECK-NEXT:     double *_t0;
// CHECK-NEXT:     double _d_temp = 0;
// CHECK-NEXT:     _t0 = arr;
// CHECK-NEXT:     double temp = modify2(arr);
// CHECK-NEXT:     _d_arr[0] += 1;
// CHECK-NEXT:     {
// CHECK-NEXT:         modify2_pullback(_t0, _d_temp, _d_arr);
// CHECK-NEXT:         clad::array<double> _r0(_d_arr);
// CHECK-NEXT:     }
// CHECK-NEXT: }

double fn6(double i=0, double j=0) {
  return i*j;
}

template<typename T>
void reset(T* arr, int n) {
  for (int i=0; i<n; ++i)
    arr[i] = 0;
}

template<typename T>
void print(T* arr, int n) {
  printf("{");
  for (int i=0; i<n; ++i) {
    printf("%.2f", arr[i]);
    if (i != n-1)
      printf(", ");
  }
  printf("}\n");
}

#define INIT(F)\
  auto F##_grad = clad::gradient(F);

#define TEST1(F, ...)\
  result[0] = 0;\
  F##_grad.execute(__VA_ARGS__, &result[0]);\
  printf("{%.2f}\n", result[0]);

#define TEST1_float(F, ...)\
  fresult[0] = 0;\
  F##_grad.execute(__VA_ARGS__, &fresult[0]);\
  printf("{%.2f}\n", fresult[0]);

#define TEST2(F, ...)\
  result[0] = result[1] = 0;\
  F##_grad.execute(__VA_ARGS__, &result[0], &result[1]);\
  printf("{%.2f, %.2f}\n", result[0], result[1]);

#define TEST_ARR5(F, ...)\
  reset(result, 5);\
  d_n = 0;\
  F##_grad.execute(__VA_ARGS__, clad::array_ref<double>(result, 5), &d_n);\
  print(result, 5);

int main() {
  double result[7];
  float fresult[7];
  double d_n;
  INIT(fn1);
  INIT(fn2);
  INIT(fn3);
  INIT(fn4);
  INIT(fn5);
  INIT(fn6);

  TEST1_float(fn1, 11);         // CHECK-EXEC: {3.00}
  TEST2(fn2, 3, 5);             // CHECK-EXEC: {1.00, 3.00}
  TEST2(fn3, 3, 5);             // CHECK-EXEC: {1.00, 3.00}
  double arr[5] = {1, 2, 3, 4, 5};
  TEST_ARR5(fn4, arr, 5);       // CHECK-EXEC: {23.00, 3.00, 3.00, 3.00, 3.00}
  TEST_ARR5(fn5, arr, 5);       // CHECK-EXEC: {5.00, 1.00, 0.00, 0.00, 0.00}
  TEST2(fn6, 3, 5);             // CHECK-EXEC: {5.00, 3.00}
}
