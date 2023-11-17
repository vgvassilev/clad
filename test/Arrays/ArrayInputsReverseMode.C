// RUN: %cladclang %s -I%S/../../include -Wno-unused-value -oArrayInputsReverseMode.out 2>&1 | FileCheck %s
// RUN: ./ArrayInputsReverseMode.out | FileCheck -check-prefix=CHECK-EXEC %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -enable-tbr %s -I%S/../../include -Wno-unused-value -oArrayInputsReverseMode.out
// RUN: ./ArrayInputsReverseMode.out | FileCheck -check-prefix=CHECK-EXEC %s

//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

double addArr(const double *arr, int n) {
  double ret = 0;
  for (int i = 0; i < n; i++) {
    ret += arr[i];
  }
  return ret;
}

//CHECK: void addArr_pullback(const double *arr, int n, double _d_y, clad::array_ref<double> _d_arr, clad::array_ref<int> _d_n) {
//CHECK-NEXT:     double _d_ret = 0;
//CHECK-NEXT:     unsigned long _t0;
//CHECK-NEXT:     int _d_i = 0;
//CHECK-NEXT:     clad::tape<double> _t1 = {};
//CHECK-NEXT:     double ret = 0;
//CHECK-NEXT:     _t0 = 0;
//CHECK-NEXT:     for (int i = 0; i < n; i++) {
//CHECK-NEXT:         _t0++;
//CHECK-NEXT:         clad::push(_t1, ret);
//CHECK-NEXT:         ret += arr[i];
//CHECK-NEXT:     }
//CHECK-NEXT:     goto _label0;
//CHECK-NEXT:   _label0:
//CHECK-NEXT:     _d_ret += _d_y;
//CHECK-NEXT:     for (; _t0; _t0--) {
//CHECK-NEXT:         i--;
//CHECK-NEXT:         {
//CHECK-NEXT:             ret = clad::pop(_t1);
//CHECK-NEXT:             double _r_d0 = _d_ret;
//CHECK-NEXT:             _d_ret += _r_d0;
//CHECK-NEXT:             _d_arr[i] += _r_d0;
//CHECK-NEXT:             _d_ret -= _r_d0;
//CHECK-NEXT:         }
//CHECK-NEXT:     }
//CHECK-NEXT: }

double f(double *arr) {
  return addArr(arr, 3);
}

//CHECK:   void f_grad(double *arr, clad::array_ref<double> _d_arr) {
//CHECK-NEXT:       double *_t0;
//CHECK-NEXT:       _t0 = arr;
//CHECK-NEXT:       goto _label0;
//CHECK-NEXT:     _label0:
//CHECK-NEXT:     {
//CHECK-NEXT:         arr = _t0;
//CHECK-NEXT:         int _grad1 = 0;
//CHECK-NEXT:         addArr_pullback(_t0, 3, 1, _d_arr, &_grad1);
//CHECK-NEXT:         clad::array<double> _r0(_d_arr);
//CHECK-NEXT:         int _r1 = _grad1;
//CHECK-NEXT:     }
//CHECK-NEXT:   }

float func(float* a, float* b) {
  float sum = 0;
  for (int i = 0; i < 3; i++) {
    a[i] *= b[i];
    sum += a[i];
  }
  return sum;
}

//CHECK: void func_grad(float *a, float *b, clad::array_ref<float> _d_a, clad::array_ref<float> _d_b) {
//CHECK-NEXT:     float _d_sum = 0;
//CHECK-NEXT:     unsigned long _t0;
//CHECK-NEXT:     int _d_i = 0;
//CHECK-NEXT:     clad::tape<float> _t1 = {};
//CHECK-NEXT:     clad::tape<float> _t2 = {};
//CHECK-NEXT:     clad::tape<float> _t3 = {};
//CHECK-NEXT:     float sum = 0;
//CHECK-NEXT:     _t0 = 0;
//CHECK-NEXT:     for (int i = 0; i < 3; i++) {
//CHECK-NEXT:         _t0++;
//CHECK-NEXT:         clad::push(_t1, a[i]);
//CHECK-NEXT:         a[i] *= clad::push(_t2, b[i]);
//CHECK-NEXT:         clad::push(_t3, sum);
//CHECK-NEXT:         sum += a[i];
//CHECK-NEXT:     }
//CHECK-NEXT:     goto _label0;
//CHECK-NEXT:   _label0:
//CHECK-NEXT:     _d_sum += 1;
//CHECK-NEXT:     for (; _t0; _t0--) {
//CHECK-NEXT:         i--;
//CHECK-NEXT:         {
//CHECK-NEXT:             sum = clad::pop(_t3);
//CHECK-NEXT:             float _r_d1 = _d_sum;
//CHECK-NEXT:             _d_sum += _r_d1;
//CHECK-NEXT:             _d_a[i] += _r_d1;
//CHECK-NEXT:             _d_sum -= _r_d1;
//CHECK-NEXT:         }
//CHECK-NEXT:         {
//CHECK-NEXT:             a[i] = clad::pop(_t1);
//CHECK-NEXT:             float _r_d0 = _d_a[i];
//CHECK-NEXT:             _d_a[i] += _r_d0 * clad::pop(_t2);
//CHECK-NEXT:             float _r0 = a[i] * _r_d0;
//CHECK-NEXT:             _d_b[i] += _r0;
//CHECK-NEXT:             _d_a[i] -= _r_d0;
//CHECK-NEXT:             _d_a[i];
//CHECK-NEXT:         }
//CHECK-NEXT:     }
//CHECK-NEXT: }

float helper(float x) {
  return 2 * x;
}

// CHECK: void helper_pullback(float x, float _d_y, clad::array_ref<float> _d_x) {
// CHECK-NEXT:     float _t0;
// CHECK-NEXT:     _t0 = x;
// CHECK-NEXT:     goto _label0;
// CHECK-NEXT:   _label0:
// CHECK-NEXT:     {
// CHECK-NEXT:         float _r0 = _d_y * _t0;
// CHECK-NEXT:         float _r1 = 2 * _d_y;
// CHECK-NEXT:         * _d_x += _r1;
// CHECK-NEXT:     }
// CHECK-NEXT: }

float func2(float* a) {
  float sum = 0;
  for (int i = 0; i < 3; i++)
    sum += helper(a[i]);
  return sum;
}

//CHECK: void func2_grad(float *a, clad::array_ref<float> _d_a) {
//CHECK-NEXT:     float _d_sum = 0;
//CHECK-NEXT:     unsigned long _t0;
//CHECK-NEXT:     int _d_i = 0;
//CHECK-NEXT:     clad::tape<float> _t1 = {};
//CHECK-NEXT:     float sum = 0;
//CHECK-NEXT:     _t0 = 0;
//CHECK-NEXT:     for (int i = 0; i < 3; i++) {
//CHECK-NEXT:         _t0++;
//CHECK-NEXT:         clad::push(_t1, sum);
//CHECK-NEXT:         sum += helper(a[i]);
//CHECK-NEXT:     }
//CHECK-NEXT:     goto _label0;
//CHECK-NEXT:   _label0:
//CHECK-NEXT:     _d_sum += 1;
//CHECK-NEXT:     for (; _t0; _t0--) {
//CHECK-NEXT:         i--;
//CHECK-NEXT:         sum = clad::pop(_t1);
//CHECK-NEXT:         float _r_d0 = _d_sum;
//CHECK-NEXT:         _d_sum += _r_d0;
//CHECK-NEXT:         float _grad0 = 0.F;
//CHECK-NEXT:         helper_pullback(a[i], _r_d0, &_grad0);
//CHECK-NEXT:         float _r0 = _grad0;
//CHECK-NEXT:         _d_a[i] += _r0;
//CHECK-NEXT:         _d_sum -= _r_d0;
//CHECK-NEXT:     }
//CHECK-NEXT: }

float func3(float* a, float* b) {
  float sum = 0;
  for (int i = 0; i < 3; i++)
    sum += (a[i] += b[i]);
  return sum;
}

//CHECK: void func3_grad(float *a, float *b, clad::array_ref<float> _d_a, clad::array_ref<float> _d_b) {
//CHECK-NEXT:     float _d_sum = 0;
//CHECK-NEXT:     unsigned long _t0;
//CHECK-NEXT:     int _d_i = 0;
//CHECK-NEXT:     clad::tape<float> _t1 = {};
//CHECK-NEXT:     clad::tape<float> _t2 = {};
//CHECK-NEXT:     float sum = 0;
//CHECK-NEXT:     _t0 = 0;
//CHECK-NEXT:     for (int i = 0; i < 3; i++) {
//CHECK-NEXT:         _t0++;
//CHECK-NEXT:         clad::push(_t1, sum);
//CHECK-NEXT:         clad::push(_t2, a[i]);
//CHECK-NEXT:         sum += (a[i] += b[i]);
//CHECK-NEXT:     }
//CHECK-NEXT:     goto _label0;
//CHECK-NEXT:   _label0:
//CHECK-NEXT:     _d_sum += 1;
//CHECK-NEXT:     for (; _t0; _t0--) {
//CHECK-NEXT:         i--;
//CHECK-NEXT:         sum = clad::pop(_t1);
//CHECK-NEXT:         float _r_d0 = _d_sum;
//CHECK-NEXT:         _d_sum += _r_d0;
//CHECK-NEXT:         _d_a[i] += _r_d0;
//CHECK-NEXT:         a[i] = clad::pop(_t2);
//CHECK-NEXT:         float _r_d1 = _d_a[i];
//CHECK-NEXT:         _d_a[i] += _r_d1;
//CHECK-NEXT:         _d_b[i] += _r_d1;
//CHECK-NEXT:         _d_a[i] -= _r_d1;
//CHECK-NEXT:         _d_sum -= _r_d0;
//CHECK-NEXT:     }
//CHECK-NEXT: }

double func4(double x) {
  double arr[3] = {x, 2 * x, x * x};
  double sum = 0;
  for (int i = 0; i < 3; i++) {
    sum += addArr(arr, 3);
  }
  return sum;
}

//CHECK: void func4_grad(double x, clad::array_ref<double> _d_x) {
//CHECK-NEXT:     double _t0;
//CHECK-NEXT:     double _t1;
//CHECK-NEXT:     clad::array<double> _d_arr(3UL);
//CHECK-NEXT:     double _d_sum = 0;
//CHECK-NEXT:     unsigned long _t2;
//CHECK-NEXT:     int _d_i = 0;
//CHECK-NEXT:     clad::tape<double> _t3 = {};
//CHECK-NEXT:     _t0 = x;
//CHECK-NEXT:     _t1 = x;
//CHECK-NEXT:     double arr[3] = {x, 2 * _t0, x * _t1};
//CHECK-NEXT:     double sum = 0;
//CHECK-NEXT:     _t2 = 0;
//CHECK-NEXT:     for (int i = 0; i < 3; i++) {
//CHECK-NEXT:         _t2++;
//CHECK-NEXT:         clad::push(_t3, sum);
//CHECK-NEXT:         sum += addArr(arr, 3);
//CHECK-NEXT:     }
//CHECK-NEXT:     goto _label0;
//CHECK-NEXT:   _label0:
//CHECK-NEXT:     _d_sum += 1;
//CHECK-NEXT:     for (; _t2; _t2--) {
//CHECK-NEXT:         i--;
//CHECK-NEXT:         {
//CHECK-NEXT:             sum = clad::pop(_t3);
//CHECK-NEXT:             double _r_d0 = _d_sum;
//CHECK-NEXT:             _d_sum += _r_d0;
//CHECK-NEXT:             int _grad1 = 0;
//CHECK-NEXT:             addArr_pullback(arr, 3, _r_d0, _d_arr, &_grad1);
//CHECK-NEXT:             clad::array<double> _r4(_d_arr);
//CHECK-NEXT:             int _r5 = _grad1;
//CHECK-NEXT:             _d_sum -= _r_d0;
//CHECK-NEXT:         }
//CHECK-NEXT:     }
//CHECK-NEXT:     {
//CHECK-NEXT:         * _d_x += _d_arr[0];
//CHECK-NEXT:         double _r0 = _d_arr[1] * _t0;
//CHECK-NEXT:         double _r1 = 2 * _d_arr[1];
//CHECK-NEXT:         * _d_x += _r1;
//CHECK-NEXT:         double _r2 = _d_arr[2] * _t1;
//CHECK-NEXT:         * _d_x += _r2;
//CHECK-NEXT:         double _r3 = x * _d_arr[2];
//CHECK-NEXT:         * _d_x += _r3;
//CHECK-NEXT:     }
//CHECK-NEXT: }

double func5(int k) {
  int n = k;
  double arr[n];
  for (int i = 0; i < n; i++) {
    arr[i] = k;
  }
  double sum = 0;
  for (int i = 0; i < 3; i++) {
    sum += addArr(arr, n);
  }
  return sum;
}

//CHECK: void func5_grad(int k, clad::array_ref<int> _d_k) {
//CHECK-NEXT:     int _d_n = 0;
//CHECK-NEXT:     unsigned long _t0;
//CHECK-NEXT:     int _d_i = 0;
//CHECK-NEXT:     clad::tape<double> _t1 = {};
//CHECK-NEXT:     double _d_sum = 0;
//CHECK-NEXT:     unsigned long _t2;
//CHECK-NEXT:     int _d_i = 0;
//CHECK-NEXT:     clad::tape<double> _t3 = {};
//CHECK-NEXT:     int n = k;
//CHECK-NEXT:     clad::array<double> _d_arr(n);
//CHECK-NEXT:     double arr[n];
//CHECK-NEXT:     _t0 = 0;
//CHECK-NEXT:     for (int i = 0; i < n; i++) {
//CHECK-NEXT:         _t0++;
//CHECK-NEXT:         clad::push(_t1, arr[i]);
//CHECK-NEXT:         arr[i] = k;
//CHECK-NEXT:     }
//CHECK-NEXT:     double sum = 0;
//CHECK-NEXT:     _t2 = 0;
//CHECK-NEXT:     for (int i = 0; i < 3; i++) {
//CHECK-NEXT:         _t2++;
//CHECK-NEXT:         clad::push(_t3, sum);
//CHECK-NEXT:         sum += addArr(arr, n);
//CHECK-NEXT:     }
//CHECK-NEXT:     goto _label0;
//CHECK-NEXT:   _label0:
//CHECK-NEXT:     _d_sum += 1;
//CHECK-NEXT:     for (; _t2; _t2--) {
//CHECK-NEXT:         i--;
//CHECK-NEXT:         {
//CHECK-NEXT:             sum = clad::pop(_t3);
//CHECK-NEXT:             double _r_d1 = _d_sum;
//CHECK-NEXT:             _d_sum += _r_d1;
//CHECK-NEXT:             int _grad1 = 0;
//CHECK-NEXT:             addArr_pullback(arr, n, _r_d1, _d_arr, &_grad1);
//CHECK-NEXT:             clad::array<double> _r0(_d_arr);
//CHECK-NEXT:             int _r1 = _grad1;
//CHECK-NEXT:             _d_n += _r1;
//CHECK-NEXT:             _d_sum -= _r_d1;
//CHECK-NEXT:         }
//CHECK-NEXT:     }
//CHECK-NEXT:     for (; _t0; _t0--) {
//CHECK-NEXT:         i--;
//CHECK-NEXT:         {
//CHECK-NEXT:             arr[i] = clad::pop(_t1);
//CHECK-NEXT:             double _r_d0 = _d_arr[i];
//CHECK-NEXT:             * _d_k += _r_d0;
//CHECK-NEXT:             _d_arr[i] -= _r_d0;
//CHECK-NEXT:             _d_arr[i];
//CHECK-NEXT:         }
//CHECK-NEXT:     }
//CHECK-NEXT:     * _d_k += _d_n;
//CHECK-NEXT: }

double func6(double seed) {
  double sum = 0;
  for (int i = 0; i < 3; i++) {
    double arr[3] = {seed, seed * i, seed + i};
    sum += addArr(arr, 3);
  }
  return sum;
}

//CHECK: void func6_grad(double seed, clad::array_ref<double> _d_seed) {
//CHECK-NEXT:     double _d_sum = 0;
//CHECK-NEXT:     unsigned long _t0;
//CHECK-NEXT:     int _d_i = 0;
//CHECK-NEXT:     clad::tape<int> _t1 = {};
//CHECK-NEXT:     clad::array<double> _d_arr(3UL);
//CHECK-NEXT:     clad::tape<double> _t2 = {};
//CHECK-NEXT:     double sum = 0;
//CHECK-NEXT:     _t0 = 0;
//CHECK-NEXT:     for (int i = 0; i < 3; i++) {
//CHECK-NEXT:         _t0++;
//CHECK-NEXT:         double arr[3] = {seed, seed * clad::push(_t1, i), seed + i};
//CHECK-NEXT:         clad::push(_t2, sum);
//CHECK-NEXT:         sum += addArr(arr, 3);
//CHECK-NEXT:     }
//CHECK-NEXT:     goto _label0;
//CHECK-NEXT:   _label0:
//CHECK-NEXT:     _d_sum += 1;
//CHECK-NEXT:     for (; _t0; _t0--) {
//CHECK-NEXT:         i--;
//CHECK-NEXT:         {
//CHECK-NEXT:             sum = clad::pop(_t2);
//CHECK-NEXT:             double _r_d0 = _d_sum;
//CHECK-NEXT:             _d_sum += _r_d0;
//CHECK-NEXT:             int _grad1 = 0;
//CHECK-NEXT:             addArr_pullback(arr, 3, _r_d0, _d_arr, &_grad1);
//CHECK-NEXT:             clad::array<double> _r2(_d_arr);
//CHECK-NEXT:             int _r3 = _grad1;
//CHECK-NEXT:             _d_sum -= _r_d0;
//CHECK-NEXT:         }
//CHECK-NEXT:         {
//CHECK-NEXT:             * _d_seed += _d_arr[0];
//CHECK-NEXT:             double _r0 = _d_arr[1] * clad::pop(_t1);
//CHECK-NEXT:             * _d_seed += _r0;
//CHECK-NEXT:             double _r1 = seed * _d_arr[1];
//CHECK-NEXT:             _d_i += _r1;
//CHECK-NEXT:             * _d_seed += _d_arr[2];
//CHECK-NEXT:             _d_i += _d_arr[2];
//CHECK-NEXT:             _d_arr = {};
//CHECK-NEXT:         }
//CHECK-NEXT:     }
//CHECK-NEXT: }

double inv_square(double *params) {
  return 1 / (params[0] * params[0]);
}

//CHECK: void inv_square_pullback(double *params, double _d_y, clad::array_ref<double> _d_params) {
//CHECK-NEXT:     double _t0;
//CHECK-NEXT:     double _t1;
//CHECK-NEXT:     _t1 = params[0];
//CHECK-NEXT:     _t0 = (params[0] * _t1);
//CHECK-NEXT:     goto _label0;
//CHECK-NEXT:   _label0:
//CHECK-NEXT:     {
//CHECK-NEXT:         double _r0 = _d_y / _t0;
//CHECK-NEXT:         double _r1 = _d_y * -1 / (_t0 * _t0);
//CHECK-NEXT:         double _r2 = _r1 * _t1;
//CHECK-NEXT:         _d_params[0] += _r2;
//CHECK-NEXT:         double _r3 = params[0] * _r1;
//CHECK-NEXT:         _d_params[0] += _r3;
//CHECK-NEXT:     }
//CHECK-NEXT: }

double func7(double *params) {
  double out = 0.0;
  for (std::size_t i = 0; i < 1; ++i) {
    double paramsPrime[1] = {params[0]};
    out = out + inv_square(paramsPrime);
  }
  return out;
}

//CHECK: void func7_grad(double *params, clad::array_ref<double> _d_params) {
//CHECK-NEXT:     double _d_out = 0;
//CHECK-NEXT:     unsigned long _t0;
//CHECK-NEXT:     std::size_t _d_i = 0;
//CHECK-NEXT:     clad::array<double> _d_paramsPrime(1UL);
//CHECK-NEXT:     clad::tape<double> _t1 = {};
//CHECK-NEXT:     double out = 0.;
//CHECK-NEXT:     _t0 = 0;
//CHECK-NEXT:     for (std::size_t i = 0; i < 1; ++i) {
//CHECK-NEXT:         _t0++;
//CHECK-NEXT:         double paramsPrime[1] = {params[0]};
//CHECK-NEXT:         clad::push(_t1, out);
//CHECK-NEXT:         out = out + inv_square(paramsPrime);
//CHECK-NEXT:     }
//CHECK-NEXT:     goto _label0;
//CHECK-NEXT:   _label0:
//CHECK-NEXT:     _d_out += 1;
//CHECK-NEXT:     for (; _t0; _t0--) {
//CHECK-NEXT:         --i;
//CHECK-NEXT:         {
//CHECK-NEXT:             out = clad::pop(_t1);
//CHECK-NEXT:             double _r_d0 = _d_out;
//CHECK-NEXT:             _d_out += _r_d0;
//CHECK-NEXT:             inv_square_pullback(paramsPrime, _r_d0, _d_paramsPrime);
//CHECK-NEXT:             clad::array<double> _r0(_d_paramsPrime);
//CHECK-NEXT:             _d_out -= _r_d0;
//CHECK-NEXT:         }
//CHECK-NEXT:         {
//CHECK-NEXT:             _d_params[0] += _d_paramsPrime[0];
//CHECK-NEXT:             _d_paramsPrime = {};
//CHECK-NEXT:         }
//CHECK-NEXT:     }
//CHECK-NEXT: }

int main() {
  double arr[] = {1, 2, 3};
  auto f_dx = clad::gradient(f);

  double darr[3] = {};
  clad::array_ref<double> darr_ref(darr, 3);
  f_dx.execute(arr, darr_ref);

  printf("Result = {%.2f, %.2f, %.2f}\n", darr[0], darr[1], darr[2]); // CHECK-EXEC: Result = {1.00, 1.00, 1.00}

  float a1[3] = {1, 1, 1}, a2[3] = {2, 3, 4}, a3[3] = {1, 1, 1}, b[3] = {2, 5, 2};
  float dpa1[3] = {0}, dpa2[3] = {0}, dpa3[3] = {0}, dpb1[3] = {0}, dpb2[3] = {0};
  clad::array_ref<float> da1(dpa1, 3), da2(dpa2, 3), da3(dpa3, 3), db1(dpb1, 3), db2(dpb2, 3);

  auto lhs = clad::gradient(func);
  lhs.execute(a1, b, da1, db1);
  printf("Result (a) = {%.2f, %.2f, %.2f}\n", da1[0], da1[1], da1[2]); // CHECK-EXEC: Result (a) = {2.00, 5.00, 2.00}

  auto funcArr = clad::gradient(func2);
  funcArr.execute(a2, da2);
  printf("Result = {%.2f, %.2f, %.2f}\n", da2[0], da2[1], da2[2]); // CHECK-EXEC: Result = {2.00, 2.00, 2.00}

  auto nested = clad::gradient(func3);
  nested.execute(a3, b, da3, db2);
  printf("Result (b) = {%.2f, %.2f, %.2f}\n", db2[0], db2[1], db2[2]); // CHECK-EXEC: Result (b) = {1.00, 1.00, 1.00}

  auto constArray = clad::gradient(func4);
  double _dx = 0;
  constArray.execute(1, &_dx);
  printf("Result = {%.2f}\n", _dx); // CHECK-EXEC: Result = {15.00}

  auto df = clad::gradient(func5);
  int dk = 0;
  // Should evaluate to k*3
  df.execute(10, &dk);
  printf("Result = {%.2d}\n", dk); // CHECK-EXEC: Result = {30}

  auto localArray = clad::gradient(func6);
  double dseed = 0;
  localArray.execute(1, &dseed);
  printf("Result = {%.2f}\n", dseed); // CHECK-EXEC: Result = {9.00}

  auto func7grad = clad::gradient(func7);
  double params = 2.0;
  double dparams = 0.0;
  func7grad.execute(&params, &dparams);
  printf("Result = {%.2f}\n", dparams); // CHECK-EXEC: Result = {-0.25}
}
