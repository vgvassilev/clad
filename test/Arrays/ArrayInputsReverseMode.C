// RUN: %cladclang %s -I%S/../../include -Wno-unused-value -oArrayInputsReverseMode.out 2>&1 | FileCheck %s
// RUN: ./ArrayInputsReverseMode.out | FileCheck -check-prefix=CHECK-EXEC %s

//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

double addArr(double *arr, int n) {
  double ret = 0;
  for (int i = 0; i < n; i++) {
    ret += arr[i];
  }
  return ret;
}

//CHECK: void addArr_pullback(double *arr, int n, double _d_y, clad::array_ref<double> _d_arr, clad::array_ref<int> _d_n) {
//CHECK-NEXT:     double _d_ret = 0;
//CHECK-NEXT:     unsigned long _t0;
//CHECK-NEXT:     int _d_i = 0;
//CHECK-NEXT:     clad::tape<int> _t1 = {};
//CHECK-NEXT:     double ret = 0;
//CHECK-NEXT:     _t0 = 0;
//CHECK-NEXT:     for (int i = 0; i < n; i++) {
//CHECK-NEXT:         _t0++;
//CHECK-NEXT:         ret += arr[clad::push(_t1, i)];
//CHECK-NEXT:     }
//CHECK-NEXT:     _d_ret += _d_y;
//CHECK-NEXT:     for (; _t0; _t0--) {
//CHECK-NEXT:         {
//CHECK-NEXT:             double _r_d0 = _d_ret;
//CHECK-NEXT:             _d_ret += _r_d0;
//CHECK-NEXT:             int _t2 = clad::pop(_t1);
//CHECK-NEXT:             _d_arr[_t2] += _r_d0;
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
//CHECK-NEXT:     {
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
//CHECK-NEXT:     clad::tape<int> _t1 = {};
//CHECK-NEXT:     clad::tape<float> _t3 = {};
//CHECK-NEXT:     clad::tape<float> _t4 = {};
//CHECK-NEXT:     clad::tape<int> _t5 = {};
//CHECK-NEXT:     clad::tape<int> _t7 = {};
//CHECK-NEXT:     float sum = 0;
//CHECK-NEXT:     _t0 = 0;
//CHECK-NEXT:     for (int i = 0; i < 3; i++) {
//CHECK-NEXT:         _t0++;
//CHECK-NEXT:         float _ref0 = a[clad::push(_t1, i)];
//CHECK-NEXT:         clad::push(_t4, _ref0);
//CHECK-NEXT:         _ref0 *= clad::push(_t3, b[clad::push(_t5, i)]);
//CHECK-NEXT:         sum += a[clad::push(_t7, i)];
//CHECK-NEXT:     }
//CHECK-NEXT:     _d_sum += 1;
//CHECK-NEXT:     for (; _t0; _t0--) {
//CHECK-NEXT:         {
//CHECK-NEXT:             float _r_d1 = _d_sum;
//CHECK-NEXT:             _d_sum += _r_d1;
//CHECK-NEXT:             int _t8 = clad::pop(_t7);
//CHECK-NEXT:             _d_a[_t8] += _r_d1;
//CHECK-NEXT:             _d_sum -= _r_d1;
//CHECK-NEXT:         }
//CHECK-NEXT:         {
//CHECK-NEXT:             int _t2 = clad::pop(_t1);
//CHECK-NEXT:             float _r_d0 = _d_a[_t2];
//CHECK-NEXT:             _d_a[_t2] += _r_d0 * clad::pop(_t3);
//CHECK-NEXT:             float _r0 = clad::pop(_t4) * _r_d0;
//CHECK-NEXT:             int _t6 = clad::pop(_t5);
//CHECK-NEXT:             _d_b[_t6] += _r0;
//CHECK-NEXT:             _d_a[_t2] -= _r_d0;
//CHECK-NEXT:             _d_a[_t2];
//CHECK-NEXT:         }
//CHECK-NEXT:     }
//CHECK-NEXT: }

float helper(float x) {
  return 2 * x;
}

// CHECK: void helper_pullback(float x, float _d_y, clad::array_ref<float> _d_x) {
// CHECK-NEXT:     float _t0;
// CHECK-NEXT:     _t0 = x;
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
//CHECK-NEXT:     clad::tape<int> _t1 = {};
//CHECK-NEXT:     clad::tape<float> _t3 = {};
//CHECK-NEXT:     float sum = 0;
//CHECK-NEXT:     _t0 = 0;
//CHECK-NEXT:     for (int i = 0; i < 3; i++) {
//CHECK-NEXT:         _t0++;
//CHECK-NEXT:         sum += helper(clad::push(_t3, a[clad::push(_t1, i)]));
//CHECK-NEXT:     }
//CHECK-NEXT:     _d_sum += 1;
//CHECK-NEXT:     for (; _t0; _t0--) {
//CHECK-NEXT:         float _r_d0 = _d_sum;
//CHECK-NEXT:         _d_sum += _r_d0;
//CHECK-NEXT:         float _grad0 = 0.F;
//CHECK-NEXT:         helper_pullback(clad::pop(_t3), _r_d0, &_grad0);
//CHECK-NEXT:         float _r0 = _grad0;
//CHECK-NEXT:         int _t2 = clad::pop(_t1);
//CHECK-NEXT:         _d_a[_t2] += _r0;
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
//CHECK-NEXT:     clad::tape<int> _t1 = {};
//CHECK-NEXT:     clad::tape<int> _t3 = {};
//CHECK-NEXT:     float sum = 0;
//CHECK-NEXT:     _t0 = 0;
//CHECK-NEXT:     for (int i = 0; i < 3; i++) {
//CHECK-NEXT:         _t0++;
//CHECK-NEXT:         sum += (a[clad::push(_t1, i)] += b[clad::push(_t3, i)]);
//CHECK-NEXT:     }
//CHECK-NEXT:     _d_sum += 1;
//CHECK-NEXT:     for (; _t0; _t0--) {
//CHECK-NEXT:         float _r_d0 = _d_sum;
//CHECK-NEXT:         _d_sum += _r_d0;
//CHECK-NEXT:         int _t2 = clad::pop(_t1);
//CHECK-NEXT:         float _r_d1 = _d_a[_t2];
//CHECK-NEXT:         _d_a[_t2] += _r_d1;
//CHECK-NEXT:         int _t4 = clad::pop(_t3);
//CHECK-NEXT:         _d_b[_t4] += _r_d1;
//CHECK-NEXT:         _d_a[_t2] -= _r_d1;
//CHECK-NEXT:         _d_a[_t2] += _r_d0;
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
//CHECK-NEXT:     double _t2;
//CHECK-NEXT:     clad::array<double> _d_arr(3UL);
//CHECK-NEXT:     double _d_sum = 0;
//CHECK-NEXT:     unsigned long _t3;
//CHECK-NEXT:     int _d_i = 0;
//CHECK-NEXT:     clad::tape<clad::array<double> > _t4 = {};
//CHECK-NEXT:     _t0 = x;
//CHECK-NEXT:     _t2 = x;
//CHECK-NEXT:     _t1 = x;
//CHECK-NEXT:     double arr[3] = {x, 2 * _t0, _t2 * _t1};
//CHECK-NEXT:     double sum = 0;
//CHECK-NEXT:     _t3 = 0;
//CHECK-NEXT:     for (int i = 0; i < 3; i++) {
//CHECK-NEXT:         _t3++;
//CHECK-NEXT:         clad::push(_t4, arr , 3UL);
//CHECK-NEXT:         sum += addArr(arr, 3);
//CHECK-NEXT:     }
//CHECK-NEXT:     _d_sum += 1;
//CHECK-NEXT:     for (; _t3; _t3--) {
//CHECK-NEXT:         {
//CHECK-NEXT:             double _r_d0 = _d_sum;
//CHECK-NEXT:             _d_sum += _r_d0;
//CHECK-NEXT:             clad::array<double> _r5 = clad::pop(_t4);
//CHECK-NEXT:             int _grad1 = 0;
//CHECK-NEXT:             addArr_pullback(_r5, 3, _r_d0, _d_arr, &_grad1);
//CHECK-NEXT:             clad::array<double> _r4(_d_arr);
//CHECK-NEXT:             int _r6 = _grad1;
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
//CHECK-NEXT:         double _r3 = _t2 * _d_arr[2];
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
//CHECK-NEXT:     clad::tape<int> _t1 = {};
//CHECK-NEXT:     double _d_sum = 0;
//CHECK-NEXT:     unsigned long _t3;
//CHECK-NEXT:     int _d_i = 0;
//CHECK-NEXT:     clad::tape<clad::array<double> > _t4 = {};
//CHECK-NEXT:     clad::tape<int> _t5 = {};
//CHECK-NEXT:     int n = k;
//CHECK-NEXT:     clad::array<double> _d_arr(n);
//CHECK-NEXT:     double arr[n];
//CHECK-NEXT:     _t0 = 0;
//CHECK-NEXT:     for (int i = 0; i < n; i++) {
//CHECK-NEXT:         _t0++;
//CHECK-NEXT:         arr[clad::push(_t1, i)] = k;
//CHECK-NEXT:     }
//CHECK-NEXT:     double sum = 0;
//CHECK-NEXT:     _t3 = 0;
//CHECK-NEXT:     for (int i = 0; i < 3; i++) {
//CHECK-NEXT:         _t3++;
//CHECK-NEXT:         clad::push(_t4, arr , n);
//CHECK-NEXT:         sum += addArr(arr, clad::push(_t5, n));
//CHECK-NEXT:     }
//CHECK-NEXT:     _d_sum += 1;
//CHECK-NEXT:     for (; _t3; _t3--) {
//CHECK-NEXT:         {
//CHECK-NEXT:             double _r_d1 = _d_sum;
//CHECK-NEXT:             _d_sum += _r_d1;
//CHECK-NEXT:             clad::array<double> _r1 = clad::pop(_t4);
//CHECK-NEXT:             int _grad1 = 0;
//CHECK-NEXT:             addArr_pullback(_r1, clad::pop(_t5), _r_d1, _d_arr, &_grad1);
//CHECK-NEXT:             clad::array<double> _r0(_d_arr);
//CHECK-NEXT:             int _r2 = _grad1;
//CHECK-NEXT:             _d_n += _r2;
//CHECK-NEXT:             _d_sum -= _r_d1;
//CHECK-NEXT:         }
//CHECK-NEXT:     }
//CHECK-NEXT:     for (; _t0; _t0--) {
//CHECK-NEXT:         {
//CHECK-NEXT:             int _t2 = clad::pop(_t1);
//CHECK-NEXT:             double _r_d0 = _d_arr[_t2];
//CHECK-NEXT:             * _d_k += _r_d0;
//CHECK-NEXT:             _d_arr[_t2] -= _r_d0;
//CHECK-NEXT:             _d_arr[_t2];
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
//CHECK-NEXT:     clad::tape<double> _t2 = {};
//CHECK-NEXT:     clad::array<double> _d_arr(3UL);
//CHECK-NEXT:     clad::tape<clad::array<double> > _t3 = {};
//CHECK-NEXT:     double sum = 0;
//CHECK-NEXT:     _t0 = 0;
//CHECK-NEXT:     for (int i = 0; i < 3; i++) {
//CHECK-NEXT:         _t0++;
//CHECK-NEXT:         double arr[3] = {seed, clad::push(_t2, seed) * clad::push(_t1, i), seed + i};
//CHECK-NEXT:         clad::push(_t3, arr , 3UL);
//CHECK-NEXT:         sum += addArr(arr, 3);
//CHECK-NEXT:     }
//CHECK-NEXT:     _d_sum += 1;
//CHECK-NEXT:     for (; _t0; _t0--) {
//CHECK-NEXT:         {
//CHECK-NEXT:             double _r_d0 = _d_sum;
//CHECK-NEXT:             _d_sum += _r_d0;
//CHECK-NEXT:             clad::array<double> _r3 = clad::pop(_t3);
//CHECK-NEXT:             int _grad1 = 0;
//CHECK-NEXT:             addArr_pullback(_r3, 3, _r_d0, _d_arr, &_grad1);
//CHECK-NEXT:             clad::array<double> _r2(_d_arr);
//CHECK-NEXT:             int _r4 = _grad1;
//CHECK-NEXT:             _d_sum -= _r_d0;
//CHECK-NEXT:         }
//CHECK-NEXT:         {
//CHECK-NEXT:             * _d_seed += _d_arr[0];
//CHECK-NEXT:             double _r0 = _d_arr[1] * clad::pop(_t1);
//CHECK-NEXT:             * _d_seed += _r0;
//CHECK-NEXT:             double _r1 = clad::pop(_t2) * _d_arr[1];
//CHECK-NEXT:             _d_i += _r1;
//CHECK-NEXT:             * _d_seed += _d_arr[2];
//CHECK-NEXT:             _d_i += _d_arr[2];
//CHECK-NEXT:             _d_arr = {};
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
  printf("Result (b) = {%.2f, %.2f, %.2f}\n", db2[0], db2[1], db2[2]); // CHECK-EXEC: Result (b) = {0.00, 0.00, 0.00}

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
  printf("Result = {%.2f}", dseed); // CHECK-EXEC: Result = {9.00}
}
