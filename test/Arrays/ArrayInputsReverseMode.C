// RUN: %cladclang %s -I%S/../../include -Wno-unused-value -oArrayInputsReverseMode.out 2>&1 | %filecheck %s
// RUN: ./ArrayInputsReverseMode.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -enable-tbr %s -I%S/../../include -Wno-unused-value -oArrayInputsReverseMode.out
// RUN: ./ArrayInputsReverseMode.out | %filecheck_exec %s

//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

double addArr(const double *arr, int n) {
  double ret = 0;
  for (int i = 0; i < n; i++) {
    ret += arr[i];
  }
  return ret;
}

//CHECK: void addArr_pullback(const double *arr, int n, double _d_y, double *_d_arr, int *_d_n);

double f(double *arr) {
  return addArr(arr, 3);
}

//CHECK:   void f_grad(double *arr, double *_d_arr) {
//CHECK-NEXT:     {
//CHECK-NEXT:         int _r0 = 0;
//CHECK-NEXT:         addArr_pullback(arr, 3, 1, _d_arr, &_r0);
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

//CHECK: void func_grad(float *a, float *b, float *_d_a, float *_d_b) {
//CHECK-NEXT:     int _d_i = 0;
//CHECK-NEXT:     int i = 0;
//CHECK-NEXT:     clad::tape<float> _t1 = {};
//CHECK-NEXT:     clad::tape<float> _t2 = {};
//CHECK-NEXT:     float _d_sum = 0;
//CHECK-NEXT:     float sum = 0;
//CHECK-NEXT:     unsigned {{int|long}} _t0 = {{0U|0UL}};
//CHECK-NEXT:     for (i = 0; ; i++) {
// CHECK-NEXT:         {
// CHECK-NEXT:             if (!(i < 3))
// CHECK-NEXT:                 break;
// CHECK-NEXT:         }
//CHECK-NEXT:         _t0++;
//CHECK-NEXT:         clad::push(_t1, a[i]);
//CHECK-NEXT:         a[i] *= b[i];
//CHECK-NEXT:         clad::push(_t2, sum);
//CHECK-NEXT:         sum += a[i];
//CHECK-NEXT:     }
//CHECK-NEXT:     _d_sum += 1;
//CHECK-NEXT:     for (;; _t0--) {
// CHECK-NEXT:         {
// CHECK-NEXT:             if (!_t0)
// CHECK-NEXT:                 break;
// CHECK-NEXT:         }
//CHECK-NEXT:         i--;
//CHECK-NEXT:         {
//CHECK-NEXT:             sum = clad::pop(_t2);
//CHECK-NEXT:             float _r_d1 = _d_sum;
//CHECK-NEXT:             _d_a[i] += _r_d1;
//CHECK-NEXT:         }
//CHECK-NEXT:         {
//CHECK-NEXT:             a[i] = clad::pop(_t1);
//CHECK-NEXT:             float _r_d0 = _d_a[i];
//CHECK-NEXT:             _d_a[i] = 0;
//CHECK-NEXT:             _d_a[i] += _r_d0 * b[i];
//CHECK-NEXT:             _d_b[i] += a[i] * _r_d0;
//CHECK-NEXT:         }
//CHECK-NEXT:     }
//CHECK-NEXT: }

float helper(float x) {
  return 2 * x;
}

// CHECK: void helper_pullback(float x, float _d_y, float *_d_x);

float func2(float* a) {
  float sum = 0;
  for (int i = 0; i < 3; i++)
    sum += helper(a[i]);
  return sum;
}

//CHECK: void func2_grad(float *a, float *_d_a) {
//CHECK-NEXT:     int _d_i = 0;
//CHECK-NEXT:     int i = 0;
//CHECK-NEXT:     clad::tape<float> _t1 = {};
//CHECK-NEXT:     float _d_sum = 0;
//CHECK-NEXT:     float sum = 0;
//CHECK-NEXT:     unsigned {{int|long}} _t0 = {{0U|0UL}};
//CHECK-NEXT:     for (i = 0; ; i++) {
// CHECK-NEXT:         {
// CHECK-NEXT:             if (!(i < 3))
// CHECK-NEXT:                 break;
// CHECK-NEXT:         }
//CHECK-NEXT:         _t0++;
//CHECK-NEXT:         clad::push(_t1, sum);
//CHECK-NEXT:         sum += helper(a[i]);
//CHECK-NEXT:     }
//CHECK-NEXT:     _d_sum += 1;
//CHECK-NEXT:     for (;; _t0--) {
// CHECK-NEXT:         {
// CHECK-NEXT:             if (!_t0)
// CHECK-NEXT:                 break;
// CHECK-NEXT:         }
//CHECK-NEXT:         i--;
//CHECK-NEXT:         sum = clad::pop(_t1);
//CHECK-NEXT:         float _r_d0 = _d_sum;
//CHECK-NEXT:         float _r0 = 0;
//CHECK-NEXT:         helper_pullback(a[i], _r_d0, &_r0);
//CHECK-NEXT:         _d_a[i] += _r0;
//CHECK-NEXT:     }
//CHECK-NEXT: }

float func3(float* a, float* b) {
  float sum = 0;
  for (int i = 0; i < 3; i++)
    sum += (a[i] += b[i]);
  return sum;
}

//CHECK: void func3_grad(float *a, float *b, float *_d_a, float *_d_b) {
//CHECK-NEXT:     int _d_i = 0;
//CHECK-NEXT:     int i = 0;
//CHECK-NEXT:     clad::tape<float> _t1 = {};
//CHECK-NEXT:     clad::tape<float> _t2 = {};
//CHECK-NEXT:     float _d_sum = 0;
//CHECK-NEXT:     float sum = 0;
//CHECK-NEXT:     unsigned {{int|long}} _t0 = {{0U|0UL}};
//CHECK-NEXT:     for (i = 0; ; i++) {
// CHECK-NEXT:         {
// CHECK-NEXT:             if (!(i < 3))
// CHECK-NEXT:                 break;
// CHECK-NEXT:         }
//CHECK-NEXT:         _t0++;
//CHECK-NEXT:         clad::push(_t1, sum);
//CHECK-NEXT:         clad::push(_t2, a[i]);
//CHECK-NEXT:         sum += (a[i] += b[i]);
//CHECK-NEXT:     }
//CHECK-NEXT:     _d_sum += 1;
//CHECK-NEXT:     for (;; _t0--) {
// CHECK-NEXT:         {
// CHECK-NEXT:             if (!_t0)
// CHECK-NEXT:                 break;
// CHECK-NEXT:         }
//CHECK-NEXT:         i--;
//CHECK-NEXT:         sum = clad::pop(_t1);
//CHECK-NEXT:         float _r_d0 = _d_sum;
//CHECK-NEXT:         _d_a[i] += _r_d0;
//CHECK-NEXT:         a[i] = clad::pop(_t2);
//CHECK-NEXT:         float _r_d1 = _d_a[i];
//CHECK-NEXT:         _d_b[i] += _r_d1;
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

//CHECK: void func4_grad(double x, double *_d_x) {
//CHECK-NEXT:     int _d_i = 0;
//CHECK-NEXT:     int i = 0;
//CHECK-NEXT:     clad::tape<double> _t1 = {};
//CHECK-NEXT:     double _d_arr[3] = {0};
//CHECK-NEXT:     double arr[3] = {x, 2 * x, x * x};
//CHECK-NEXT:     double _d_sum = 0;
//CHECK-NEXT:     double sum = 0;
//CHECK-NEXT:     unsigned {{int|long}} _t0 = {{0U|0UL}};
//CHECK-NEXT:     for (i = 0; ; i++) {
// CHECK-NEXT:         {
// CHECK-NEXT:             if (!(i < 3))
// CHECK-NEXT:                 break;
// CHECK-NEXT:         }
//CHECK-NEXT:         _t0++;
//CHECK-NEXT:         clad::push(_t1, sum);
//CHECK-NEXT:         sum += addArr(arr, 3);
//CHECK-NEXT:     }
//CHECK-NEXT:     _d_sum += 1;
//CHECK-NEXT:     for (;; _t0--) {
// CHECK-NEXT:         {
// CHECK-NEXT:             if (!_t0)
// CHECK-NEXT:                 break;
// CHECK-NEXT:         }
//CHECK-NEXT:         i--;
//CHECK-NEXT:         {
//CHECK-NEXT:             sum = clad::pop(_t1);
//CHECK-NEXT:             double _r_d0 = _d_sum;
//CHECK-NEXT:             int _r0 = 0;
//CHECK-NEXT:             addArr_pullback(arr, 3, _r_d0, _d_arr, &_r0);
//CHECK-NEXT:         }
//CHECK-NEXT:     }
//CHECK-NEXT:     {
//CHECK-NEXT:         *_d_x += _d_arr[0];
//CHECK-NEXT:         *_d_x += 2 * _d_arr[1];
//CHECK-NEXT:         *_d_x += _d_arr[2] * x;
//CHECK-NEXT:         *_d_x += x * _d_arr[2];
//CHECK-NEXT:     }
//CHECK-NEXT: }

double func5(int k) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunknown-warning-option"
  #pragma clang diagnostic ignored "-Wvla-cxx-extension"
  int n = k;
  double arr[n];
  #pragma clang diagnostic pop
  for (int i = 0; i < n; i++) {
    arr[i] = k;
  }
  double sum = 0;
  for (int i = 0; i < 3; i++) {
    sum += addArr(arr, n);
  }
  return sum;
}

//CHECK: void func5_grad(int k, int *_d_k) {
//CHECK-NEXT:     int _d_i = 0;
//CHECK-NEXT:     int i = 0;
//CHECK-NEXT:     clad::tape<double> _t1 = {};
//CHECK-NEXT:     int _d_i0 = 0;
//CHECK-NEXT:     int i0 = 0;
//CHECK-NEXT:     clad::tape<double> _t3 = {};
//CHECK-NEXT:     int _d_n = 0;
//CHECK-NEXT:     int n = k;
//CHECK-NEXT:     double _d_arr[n];
//CHECK-NEXT:     clad::zero_init(_d_arr, n);
//CHECK-NEXT:     double arr[n];
//CHECK-NEXT:     unsigned {{int|long}} _t0 = {{0U|0UL}};
//CHECK-NEXT:     for (i = 0; ; i++) {
// CHECK-NEXT:         {
// CHECK-NEXT:             if (!(i < n))
// CHECK-NEXT:                 break;
// CHECK-NEXT:         }
//CHECK-NEXT:         _t0++;
//CHECK-NEXT:         clad::push(_t1, arr[i]);
//CHECK-NEXT:         arr[i] = k;
//CHECK-NEXT:     }
//CHECK-NEXT:     double _d_sum = 0;
//CHECK-NEXT:     double sum = 0;
//CHECK-NEXT:     unsigned {{int|long}} _t2 = {{0U|0UL}};
//CHECK-NEXT:     for (i0 = 0; ; i0++) {
// CHECK-NEXT:         {
// CHECK-NEXT:             if (!(i0 < 3))
// CHECK-NEXT:                 break;
// CHECK-NEXT:         }
//CHECK-NEXT:         _t2++;
//CHECK-NEXT:         clad::push(_t3, sum);
//CHECK-NEXT:         sum += addArr(arr, n);
//CHECK-NEXT:     }
//CHECK-NEXT:     _d_sum += 1;
//CHECK-NEXT:     for (;; _t2--) {
// CHECK-NEXT:         {
// CHECK-NEXT:             if (!_t2)
// CHECK-NEXT:                 break;
// CHECK-NEXT:         }
//CHECK-NEXT:         i0--;
//CHECK-NEXT:         {
//CHECK-NEXT:             sum = clad::pop(_t3);
//CHECK-NEXT:             double _r_d1 = _d_sum;
//CHECK-NEXT:             int _r0 = 0;
//CHECK-NEXT:             addArr_pullback(arr, n, _r_d1, _d_arr, &_r0);
//CHECK-NEXT:             _d_n += _r0;
//CHECK-NEXT:         }
//CHECK-NEXT:     }
//CHECK-NEXT:     for (;; _t0--) {
// CHECK-NEXT:         {
// CHECK-NEXT:             if (!_t0)
// CHECK-NEXT:                 break;
// CHECK-NEXT:         }
//CHECK-NEXT:         i--;
//CHECK-NEXT:         {
//CHECK-NEXT:             arr[i] = clad::pop(_t1);
//CHECK-NEXT:             double _r_d0 = _d_arr[i];
//CHECK-NEXT:             _d_arr[i] = 0;
//CHECK-NEXT:             *_d_k += _r_d0;
//CHECK-NEXT:         }
//CHECK-NEXT:     }
//CHECK-NEXT:     *_d_k += _d_n;
//CHECK-NEXT: }

double func6(double seed) {
  double sum = 0;
  for (int i = 0; i < 3; i++) {
    double arr[3] = {seed, seed * i, seed + i};
    sum += addArr(arr, 3);
  }
  return sum;
}

//CHECK: void func6_grad(double seed, double *_d_seed) {
//CHECK-NEXT:     int _d_i = 0;
//CHECK-NEXT:     int i = 0;
//CHECK-NEXT:     clad::tape<clad::array<double> > _t1 = {};
//CHECK-NEXT:     double _d_arr[3] = {0};
//CHECK-NEXT:     clad::array<double> arr({{3U|3UL}});
//CHECK-NEXT:     clad::tape<double> _t2 = {};
//CHECK-NEXT:     double _d_sum = 0;
//CHECK-NEXT:     double sum = 0;
//CHECK-NEXT:     unsigned {{int|long}} _t0 = {{0U|0UL}};
//CHECK-NEXT:     for (i = 0; ; i++) {
// CHECK-NEXT:         {
// CHECK-NEXT:             if (!(i < 3))
// CHECK-NEXT:                 break;
// CHECK-NEXT:         }
//CHECK-NEXT:         _t0++;
//CHECK-NEXT:         clad::push(_t1, arr) , arr = {seed, seed * i, seed + i};
//CHECK-NEXT:         clad::push(_t2, sum);
//CHECK-NEXT:         sum += addArr(arr, 3);
//CHECK-NEXT:     }
//CHECK-NEXT:     _d_sum += 1;
//CHECK-NEXT:     for (;; _t0--) {
// CHECK-NEXT:         {
// CHECK-NEXT:             if (!_t0)
// CHECK-NEXT:                 break;
// CHECK-NEXT:         }
//CHECK-NEXT:         i--;
//CHECK-NEXT:         {
//CHECK-NEXT:             sum = clad::pop(_t2);
//CHECK-NEXT:             double _r_d0 = _d_sum;
//CHECK-NEXT:             int _r0 = 0;
//CHECK-NEXT:             addArr_pullback(arr, 3, _r_d0, _d_arr, &_r0);
//CHECK-NEXT:         }
//CHECK-NEXT:         {
//CHECK-NEXT:             *_d_seed += _d_arr[0];
//CHECK-NEXT:             *_d_seed += _d_arr[1] * i;
//CHECK-NEXT:             _d_i += seed * _d_arr[1];
//CHECK-NEXT:             *_d_seed += _d_arr[2];
//CHECK-NEXT:             _d_i += _d_arr[2];
//CHECK-NEXT:             clad::zero_init(_d_arr);
//CHECK-NEXT:             arr = clad::pop(_t1);
//CHECK-NEXT:         }
//CHECK-NEXT:     }
//CHECK-NEXT: }

double inv_square(double *params) {
  return 1 / (params[0] * params[0]);
}

//CHECK: void inv_square_pullback(double *params, double _d_y, double *_d_params);

double func7(double *params) {
  double out = 0.0;
  for (std::size_t i = 0; i < 1; ++i) {
    double paramsPrime[1] = {params[0]};
    out = out + inv_square(paramsPrime);
  }
  return out;
}

//CHECK: void func7_grad(double *params, double *_d_params) {
//CHECK-NEXT:     std::size_t _d_i = 0;
//CHECK-NEXT:     std::size_t i = 0;
//CHECK-NEXT:     clad::tape<clad::array<double> > _t1 = {};
//CHECK-NEXT:     double _d_paramsPrime[1] = {0};
//CHECK-NEXT:     clad::array<double> paramsPrime({{1U|1UL}});
//CHECK-NEXT:     clad::tape<double> _t2 = {};
//CHECK-NEXT:     double _d_out = 0;
//CHECK-NEXT:     double out = 0.;
//CHECK-NEXT:     unsigned {{int|long}} _t0 = {{0U|0UL}};
// CHECK-NEXT:     for (i = 0; ; ++i) {
// CHECK-NEXT:         {
// CHECK-NEXT:             if (!(i < 1))
// CHECK-NEXT:                 break;
// CHECK-NEXT:         }
// CHECK-NEXT:         _t0++;
// CHECK-NEXT:         clad::push(_t1, paramsPrime) , paramsPrime = {params[0]};
// CHECK-NEXT:         clad::push(_t2, out);
// CHECK-NEXT:         out = out + inv_square(paramsPrime);
// CHECK-NEXT:     }
// CHECK-NEXT:     _d_out += 1;
// CHECK-NEXT:     for (;; _t0--) {
// CHECK-NEXT:         {
// CHECK-NEXT:             if (!_t0)
// CHECK-NEXT:                 break;
// CHECK-NEXT:         }
//CHECK-NEXT:         --i;
//CHECK-NEXT:         {
//CHECK-NEXT:             out = clad::pop(_t2);
//CHECK-NEXT:             double _r_d0 = _d_out;
//CHECK-NEXT:             _d_out = 0;
//CHECK-NEXT:             _d_out += _r_d0;
//CHECK-NEXT:             inv_square_pullback(paramsPrime, _r_d0, _d_paramsPrime);
//CHECK-NEXT:         }
//CHECK-NEXT:         {
//CHECK-NEXT:             _d_params[0] += _d_paramsPrime[0];
//CHECK-NEXT:             clad::zero_init(_d_paramsPrime);
//CHECK-NEXT:             paramsPrime = clad::pop(_t1);
//CHECK-NEXT:         }
//CHECK-NEXT:     }
//CHECK-NEXT: }

double helper2(double i, double *arr, int n) {
  return arr[0]*i;
}

//CHECK: void helper2_pullback(double i, double *arr, int n, double _d_y, double *_d_i, double *_d_arr, int *_d_n);

double func8(double i, double *arr, int n) {
  double res = 0;
  arr[0] = 1;
  res = helper2(i, arr, n);
  arr[0] = 5;
  return res;
}

//CHECK: void func8_grad(double i, double *arr, int n, double *_d_i, double *_d_arr, int *_d_n) {
//CHECK-NEXT:     double _d_res = 0;
//CHECK-NEXT:     double res = 0;
//CHECK-NEXT:     double _t0 = arr[0];
//CHECK-NEXT:     arr[0] = 1;
//CHECK-NEXT:     double _t1 = res;
//CHECK-NEXT:     res = helper2(i, arr, n);
//CHECK-NEXT:     double _t2 = arr[0];
//CHECK-NEXT:     arr[0] = 5;
//CHECK-NEXT:     _d_res += 1;
//CHECK-NEXT:     {
//CHECK-NEXT:         arr[0] = _t2;
//CHECK-NEXT:         double _r_d2 = _d_arr[0];
//CHECK-NEXT:         _d_arr[0] = 0;
//CHECK-NEXT:     }
//CHECK-NEXT:     {
//CHECK-NEXT:         res = _t1;
//CHECK-NEXT:         double _r_d1 = _d_res;
//CHECK-NEXT:         _d_res = 0;
//CHECK-NEXT:         double _r0 = 0;
//CHECK-NEXT:         int _r1 = 0;
//CHECK-NEXT:         helper2_pullback(i, arr, n, _r_d1, &_r0, _d_arr, &_r1);
//CHECK-NEXT:         *_d_i += _r0;
//CHECK-NEXT:         *_d_n += _r1;
//CHECK-NEXT:     }
//CHECK-NEXT:     {
//CHECK-NEXT:         arr[0] = _t0;
//CHECK-NEXT:         double _r_d0 = _d_arr[0];
//CHECK-NEXT:         _d_arr[0] = 0;
//CHECK-NEXT:     }
//CHECK-NEXT: }

void modify(double& elem, double val) {
  elem = val;
}

//CHECK: void modify_pullback(double &elem, double val, double *_d_elem, double *_d_val);

double func9(double i, double j) {
  double arr[5] = {};
  for (int idx = 0; idx < 5; ++idx) {
    modify(arr[idx], i);
  }
  return arr[0] + arr[1] + arr[2] + arr[3] + arr[4];
}


//CHECK: void func9_grad(double i, double j, double *_d_i, double *_d_j) {
//CHECK-NEXT:     int _d_idx = 0;
//CHECK-NEXT:     int idx = 0;
//CHECK-NEXT:     clad::tape<double> _t1 = {};
//CHECK-NEXT:     double _d_arr[5] = {0};
//CHECK-NEXT:     double arr[5] = {};
//CHECK-NEXT:     unsigned {{int|long}} _t0 = {{0U|0UL}};
// CHECK-NEXT:     for (idx = 0; ; ++idx) {
// CHECK-NEXT:         {
// CHECK-NEXT:             if (!(idx < 5))
// CHECK-NEXT:                 break;
// CHECK-NEXT:         }
// CHECK-NEXT:         _t0++;
// CHECK-NEXT:         clad::push(_t1, arr[idx]);
// CHECK-NEXT:         modify(arr[idx], i);
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         _d_arr[0] += 1;
// CHECK-NEXT:         _d_arr[1] += 1;
// CHECK-NEXT:         _d_arr[2] += 1;
// CHECK-NEXT:         _d_arr[3] += 1;
// CHECK-NEXT:         _d_arr[4] += 1;
// CHECK-NEXT:     }
// CHECK-NEXT:     for (;; _t0--) {
// CHECK-NEXT:         {
// CHECK-NEXT:             if (!_t0)
// CHECK-NEXT:                 break;
// CHECK-NEXT:         }
//CHECK-NEXT:         --idx;
//CHECK-NEXT:         {
//CHECK-NEXT:             arr[idx] = clad::back(_t1);
//CHECK-NEXT:             double _r0 = 0;
//CHECK-NEXT:             modify_pullback(clad::back(_t1), i, &_d_arr[idx], &_r0);
//CHECK-NEXT:             clad::pop(_t1);
//CHECK-NEXT:             *_d_i += _r0;
//CHECK-NEXT:         }
//CHECK-NEXT:     }
//CHECK-NEXT: }

double sq(double& elem) {
  elem = elem * elem;
  return elem;
}

//CHECK: void sq_pullback(double &elem, double _d_y, double *_d_elem);

double func10(double *arr, int n) {
  double res = 0;
  for (int i=0; i<n; ++i) {
    res += sq(arr[i]);
  }
  return res;
}

//CHECK: void func10_grad_0(double *arr, int n, double *_d_arr) {
//CHECK-NEXT:     int _d_n = 0;
//CHECK-NEXT:     int _d_i = 0;
//CHECK-NEXT:     int i = 0;
//CHECK-NEXT:     clad::tape<double> _t1 = {};
//CHECK-NEXT:     clad::tape<double> _t2 = {};
//CHECK-NEXT:     double _d_res = 0;
//CHECK-NEXT:     double res = 0;
//CHECK-NEXT:     unsigned {{int|long}} _t0 = {{0U|0UL}};
// CHECK-NEXT:     for (i = 0; ; ++i) {
// CHECK-NEXT:         {
// CHECK-NEXT:             if (!(i < n))
// CHECK-NEXT:                 break;
// CHECK-NEXT:         }
// CHECK-NEXT:         _t0++;
// CHECK-NEXT:         clad::push(_t1, res);
// CHECK-NEXT:         clad::push(_t2, arr[i]);
// CHECK-NEXT:         res += sq(arr[i]);
// CHECK-NEXT:     }
// CHECK-NEXT:     _d_res += 1;
// CHECK-NEXT:     for (;; _t0--) {
// CHECK-NEXT:         {
// CHECK-NEXT:             if (!_t0)
// CHECK-NEXT:                 break;
// CHECK-NEXT:         }
//CHECK-NEXT:         --i;
//CHECK-NEXT:         {
//CHECK-NEXT:             res = clad::pop(_t1);
//CHECK-NEXT:             double _r_d0 = _d_res;
//CHECK-NEXT:             arr[i] = clad::back(_t2);
//CHECK-NEXT:             sq_pullback(clad::back(_t2), _r_d0, &_d_arr[i]);
//CHECK-NEXT:             clad::pop(_t2);
//CHECK-NEXT:         }
//CHECK-NEXT:     }
//CHECK-NEXT: }

int main() {
  double arr[] = {1, 2, 3};
  auto f_dx = clad::gradient(f);

  double darr[3] = {};
  f_dx.execute(arr, darr);

  printf("Result = {%.2f, %.2f, %.2f}\n", darr[0], darr[1], darr[2]); // CHECK-EXEC: Result = {1.00, 1.00, 1.00}

  float a1[3] = {1, 1, 1}, a2[3] = {2, 3, 4}, a3[3] = {1, 1, 1}, b[3] = {2, 5, 2};
  float da1[3] = {0}, da2[3] = {0}, da3[3] = {0}, db1[3] = {0}, db2[3] = {0};

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

  auto func8grad = clad::gradient(func8);
  double arr2[5] = {1, 2, 3, 4, 5};
  double _d_arr2[5] = {};
  double d_i = 0, d_n = 0;
  func8grad.execute(3, arr, 5, &d_i, _d_arr2, &d_n);
  printf("Result = {%.2f}\n", d_i); // CHECK-EXEC: Result = {1.00}

  auto func9grad = clad::gradient(func9);
  double d_j;
  d_i = d_j = 0;
  func9grad.execute(3, 5, &d_i, &d_j);
  printf("Result = {%.2f}\n", d_i); // CHECK-EXEC: Result = {5.00}

  auto func10grad = clad::gradient(func10, "arr");
  double arr3[5] = {1, 2, 3, 4, 5};
  double _d_arr3[5] = {};
  func10grad.execute(arr3, 5, _d_arr3);
  printf("Result (arr) = {%.2f, %.2f, %.2f, %.2f, %.2f}\n", _d_arr3[0], _d_arr3[1], _d_arr3[2], _d_arr3[3], _d_arr3[4]); // CHECK-EXEC: Result (arr) = {2.00, 4.00, 6.00, 8.00, 10.00}
}

//CHECK: void addArr_pullback(const double *arr, int n, double _d_y, double *_d_arr, int *_d_n) {
//CHECK-NEXT:     int _d_i = 0;
//CHECK-NEXT:     int i = 0;
//CHECK-NEXT:     clad::tape<double> _t1 = {};
//CHECK-NEXT:     double _d_ret = 0;
//CHECK-NEXT:     double ret = 0;
//CHECK-NEXT:     unsigned {{int|long}} _t0 = {{0U|0UL}};
// CHECK-NEXT:     for (i = 0; ; i++) {
// CHECK-NEXT:         {
// CHECK-NEXT:             if (!(i < n))
// CHECK-NEXT:                 break;
// CHECK-NEXT:         }
// CHECK-NEXT:         _t0++;
// CHECK-NEXT:         clad::push(_t1, ret);
// CHECK-NEXT:         ret += arr[i];
// CHECK-NEXT:     }
// CHECK-NEXT:     _d_ret += _d_y;
// CHECK-NEXT:     for (;; _t0--) {
// CHECK-NEXT:         {
// CHECK-NEXT:             if (!_t0)
// CHECK-NEXT:                 break;
// CHECK-NEXT:         }
//CHECK-NEXT:         i--;
//CHECK-NEXT:         {
//CHECK-NEXT:             ret = clad::pop(_t1);
//CHECK-NEXT:             double _r_d0 = _d_ret;
//CHECK-NEXT:             _d_arr[i] += _r_d0;
//CHECK-NEXT:         }
//CHECK-NEXT:     }
//CHECK-NEXT: }

// CHECK: void helper_pullback(float x, float _d_y, float *_d_x) {
// CHECK-NEXT:     *_d_x += 2 * _d_y;
// CHECK-NEXT: }

//CHECK: void inv_square_pullback(double *params, double _d_y, double *_d_params) {
//CHECK-NEXT:     double _t0 = (params[0] * params[0]);
//CHECK-NEXT:     {
//CHECK-NEXT:         double _r0 = _d_y * -(1 / (_t0 * _t0));
//CHECK-NEXT:         _d_params[0] += _r0 * params[0];
//CHECK-NEXT:         _d_params[0] += params[0] * _r0;
//CHECK-NEXT:     }
//CHECK-NEXT: }

//CHECK: void helper2_pullback(double i, double *arr, int n, double _d_y, double *_d_i, double *_d_arr, int *_d_n) {
//CHECK-NEXT:     {
//CHECK-NEXT:         _d_arr[0] += _d_y * i;
//CHECK-NEXT:         *_d_i += arr[0] * _d_y;
//CHECK-NEXT:     }
//CHECK-NEXT: }

//CHECK: void modify_pullback(double &elem, double val, double *_d_elem, double *_d_val) {
//CHECK-NEXT:     double _t0 = elem;
//CHECK-NEXT:     elem = val;
//CHECK-NEXT:     {
//CHECK-NEXT:         elem = _t0;
//CHECK-NEXT:         double _r_d0 = *_d_elem;
//CHECK-NEXT:         *_d_elem = 0;
//CHECK-NEXT:         *_d_val += _r_d0;
//CHECK-NEXT:     }
//CHECK-NEXT: }

//CHECK: void sq_pullback(double &elem, double _d_y, double *_d_elem) {
//CHECK-NEXT:     double _t0 = elem;
//CHECK-NEXT:     elem = elem * elem;
//CHECK-NEXT:     *_d_elem += _d_y;
//CHECK-NEXT:     {
//CHECK-NEXT:         elem = _t0;
//CHECK-NEXT:         double _r_d0 = *_d_elem;
//CHECK-NEXT:         *_d_elem = 0;
//CHECK-NEXT:         *_d_elem += _r_d0 * elem;
//CHECK-NEXT:         *_d_elem += elem * _r_d0;
//CHECK-NEXT:     }
//CHECK-NEXT: }