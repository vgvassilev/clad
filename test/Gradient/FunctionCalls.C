// RUN: %cladnumdiffclang -std=c++17 %s  -I%S/../../include -oFunctionCalls.out 2>&1 | FileCheck %s
// RUN: ./FunctionCalls.out | FileCheck -check-prefix=CHECK-EXEC %s

//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

namespace A {
  template <typename T> T constantFn(T i) { return 3; }
  // CHECK: void constantFn_pullback(float i, float _d_y, clad::array_ref<float> _d_i) {
  // CHECK-NEXT:     goto _label0;
  // CHECK-NEXT:   _label0:
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
// CHECK-NEXT:     goto _label0;
// CHECK-NEXT:   _label0:
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
// CHECK-NEXT:     goto _label0;
// CHECK-NEXT:   _label0:
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
// CHECK-NEXT:     goto _label0;
// CHECK-NEXT:   _label0:
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
// CHECK-NEXT:     goto _label0;
// CHECK-NEXT:   _label0:
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
// CHECK-NEXT:     goto _label0;
// CHECK-NEXT:   _label0:
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
// CHECK-NEXT:     goto _label0;
// CHECK-NEXT:   _label0:
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
// CHECK-NEXT:     goto _label0;
// CHECK-NEXT:   _label0:
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
// CHECK-NEXT:     goto _label0;
// CHECK-NEXT:   _label0:
// CHECK-NEXT:     _d_arr[0] += 1;
// CHECK-NEXT:     {
// CHECK-NEXT:         modify2_pullback(_t0, _d_temp, _d_arr);
// CHECK-NEXT:         clad::array<double> _r0(_d_arr);
// CHECK-NEXT:     }
// CHECK-NEXT: }

double fn6(double i=0, double j=0) {
  return i*j;
}

struct MyStruct {
    static void myFunction() {}
};

double& identity(double& i) {
  MyStruct::myFunction();
  double _d_i = i;
  _d_i += 1;
  return i;
}

double fn7(double i, double j) {
  double& k = identity(i);
  double& l = identity(j);
  k += 7*j;
  l += 9*i;
  return i + j;
}

// CHECK: void fn6_grad(double i, double j, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j) {
// CHECK-NEXT:     double _t0;
// CHECK-NEXT:     double _t1;
// CHECK-NEXT:     _t1 = i;
// CHECK-NEXT:     _t0 = j;
// CHECK-NEXT:     goto _label0;
// CHECK-NEXT:   _label0:
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r0 = 1 * _t0;
// CHECK-NEXT:         * _d_i += _r0;
// CHECK-NEXT:         double _r1 = _t1 * 1;
// CHECK-NEXT:         * _d_j += _r1;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: void identity_pullback(double &i, double _d_y, clad::array_ref<double> _d_i) {
// CHECK-NEXT:     double _d__d_i = 0;
// CHECK-NEXT:     MyStruct::myFunction();
// CHECK-NEXT:     double _d_i0 = i;
// CHECK-NEXT:     _d_i0 += 1;
// CHECK-NEXT:     goto _label0;
// CHECK-NEXT:   _label0:
// CHECK-NEXT:     * _d_i += _d_y;
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r_d0 = _d__d_i;
// CHECK-NEXT:         _d__d_i += _r_d0;
// CHECK-NEXT:         _d__d_i -= _r_d0;
// CHECK-NEXT:     }
// CHECK-NEXT:     * _d_i += _d__d_i;
// CHECK-NEXT: }

// CHECK: clad::ValueAndAdjoint<double &, double &> identity_forw(double &i, clad::array_ref<double> _d_i) {
// CHECK-NEXT:     double _d__d_i = 0;
// CHECK-NEXT:     MyStruct::myFunction();
// CHECK-NEXT:     double _d_i0 = i;
// CHECK-NEXT:     _d_i0 += 1;
// CHECK-NEXT:     return {i, * _d_i};
// CHECK-NEXT: }

// CHECK: void fn7_grad(double i, double j, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j) {
// CHECK-NEXT:     double _t0;
// CHECK-NEXT:     double *_d_k = 0;
// CHECK-NEXT:     double _t2;
// CHECK-NEXT:     double *_d_l = 0;
// CHECK-NEXT:     double _t4;
// CHECK-NEXT:     double _t5;
// CHECK-NEXT:     _t0 = i;
// CHECK-NEXT:     clad::ValueAndAdjoint<double &, double &> _t1 = identity_forw(i, &* _d_i);
// CHECK-NEXT:     _d_k = &_t1.adjoint;
// CHECK-NEXT:     double &k = _t1.value;
// CHECK-NEXT:     _t2 = j;
// CHECK-NEXT:     clad::ValueAndAdjoint<double &, double &> _t3 = identity_forw(j, &* _d_j);
// CHECK-NEXT:     _d_l = &_t3.adjoint;
// CHECK-NEXT:     double &l = _t3.value;
// CHECK-NEXT:     _t4 = j;
// CHECK-NEXT:     k += 7 * _t4;
// CHECK-NEXT:     _t5 = i;
// CHECK-NEXT:     l += 9 * _t5;
// CHECK-NEXT:     goto _label0;
// CHECK-NEXT:   _label0:
// CHECK-NEXT:     {
// CHECK-NEXT:         * _d_i += 1;
// CHECK-NEXT:         * _d_j += 1;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r_d1 = *_d_l;
// CHECK-NEXT:         *_d_l += _r_d1;
// CHECK-NEXT:         double _r4 = _r_d1 * _t5;
// CHECK-NEXT:         double _r5 = 9 * _r_d1;
// CHECK-NEXT:         * _d_i += _r5;
// CHECK-NEXT:         *_d_l -= _r_d1;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r_d0 = *_d_k;
// CHECK-NEXT:         *_d_k += _r_d0;
// CHECK-NEXT:         double _r2 = _r_d0 * _t4;
// CHECK-NEXT:         double _r3 = 7 * _r_d0;
// CHECK-NEXT:         * _d_j += _r3;
// CHECK-NEXT:         *_d_k -= _r_d0;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         identity_pullback(_t2, 0, &* _d_j);
// CHECK-NEXT:         double _r1 = * _d_j;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         identity_pullback(_t0, 0, &* _d_i);
// CHECK-NEXT:         double _r0 = * _d_i;
// CHECK-NEXT:     }
// CHECK-NEXT: }

double fn8(double x, double y) {
  return x*y*std::tanh(1.0)*std::max(1.0, 2.0);
}

// CHECK: void fn8_grad(double x, double y, clad::array_ref<double> _d_x, clad::array_ref<double> _d_y) {
// CHECK-NEXT:     double _t0;
// CHECK-NEXT:     double _t1;
// CHECK-NEXT:     double _t2;
// CHECK-NEXT:     double _t3;
// CHECK-NEXT:     double _t4;
// CHECK-NEXT:     double _t5;
// CHECK-NEXT:     _t3 = x;
// CHECK-NEXT:     _t2 = y;
// CHECK-NEXT:     _t4 = _t3 * _t2;
// CHECK-NEXT:     _t1 = std::tanh(1.);
// CHECK-NEXT:     _t5 = _t4 * _t1;
// CHECK-NEXT:     _t0 = std::max(1., 2.);
// CHECK-NEXT:     goto _label0;
// CHECK-NEXT:   _label0:
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r0 = 1 * _t0;
// CHECK-NEXT:         double _r1 = _r0 * _t1;
// CHECK-NEXT:         double _r2 = _r1 * _t2;
// CHECK-NEXT:         * _d_x += _r2;
// CHECK-NEXT:         double _r3 = _t3 * _r1;
// CHECK-NEXT:         * _d_y += _r3;
// CHECK-NEXT:         double _r4 = _t4 * _r0;
// CHECK-NEXT:         double _r5 = _t5 * 1;
// CHECK-NEXT:     }
// CHECK-NEXT: }

double custom_max(const double& a, const double& b) {
  return a > b ? a : b;
}

// CHECK: void custom_max_pullback(const double &a, const double &b, double _d_y, clad::array_ref<double> _d_a, clad::array_ref<double> _d_b) {
// CHECK-NEXT:     bool _cond0;
// CHECK-NEXT:     _cond0 = a > b;
// CHECK-NEXT:     goto _label0;
// CHECK-NEXT:   _label0:
// CHECK-NEXT:     if (_cond0)
// CHECK-NEXT:         * _d_a += _d_y;
// CHECK-NEXT:     else
// CHECK-NEXT:         * _d_b += _d_y;
// CHECK-NEXT: }

double fn9(double x, double y) {
  return custom_max(x*y, y);
}

// CHECK: void fn9_grad(double x, double y, clad::array_ref<double> _d_x, clad::array_ref<double> _d_y) {
// CHECK-NEXT:    double _t0;
// CHECK-NEXT:    double _t1;
// CHECK-NEXT:    double _t2;
// CHECK-NEXT:    double _t3;
// CHECK-NEXT:    _t1 = x;
// CHECK-NEXT:    _t0 = y;
// CHECK-NEXT:    _t2 = _t1 * _t0;
// CHECK-NEXT:    _t3 = y;
// CHECK-NEXT:    goto _label0;
// CHECK-NEXT:  _label0:
// CHECK-NEXT:    {
// CHECK-NEXT:        double _grad0 = 0.;
// CHECK-NEXT:        custom_max_pullback(_t2, _t3, 1, &_grad0, &* _d_y);
// CHECK-NEXT:        double _r0 = _grad0;
// CHECK-NEXT:        double _r1 = _r0 * _t0;
// CHECK-NEXT:        * _d_x += _r1;
// CHECK-NEXT:        double _r2 = _t1 * _r0;
// CHECK-NEXT:        * _d_y += _r2;
// CHECK-NEXT:        double _r3 = * _d_y;
// CHECK-NEXT:    }
// CHECK-NEXT: }

double fn10(double x, double y) {
  double out = x;
  out = std::max(out, 0.0);
  out = std::min(out, 10.0);
  out = std::clamp(out, 3.0, 7.0);
  return out * y;
}

// CHECK: void fn10_grad(double x, double y, clad::array_ref<double> _d_x, clad::array_ref<double> _d_y) {
// CHECK-NEXT:    double _d_out = 0;
// CHECK-NEXT:    double _t0;
// CHECK-NEXT:    double _t1;
// CHECK-NEXT:    double _t2;
// CHECK-NEXT:    double _t3;
// CHECK-NEXT:    double _t4;
// CHECK-NEXT:    double out = x;
// CHECK-NEXT:    _t0 = out;
// CHECK-NEXT:    out = std::max(out, 0.);
// CHECK-NEXT:    _t1 = out;
// CHECK-NEXT:    out = std::min(out, 10.);
// CHECK-NEXT:    _t2 = out;
// CHECK-NEXT:    out = std::clamp(out, 3., 7.);
// CHECK-NEXT:    _t4 = out;
// CHECK-NEXT:    _t3 = y;
// CHECK-NEXT:    goto _label0;
// CHECK-NEXT:  _label0:
// CHECK-NEXT:    {
// CHECK-NEXT:        double _r7 = 1 * _t3;
// CHECK-NEXT:        _d_out += _r7;
// CHECK-NEXT:        double _r8 = _t4 * 1;
// CHECK-NEXT:        * _d_y += _r8;
// CHECK-NEXT:    }
// CHECK-NEXT:    {
// CHECK-NEXT:        double _r_d2 = _d_out;
// CHECK-NEXT:        double _grad5 = 0.;
// CHECK-NEXT:        double _grad6 = 0.;
// CHECK-NEXT:        clad::custom_derivatives::std::clamp_pullback(_t2, 3., 7., _r_d2, &_d_out, &_grad5, &_grad6);
// CHECK-NEXT:        double _r4 = _d_out;
// CHECK-NEXT:        double _r5 = _grad5;
// CHECK-NEXT:        double _r6 = _grad6;
// CHECK-NEXT:        _d_out -= _r_d2;
// CHECK-NEXT:    }
// CHECK-NEXT:    {
// CHECK-NEXT:        double _r_d1 = _d_out;
// CHECK-NEXT:        double _grad3 = 0.;
// CHECK-NEXT:        clad::custom_derivatives::std::min_pullback(_t1, 10., _r_d1, &_d_out, &_grad3);
// CHECK-NEXT:        double _r2 = _d_out;
// CHECK-NEXT:        double _r3 = _grad3;
// CHECK-NEXT:        _d_out -= _r_d1;
// CHECK-NEXT:    }
// CHECK-NEXT:    {
// CHECK-NEXT:        double _r_d0 = _d_out;
// CHECK-NEXT:        double _grad1 = 0.;
// CHECK-NEXT:        clad::custom_derivatives::std::max_pullback(_t0, 0., _r_d0, &_d_out, &_grad1);
// CHECK-NEXT:        double _r0 = _d_out;
// CHECK-NEXT:        double _r1 = _grad1;
// CHECK-NEXT:        _d_out -= _r_d0;
// CHECK-NEXT:    }
// CHECK-NEXT:    * _d_x += _d_out;
// CHECK-NEXT: }

namespace n1{
  inline namespace n2{
    double sum(const double& x, const double& y) {
      return x + y;
    }
  }
}

namespace clad{
namespace custom_derivatives{
  namespace n1{
    inline namespace n2{
      void sum_pullback(const double& x, const double& y, double _d_y0, clad::array_ref<double> _d_x, clad::array_ref<double> _d_y) {
        * _d_x += _d_y0;
        * _d_y += _d_y0;
      }
    }
  }
}
}

double fn11(double x, double y) {
  return n1::n2::sum(x, y);
}

// CHECK: void fn11_grad(double x, double y, clad::array_ref<double> _d_x, clad::array_ref<double> _d_y) {
// CHECK-NEXT:    double _t0;
// CHECK-NEXT:    double _t1;
// CHECK-NEXT:    _t0 = x;
// CHECK-NEXT:    _t1 = y;
// CHECK-NEXT:    goto _label0;
// CHECK-NEXT:  _label0:
// CHECK-NEXT:    {
// CHECK-NEXT:        clad::custom_derivatives::n1::sum_pullback(_t0, _t1, 1, &* _d_x, &* _d_y);
// CHECK-NEXT:        double _r0 = * _d_x;
// CHECK-NEXT:        double _r1 = * _d_y;
// CHECK-NEXT:    }
// CHECK-NEXT: }

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
  INIT(fn7);
  INIT(fn8);
  INIT(fn9);
  INIT(fn10);
  INIT(fn11);

  TEST1_float(fn1, 11);         // CHECK-EXEC: {3.00}
  TEST2(fn2, 3, 5);             // CHECK-EXEC: {1.00, 3.00}
  TEST2(fn3, 3, 5);             // CHECK-EXEC: {1.00, 3.00}
  double arr[5] = {1, 2, 3, 4, 5};
  TEST_ARR5(fn4, arr, 5);       // CHECK-EXEC: {23.00, 3.00, 3.00, 3.00, 3.00}
  TEST_ARR5(fn5, arr, 5);       // CHECK-EXEC: {5.00, 1.00, 0.00, 0.00, 0.00}
  TEST2(fn6, 3, 5);             // CHECK-EXEC: {5.00, 3.00}
  TEST2(fn7, 3, 5);             // CHECK-EXEC: {10.00, 71.00}
  TEST2(fn8, 3, 5);             // CHECK-EXEC: {7.62, 4.57}
  TEST2(fn9, 3, 5);             // CHECK-EXEC: {5.00, 3.00}
  TEST2(fn10, 8, 5);            // CHECK-EXEC: {0.00, 7.00}
  TEST2(fn11, 3, 5);            // CHECK-EXEC: {1.00, 1.00}
}
