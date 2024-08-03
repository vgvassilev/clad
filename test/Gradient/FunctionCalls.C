// RUN: %cladnumdiffclang -std=c++17 -Wno-writable-strings %s  -I%S/../../include -oFunctionCalls.out 2>&1 | %filecheck %s
// RUN: ./FunctionCalls.out | %filecheck_exec %s
// RUN: %cladnumdiffclang -Xclang -plugin-arg-clad -Xclang -enable-tbr -std=c++17 -Wno-writable-strings %s  -I%S/../../include -oFunctionCalls.out
// RUN: ./FunctionCalls.out | %filecheck_exec %s

// CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

namespace A {
  template <typename T> T constantFn(T i) { return 3; }
  // CHECK: void constantFn_pullback(float i, float _d_y, float *_d_i);
} // namespace A

double constantFn(double i) {
  return 5;
}

double fn1(float i) {
  float res = A::constantFn(i);
  double a = res*i;
  return a;
}

// CHECK: void fn1_grad(float i, float *_d_i) {
// CHECK-NEXT:     float _d_res = 0;
// CHECK-NEXT:     float res = A::constantFn(i);
// CHECK-NEXT:     double _d_a = 0;
// CHECK-NEXT:     double a = res * i;
// CHECK-NEXT:     _d_a += 1;
// CHECK-NEXT:     {
// CHECK-NEXT:         _d_res += _d_a * i;
// CHECK-NEXT:         *_d_i += res * _d_a;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         float _r0 = 0;
// CHECK-NEXT:         constantFn_pullback(i, _d_res, &_r0);
// CHECK-NEXT:         *_d_i += _r0;
// CHECK-NEXT:     }
// CHECK-NEXT: }

double modify1(double& i, double& j) {
  i += j;
  j += j;
  double res = i + j;
  return res;
}

// CHECK: void modify1_pullback(double &i, double &j, double _d_y, double *_d_i, double *_d_j);

double fn2(double i, double j) {
  double temp = 0;
  temp = modify1(i, j);
  temp = modify1(i, j);
  return i;
}

// CHECK: void fn2_grad(double i, double j, double *_d_i, double *_d_j) {
// CHECK-NEXT:     double _d_temp = 0;
// CHECK-NEXT:     double temp = 0;
// CHECK-NEXT:     double _t0 = temp;
// CHECK-NEXT:     double _t1 = i;
// CHECK-NEXT:     double _t2 = j;
// CHECK-NEXT:     temp = modify1(i, j);
// CHECK-NEXT:     double _t3 = temp;
// CHECK-NEXT:     double _t4 = i;
// CHECK-NEXT:     double _t5 = j;
// CHECK-NEXT:     temp = modify1(i, j);
// CHECK-NEXT:     *_d_i += 1;
// CHECK-NEXT:     {
// CHECK-NEXT:         temp = _t3;
// CHECK-NEXT:         double _r_d1 = _d_temp;
// CHECK-NEXT:         _d_temp = 0;
// CHECK-NEXT:         i = _t4;
// CHECK-NEXT:         j = _t5;
// CHECK-NEXT:         modify1_pullback(_t4, _t5, _r_d1, &*_d_i, &*_d_j);
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         temp = _t0;
// CHECK-NEXT:         double _r_d0 = _d_temp;
// CHECK-NEXT:         _d_temp = 0;
// CHECK-NEXT:         i = _t1;
// CHECK-NEXT:         j = _t2;
// CHECK-NEXT:         modify1_pullback(_t1, _t2, _r_d0, &*_d_i, &*_d_j);
// CHECK-NEXT:     }
// CHECK-NEXT: }

void update1(double& i, double& j) {
  i += j;
  j += j;
}

// CHECK: void update1_pullback(double &i, double &j, double *_d_i, double *_d_j);

double fn3(double i, double j) {
  update1(i, j);
  update1(i, j);
  return i;
}

// CHECK: void fn3_grad(double i, double j, double *_d_i, double *_d_j) {
// CHECK-NEXT:     double _t0 = i;
// CHECK-NEXT:     double _t1 = j;
// CHECK-NEXT:     update1(i, j);
// CHECK-NEXT:     double _t2 = i;
// CHECK-NEXT:     double _t3 = j;
// CHECK-NEXT:     update1(i, j);
// CHECK-NEXT:     *_d_i += 1;
// CHECK-NEXT:     {
// CHECK-NEXT:         i = _t2;
// CHECK-NEXT:         j = _t3;
// CHECK-NEXT:         update1_pullback(_t2, _t3, &*_d_i, &*_d_j);
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         i = _t0;
// CHECK-NEXT:         j = _t1;
// CHECK-NEXT:         update1_pullback(_t0, _t1, &*_d_i, &*_d_j);
// CHECK-NEXT:     }
// CHECK-NEXT: }

float sum(double* arr, int n) {
  float res = 0;
  for (int i=0; i<n; ++i)
    res += arr[i];
  arr[0] += 10*arr[0];
  return res;
}

// CHECK: void sum_pullback(double *arr, int n, float _d_y, double *_d_arr, int *_d_n);

void twice(double& d) {
  d = 2*d;
}

// CHECK: void twice_pullback(double &d, double *_d_d);

double fn4(double* arr, int n) {
  double res = 0;
  res += sum(arr, n);
  for (int i=0; i<n; ++i) {
    twice(arr[i]);
    res += arr[i];
  }
  return res;
}

// CHECK: void fn4_grad(double *arr, int n, double *_d_arr, int *_d_n) {
// CHECK-NEXT:     int _d_i = 0;
// CHECK-NEXT:     int i = 0;
// CHECK-NEXT:     clad::tape<double> _t2 = {};
// CHECK-NEXT:     clad::tape<double> _t3 = {};
// CHECK-NEXT:     double _d_res = 0;
// CHECK-NEXT:     double res = 0;
// CHECK-NEXT:     double _t0 = res;
// CHECK-NEXT:     res += sum(arr, n);
// CHECK-NEXT:     unsigned {{int|long}} _t1 = {{0U|0UL}};
// CHECK-NEXT:     for (i = 0; ; ++i) {
// CHECK-NEXT:     {
// CHECK-NEXT:          if (!(i < n))
// CHECK-NEXT:          break;
// CHECK-NEXT:     }
// CHECK-NEXT:         _t1++;
// CHECK-NEXT:         clad::push(_t2, arr[i]);
// CHECK-NEXT:         twice(arr[i]);
// CHECK-NEXT:         clad::push(_t3, res);
// CHECK-NEXT:         res += arr[i];
// CHECK-NEXT:     }
// CHECK-NEXT:     _d_res += 1;
// CHECK-NEXT:     for (;; _t1--) {
// CHECK-NEXT:         {
// CHECK-NEXT:             if (!_t1)
// CHECK-NEXT:                 break;
// CHECK-NEXT:         }
// CHECK-NEXT:         --i;
// CHECK-NEXT:         {
// CHECK-NEXT:             res = clad::pop(_t3);
// CHECK-NEXT:             double _r_d1 = _d_res;
// CHECK-NEXT:             _d_arr[i] += _r_d1;
// CHECK-NEXT:         }
// CHECK-NEXT:         {
// CHECK-NEXT:             arr[i] = clad::back(_t2);
// CHECK-NEXT:             twice_pullback(clad::back(_t2), &_d_arr[i]);
// CHECK-NEXT:             clad::pop(_t2);
// CHECK-NEXT:         }
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         res = _t0;
// CHECK-NEXT:         double _r_d0 = _d_res;
// CHECK-NEXT:         int _r0 = 0;
// CHECK-NEXT:         sum_pullback(arr, n, _r_d0, _d_arr, &_r0);
// CHECK-NEXT:         *_d_n += _r0;
// CHECK-NEXT:     }
// CHECK-NEXT: }

double modify2(double* arr) {
    arr[0] = 5*arr[0] + arr[1];
    return 1;
}

// CHECK: void modify2_pullback(double *arr, double _d_y, double *_d_arr);

double fn5(double* arr, int n) {
    double temp = modify2(arr);
    return arr[0];
}

// CHECK: void fn5_grad(double *arr, int n, double *_d_arr, int *_d_n) {
// CHECK-NEXT:     double _d_temp = 0;
// CHECK-NEXT:     double temp = modify2(arr);
// CHECK-NEXT:     _d_arr[0] += 1;
// CHECK-NEXT:     modify2_pullback(arr, _d_temp, _d_arr);
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

namespace clad{
namespace custom_derivatives{
  clad::ValueAndAdjoint<double &, double &> custom_identity_forw(double &i, double *d_i) {
    return {i, *d_i};
  }
} // namespace custom_derivatives
} // namespace clad

double& custom_identity(double& i) {
  return i;
}

double fn7(double i, double j) {
  double& k = identity(i);
  double& l = identity(j);
  double& temp = custom_identity(i);
  k += 7*j;
  l += 9*i;
  return i + j + temp;
}

// CHECK: void fn6_grad(double i, double j, double *_d_i, double *_d_j) {
// CHECK-NEXT:     {
// CHECK-NEXT:         *_d_i += 1 * j;
// CHECK-NEXT:         *_d_j += i * 1;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: void identity_pullback(double &i, double _d_y, double *_d_i);

// CHECK: clad::ValueAndAdjoint<double &, double &> identity_forw(double &i, double *_d_i);

// CHECK: void custom_identity_pullback(double &i, double _d_y, double *_d_i);

// CHECK: clad::ValueAndAdjoint<double &, double &> custom_identity_forw(double &i, double *d_i) {
// CHECK-NEXT:     return {i, *d_i};
// CHECK-NEXT: }

// CHECK: void fn7_grad(double i, double j, double *_d_i, double *_d_j) {
// CHECK-NEXT:     double _t0 = i;
// CHECK-NEXT:     clad::ValueAndAdjoint<double &, double &> _t1 = identity_forw(i, &*_d_i);
// CHECK-NEXT:     double &_d_k = _t1.adjoint;
// CHECK-NEXT:     double &k = _t1.value;
// CHECK-NEXT:     double _t2 = j;
// CHECK-NEXT:     clad::ValueAndAdjoint<double &, double &> _t3 = identity_forw(j, &*_d_j);
// CHECK-NEXT:     double &_d_l = _t3.adjoint;
// CHECK-NEXT:     double &l = _t3.value;
// CHECK-NEXT:     double _t4 = i;
// CHECK-NEXT:     clad::ValueAndAdjoint<double &, double &> _t5 = custom_identity_forw(i, &*_d_i);
// CHECK-NEXT:     double &_d_temp = _t5.adjoint;
// CHECK-NEXT:     double &temp = _t5.value;
// CHECK-NEXT:     double _t6 = k;
// CHECK-NEXT:     k += 7 * j;
// CHECK-NEXT:     double _t7 = l;
// CHECK-NEXT:     l += 9 * i;
// CHECK-NEXT:     {
// CHECK-NEXT:         *_d_i += 1;
// CHECK-NEXT:         *_d_j += 1;
// CHECK-NEXT:         _d_temp += 1;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         l = _t7;
// CHECK-NEXT:         double _r_d1 = _d_l;
// CHECK-NEXT:         *_d_i += 9 * _r_d1;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         k = _t6;
// CHECK-NEXT:         double _r_d0 = _d_k;
// CHECK-NEXT:         *_d_j += 7 * _r_d0;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         i = _t4;
// CHECK-NEXT:         custom_identity_pullback(_t4, 0, &*_d_i);
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         j = _t2;
// CHECK-NEXT:         identity_pullback(_t2, 0, &*_d_j);
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         i = _t0;
// CHECK-NEXT:         identity_pullback(_t0, 0, &*_d_i);
// CHECK-NEXT:     }
// CHECK-NEXT: }

double check_and_return(double x, char c, const char* s) {
  if (c == 'a' && s[0] == 'a')
    return x;
  return 1;
}

// CHECK: void check_and_return_pullback(double x, char c, const char *s, double _d_y, double *_d_x, char *_d_c, char *_d_s);

double fn8(double x, double y) {
  return check_and_return(x, 'a', "aa") * y * std::tanh(1.0) * std::max(1.0, 2.0); // expected-warning {{ISO C++11 does not allow conversion from string literal to 'char *' [-Wwritable-strings]}}
}

// CHECK: void fn8_grad(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:     double _t2 = check_and_return(x, 'a', "aa");
// CHECK-NEXT:     double _t1 = std::tanh(1.);
// CHECK-NEXT:     double _t0 = std::max(1., 2.);
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r0 = 0;
// CHECK-NEXT:         char _r1 = 0;
// CHECK-NEXT:         check_and_return_pullback(x, 'a', "aa", 1 * _t0 * _t1 * y, &_r0, &_r1, "");
// CHECK-NEXT:         *_d_x += _r0;
// CHECK-NEXT:         *_d_y += _t2 * 1 * _t0 * _t1;
// CHECK-NEXT:     }
// CHECK-NEXT: }

double custom_max(const double& a, const double& b) {
  return a > b ? a : b;
}

// CHECK: void custom_max_pullback(const double &a, const double &b, double _d_y, double *_d_a, double *_d_b);

double fn9(double x, double y) {
  return custom_max(x*y, y);
}

// CHECK:void fn9_grad(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:    double _t0 = y;
// CHECK-NEXT:    {
// CHECK-NEXT:        y = _t0;
// CHECK-NEXT:        double _r0 = 0;
// CHECK-NEXT:        custom_max_pullback(x * y, _t0, 1, &_r0, &*_d_y);
// CHECK-NEXT:        *_d_x += _r0 * y;
// CHECK-NEXT:        *_d_y += x * _r0;
// CHECK-NEXT:    }
// CHECK-NEXT: }

double fn10(double x, double y) {
  double out = x;
  out = std::max(out, 0.0);
  out = std::min(out, 10.0);
  out = std::clamp(out, 3.0, 7.0);
  return out * y;
}

// CHECK: void fn10_grad(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:    double _d_out = 0;
// CHECK-NEXT:    double out = x;
// CHECK-NEXT:    double _t0 = out;
// CHECK-NEXT:    double _t1 = out;
// CHECK-NEXT:    out = std::max(out, 0.);
// CHECK-NEXT:    double _t2 = out;
// CHECK-NEXT:    double _t3 = out;
// CHECK-NEXT:    out = std::min(out, 10.);
// CHECK-NEXT:    double _t4 = out;
// CHECK-NEXT:    double _t5 = out;
// CHECK-NEXT:    out = std::clamp(out, 3., 7.);
// CHECK-NEXT:    {
// CHECK-NEXT:        _d_out += 1 * y;
// CHECK-NEXT:        *_d_y += out * 1;
// CHECK-NEXT:    }
// CHECK-NEXT:    {
// CHECK-NEXT:        out = _t4;
// CHECK-NEXT:        double _r_d2 = _d_out;
// CHECK-NEXT:        _d_out = 0;
// CHECK-NEXT:        out = _t5;
// CHECK-NEXT:        double _r2 = 0;
// CHECK-NEXT:        double _r3 = 0;
// CHECK-NEXT:        clad::custom_derivatives::std::clamp_pullback(_t5, 3., 7., _r_d2, &_d_out, &_r2, &_r3);
// CHECK-NEXT:    }
// CHECK-NEXT:    {
// CHECK-NEXT:        out = _t2;
// CHECK-NEXT:        double _r_d1 = _d_out;
// CHECK-NEXT:        _d_out = 0;
// CHECK-NEXT:        out = _t3;
// CHECK-NEXT:        double _r1 = 0;
// CHECK-NEXT:        clad::custom_derivatives::std::min_pullback(_t3, 10., _r_d1, &_d_out, &_r1);
// CHECK-NEXT:    }
// CHECK-NEXT:    {
// CHECK-NEXT:        out = _t0;
// CHECK-NEXT:        double _r_d0 = _d_out;
// CHECK-NEXT:        _d_out = 0;
// CHECK-NEXT:        out = _t1;
// CHECK-NEXT:        double _r0 = 0;
// CHECK-NEXT:        clad::custom_derivatives::std::max_pullback(_t1, 0., _r_d0, &_d_out, &_r0);
// CHECK-NEXT:    }
// CHECK-NEXT:    *_d_x += _d_out;
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
      void sum_pullback(const double& x, const double& y, double _d_y0, double *_d_x, double *_d_y) {
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

// CHECK: void fn11_grad(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:    double _t0 = x;
// CHECK-NEXT:    double _t1 = y;
// CHECK-NEXT:    {
// CHECK-NEXT:        x = _t0;
// CHECK-NEXT:        y = _t1;
// CHECK-NEXT:        clad::custom_derivatives::n1::sum_pullback(_t0, _t1, 1, &*_d_x, &*_d_y);
// CHECK-NEXT:    }
// CHECK-NEXT: }

double do_nothing(double* u, double* v, double* w) {
  return u[0];
}

// CHECK: void do_nothing_pullback(double *u, double *v, double *w, double _d_y, double *_d_u, double *_d_v, double *_d_w);

double fn12(double x, double y) {
  return do_nothing(&x, nullptr, 0);
}

// CHECK: void fn12_grad(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:     do_nothing_pullback(&x, nullptr, 0, 1, &*_d_x, nullptr, 0);
// CHECK-NEXT: }

double multiply(double* a, double* b) {
  return a[0] * b[0];
}

// CHECK: void multiply_pullback(double *a, double *b, double _d_y, double *_d_a, double *_d_b);

double fn13(double* x, const double* w) {
  double wCopy[2];
  for(std::size_t i = 0; i < 2; ++i) { wCopy[i] = w[i]; }
  return multiply(x, wCopy + 1);
}

// CHECK: void fn13_grad_0(double *x, const double *w, double *_d_x) {
// CHECK-NEXT:     std::size_t _d_i = 0;
// CHECK-NEXT:     std::size_t i = 0;
// CHECK-NEXT:     clad::tape<double> _t1 = {};
// CHECK-NEXT:     double _d_wCopy[2] = {0};
// CHECK-NEXT:     double wCopy[2];
// CHECK-NEXT:     unsigned {{int|long}} _t0 = {{0U|0UL}};
// CHECK-NEXT:     for (i = 0; ; ++i) {
// CHECK-NEXT:         {
// CHECK-NEXT:             if (!(i < 2))
// CHECK-NEXT:                 break;
// CHECK-NEXT:         }
// CHECK-NEXT:         _t0++;
// CHECK-NEXT:         clad::push(_t1, wCopy[i]);
// CHECK-NEXT:         wCopy[i] = w[i];
// CHECK-NEXT:     }
// CHECK-NEXT:     multiply_pullback(x, wCopy + 1, 1, _d_x, _d_wCopy + 1);
// CHECK-NEXT:     for (;; _t0--) {
// CHECK-NEXT:         {
// CHECK-NEXT:             if (!_t0)
// CHECK-NEXT:                 break;
// CHECK-NEXT:         }
// CHECK-NEXT:         --i;
// CHECK-NEXT:         {
// CHECK-NEXT:             wCopy[i] = clad::pop(_t1);
// CHECK-NEXT:             double _r_d0 = _d_wCopy[i];
// CHECK-NEXT:             _d_wCopy[i] = 0;
// CHECK-NEXT:         }
// CHECK-NEXT:     }
// CHECK-NEXT: }

void emptyFn(double &x, double y) {}

// CHECK: void emptyFn_pullback(double &x, double y, double *_d_x, double *_d_y);

double fn14(double x, double y) {
    emptyFn(x, y);
    return x + y;
}

// CHECK: void fn14_grad(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:     double _t0 = x;
// CHECK-NEXT:     emptyFn(x, y);
// CHECK-NEXT:     {
// CHECK-NEXT:         *_d_x += 1;
// CHECK-NEXT:         *_d_y += 1;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         x = _t0;
// CHECK-NEXT:         double _r0 = 0;
// CHECK-NEXT:         emptyFn_pullback(_t0, y, &*_d_x, &_r0);
// CHECK-NEXT:         *_d_y += _r0;
// CHECK-NEXT:     }
// CHECK-NEXT: }

double fn15(double x, double y) {
    A::constantFn(y += x);
    return y;
}

//CHECK: void fn15_grad(double x, double y, double *_d_x, double *_d_y) {
//CHECK-NEXT:     double _t0 = y;
//CHECK-NEXT:     A::constantFn(y += x);
//CHECK-NEXT:     *_d_y += 1;
//CHECK-NEXT:     {
//CHECK-NEXT:         y = _t0;
//CHECK-NEXT:         double _r_d0 = *_d_y;
//CHECK-NEXT:         *_d_x += _r_d0;
//CHECK-NEXT:     }
//CHECK-NEXT: }

double recFun (double x, double y) {
    if (x > y)
        return recFun(x-1, y);
    return x * y;
}

//CHECK: void recFun_pullback(double x, double y, double _d_y0, double *_d_x, double *_d_y);

double fn16(double x, double y) {
    return recFun(x, y);
}

//CHECK: void fn16_grad(double x, double y, double *_d_x, double *_d_y) {
//CHECK-NEXT:     {
//CHECK-NEXT:         double _r0 = 0;
//CHECK-NEXT:         double _r1 = 0;
//CHECK-NEXT:         recFun_pullback(x, y, 1, &_r0, &_r1);
//CHECK-NEXT:         *_d_x += _r0;
//CHECK-NEXT:         *_d_y += _r1;
//CHECK-NEXT:     }
//CHECK-NEXT: }

double add(double a, double* b) {
    return a + b[0];
}


//CHECK: void add_pullback(double a, double *b, double _d_y, double *_d_a);

//CHECK: void add_pullback(double a, double *b, double _d_y, double *_d_a, double *_d_b);

double fn17 (double x, double* y) {
    x = add(x, y);
    x = add(x, &x);
    return x;
}

//CHECK: void fn17_grad_0(double x, double *y, double *_d_x) {
//CHECK-NEXT:     double _t0 = x;
//CHECK-NEXT:     x = add(x, y);
//CHECK-NEXT:     double _t1 = x;
//CHECK-NEXT:     x = add(x, &x);
//CHECK-NEXT:     *_d_x += 1;
//CHECK-NEXT:     {
//CHECK-NEXT:         x = _t1;
//CHECK-NEXT:         double _r_d1 = *_d_x;
//CHECK-NEXT:         *_d_x = 0;
//CHECK-NEXT:         double _r1 = 0;
//CHECK-NEXT:         add_pullback(x, &x, _r_d1, &_r1, &*_d_x);
//CHECK-NEXT:         *_d_x += _r1;
//CHECK-NEXT:     }
//CHECK-NEXT:     {
//CHECK-NEXT:         x = _t0;
//CHECK-NEXT:         double _r_d0 = *_d_x;
//CHECK-NEXT:         *_d_x = 0;
//CHECK-NEXT:         double _r0 = 0;
//CHECK-NEXT:         add_pullback(x, y, _r_d0, &_r0);
//CHECK-NEXT:         *_d_x += _r0;
//CHECK-NEXT:     }
//CHECK-NEXT: }

double sq_defined_later(double x);

// CHECK: void sq_defined_later_pullback(double x, double _d_y, double *_d_x);

double fn18(double x, double y) {
    return sq_defined_later(x) + sq_defined_later(y);
}

// CHECK: void fn18_grad(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r0 = 0;
// CHECK-NEXT:         sq_defined_later_pullback(x, 1, &_r0);
// CHECK-NEXT:         *_d_x += _r0;
// CHECK-NEXT:         double _r1 = 0;
// CHECK-NEXT:         sq_defined_later_pullback(y, 1, &_r1);
// CHECK-NEXT:         *_d_y += _r1;
// CHECK-NEXT:     }
// CHECK-NEXT: }

template<typename T>
T templated_fn(double x) {
  static_assert(std::is_floating_point<T>::value,
                    "template argument must be a floating point type");
  return x;
}

// CHECK: void templated_fn_pullback(double x, double _d_y, double *_d_x);

double fn19(double x) {
  return templated_fn<double>(x);
}

// CHECK: void fn19_grad(double x, double *_d_x) {
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r0 = 0;
// CHECK-NEXT:         templated_fn_pullback(x, 1, &_r0);
// CHECK-NEXT:         *_d_x += _r0;
// CHECK-NEXT:     }
// CHECK-NEXT: }

double weighted_sum(double* x, const double* w) {
  return w[0] * x[0] + w[1] * x[1];
}

// CHECK: void weighted_sum_pullback(double *x, const double *w, double _d_y, double *_d_x);

double fn20(double* x, const double* w) {
  const double* auxW = w + 1;
  return weighted_sum(x, auxW);
}

// CHECK: void fn20_grad_0(double *x, const double *w, double *_d_x) {
// CHECK-NEXT:     const double *auxW = w + 1;
// CHECK-NEXT:     weighted_sum_pullback(x, auxW, 1, _d_x);
// CHECK-NEXT: }

double ptrRef(double*& ptr_ref) {
  return *ptr_ref;
}

// CHECK: void ptrRef_pullback(double *&ptr_ref, double _d_y, double **_d_ptr_ref);

double fn21(double x) {
  double* ptr = &x;
  return ptrRef(ptr);
}

// CHECK: void fn21_grad(double x, double *_d_x) {
// CHECK-NEXT:     double *_d_ptr = &*_d_x;
// CHECK-NEXT:     double *ptr = &x;
// CHECK-NEXT:     double *_t0 = ptr;
// CHECK-NEXT:     {
// CHECK-NEXT:         ptr = _t0;
// CHECK-NEXT:         ptrRef_pullback(_t0, 1, &_d_ptr);
// CHECK-NEXT:     }

namespace clad{
namespace custom_derivatives{
  void fn22_grad_1(double x, double y, double *d_y) {
    *d_y += x;
  }
}
}

double fn22(double x, double y) {
  return x*y; // fn22 has a custom derivative defined.
}

// CHECK: void fn22_grad_1(double x, double y, double *d_y) {
// CHECK-NEXT:   *d_y += x;
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
  F##_grad.execute(__VA_ARGS__, result, &d_n);\
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
  INIT(fn12);

  TEST1_float(fn1, 11);         // CHECK-EXEC: {3.00}
  TEST2(fn2, 3, 5);             // CHECK-EXEC: {1.00, 3.00}
  TEST2(fn3, 3, 5);             // CHECK-EXEC: {1.00, 3.00}
  double arr[5] = {1, 2, 3, 4, 5};
  TEST_ARR5(fn4, arr, 5);       // CHECK-EXEC: {23.00, 3.00, 3.00, 3.00, 3.00}
  TEST_ARR5(fn5, arr, 5);       // CHECK-EXEC: {5.00, 1.00, 0.00, 0.00, 0.00}
  TEST2(fn6, 3, 5);             // CHECK-EXEC: {5.00, 3.00}
  TEST2(fn7, 3, 5);             // CHECK-EXEC: {11.00, 78.00}
  TEST2(fn8, 3, 5);             // CHECK-EXEC: {7.62, 4.57}
  TEST2(fn9, 3, 5);             // CHECK-EXEC: {5.00, 3.00}
  TEST2(fn10, 8, 5);            // CHECK-EXEC: {0.00, 7.00}
  TEST2(fn11, 3, 5);            // CHECK-EXEC: {1.00, 1.00}
  TEST2(fn12, 3, 5);            // CHECK-EXEC: {1.00, 0.00}

  // Testing the partial gradient of a function with multiple pointer arguments
  auto fn13_grad_0 = clad::gradient(fn13, "x");
  double x = 2.0;
  double w[] = {2.0, 3.0};
  double fn13_result = 0.0;
  fn13_grad_0.execute(&x, w, &fn13_result);
  printf("{%.2f}\n", fn13_result);   // CHECK-EXEC: {3.00}

  INIT(fn14);
  TEST2(fn14, 3, 5);  // CHECK-EXEC: {1.00, 1.00}
  INIT(fn15);
  TEST2(fn15, 6, -2)  // CHECK-EXEC: {1.00, 1.00}
  INIT(fn16);
  TEST2(fn16, 12, 8)  // CHECK-EXEC: {8.00, 8.00}

  auto fn17_grad_0 = clad::gradient(fn17, "x");
  double y[] = {3.0, 2.0}, dx = 0;
  fn17_grad_0.execute(5, y, &dx);
  printf("{%.2f}\n", dx);   // CHECK-EXEC: {2.00}

  INIT(fn18);
  TEST2(fn18, 3, 5);  // CHECK-EXEC: {6.00, 10.00}

  INIT(fn19);
  TEST1(fn19, 3);  // CHECK-EXEC: {1.00}

  auto fn20_grad_0 = clad::gradient(fn20, "x");
  double x1[] = {3.0, 5.0}, w1[] = {-1.0, 2.0, 3.0};
  double dx1[] = {0.0, 0.0};
  fn20_grad_0.execute(x1, w1, dx1);
  printf("{%.2f, %.2f}\n", dx1[0], dx1[1]);  // CHECK-EXEC: {2.00, 3.00}

  INIT(fn21);
  TEST1(fn21, 8);  // CHECK-EXEC: {1.00}

  auto fn22_grad_1 = clad::gradient(fn22, "y");
  double dy = 0;
  fn22_grad_1.execute(3, 5, &dy);
  printf("{%.2f}\n", dy);  // CHECK-EXEC: {3.00}
}

double sq_defined_later(double x) {
    return x*x;
}

// CHECK: void constantFn_pullback(float i, float _d_y, float *_d_i) {
// CHECK-NEXT: }

// CHECK: void modify1_pullback(double &i, double &j, double _d_y, double *_d_i, double *_d_j) {
// CHECK-NEXT:     double _t0 = i;
// CHECK-NEXT:     i += j;
// CHECK-NEXT:     double _t1 = j;
// CHECK-NEXT:     j += j;
// CHECK-NEXT:     double _d_res = 0;
// CHECK-NEXT:     double res = i + j;
// CHECK-NEXT:     _d_res += _d_y;
// CHECK-NEXT:     {
// CHECK-NEXT:         *_d_i += _d_res;
// CHECK-NEXT:         *_d_j += _d_res;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         j = _t1;
// CHECK-NEXT:         double _r_d1 = *_d_j;
// CHECK-NEXT:         *_d_j += _r_d1;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         i = _t0;
// CHECK-NEXT:         double _r_d0 = *_d_i;
// CHECK-NEXT:         *_d_j += _r_d0;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: void update1_pullback(double &i, double &j, double *_d_i, double *_d_j) {
// CHECK-NEXT:     double _t0 = i;
// CHECK-NEXT:     i += j;
// CHECK-NEXT:     double _t1 = j;
// CHECK-NEXT:     j += j;
// CHECK-NEXT:     {
// CHECK-NEXT:         j = _t1;
// CHECK-NEXT:         double _r_d1 = *_d_j;
// CHECK-NEXT:         *_d_j += _r_d1;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         i = _t0;
// CHECK-NEXT:         double _r_d0 = *_d_i;
// CHECK-NEXT:         *_d_j += _r_d0;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: void sum_pullback(double *arr, int n, float _d_y, double *_d_arr, int *_d_n) {
// CHECK-NEXT:     int _d_i = 0;
// CHECK-NEXT:     int i = 0;
// CHECK-NEXT:     clad::tape<float> _t1 = {};
// CHECK-NEXT:     float _d_res = 0;
// CHECK-NEXT:     float res = 0;
// CHECK-NEXT:     unsigned {{int|long}} _t0 = {{0U|0UL}};
// CHECK-NEXT:     for (i = 0; ; ++i) {
// CHECK-NEXT:         {
// CHECK-NEXT:             if (!(i < n))
// CHECK-NEXT:                 break;
// CHECK-NEXT:         }
// CHECK-NEXT:         _t0++;
// CHECK-NEXT:         clad::push(_t1, res);
// CHECK-NEXT:         res += arr[i];
// CHECK-NEXT:     }
// CHECK-NEXT:     double _t2 = arr[0];
// CHECK-NEXT:     arr[0] += 10 * arr[0];
// CHECK-NEXT:     _d_res += _d_y;
// CHECK-NEXT:     {
// CHECK-NEXT:         arr[0] = _t2;
// CHECK-NEXT:         double _r_d1 = _d_arr[0];
// CHECK-NEXT:         _d_arr[0] += 10 * _r_d1;
// CHECK-NEXT:     }
// CHECK-NEXT:     for (;; _t0--) {
// CHECK-NEXT:         {
// CHECK-NEXT:             if (!_t0)
// CHECK-NEXT:                 break;
// CHECK-NEXT:         }
// CHECK-NEXT:         --i;
// CHECK-NEXT:         res = clad::pop(_t1);
// CHECK-NEXT:         float _r_d0 = _d_res;
// CHECK-NEXT:         _d_arr[i] += _r_d0;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: void twice_pullback(double &d, double *_d_d) {
// CHECK-NEXT:     double _t0 = d;
// CHECK-NEXT:     d = 2 * d;
// CHECK-NEXT:     {
// CHECK-NEXT:         d = _t0;
// CHECK-NEXT:         double _r_d0 = *_d_d;
// CHECK-NEXT:         *_d_d = 0;
// CHECK-NEXT:         *_d_d += 2 * _r_d0;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: void modify2_pullback(double *arr, double _d_y, double *_d_arr) {
// CHECK-NEXT:     double _t0 = arr[0];
// CHECK-NEXT:     arr[0] = 5 * arr[0] + arr[1];
// CHECK-NEXT:     {
// CHECK-NEXT:         arr[0] = _t0;
// CHECK-NEXT:         double _r_d0 = _d_arr[0];
// CHECK-NEXT:         _d_arr[0] = 0;
// CHECK-NEXT:         _d_arr[0] += 5 * _r_d0;
// CHECK-NEXT:         _d_arr[1] += _r_d0;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: void identity_pullback(double &i, double _d_y, double *_d_i) {
// CHECK-NEXT:     MyStruct::myFunction();
// CHECK-NEXT:     double _d__d_i = 0;
// CHECK-NEXT:     double _d_i0 = i;
// CHECK-NEXT:     double _t0 = _d_i0;
// CHECK-NEXT:     _d_i0 += 1;
// CHECK-NEXT:     *_d_i += _d_y;
// CHECK-NEXT:     {
// CHECK-NEXT:         _d_i0 = _t0;
// CHECK-NEXT:         double _r_d0 = _d__d_i;
// CHECK-NEXT:     }
// CHECK-NEXT:     *_d_i += _d__d_i;
// CHECK-NEXT: }

// CHECK: clad::ValueAndAdjoint<double &, double &> identity_forw(double &i, double *_d_i) {
// CHECK-NEXT:     MyStruct::myFunction();
// CHECK-NEXT:     double _d__d_i = 0;
// CHECK-NEXT:     double _d_i0 = i;
// CHECK-NEXT:     double _t0 = _d_i0;
// CHECK-NEXT:     _d_i0 += 1;
// CHECK-NEXT:     return {i, *_d_i};
// CHECK-NEXT: }

// CHECK: void custom_identity_pullback(double &i, double _d_y, double *_d_i) {
// CHECK-NEXT:     *_d_i += _d_y; 
// CHECK-NEXT: }

// CHECK: void check_and_return_pullback(double x, char c, const char *s, double _d_y, double *_d_x, char *_d_c, char *_d_s) {
// CHECK-NEXT:    bool _cond0;
// CHECK-NEXT:    double _d_cond0;
// CHECK-NEXT:    _d_cond0 = 0.;
// CHECK-NEXT:    bool _cond1;
// CHECK-NEXT:    bool _t0;
// CHECK-NEXT:    bool _cond2;
// CHECK-NEXT:    {
// CHECK-NEXT:        {
// CHECK-NEXT:            _cond1 = c == 'a';
// CHECK-NEXT:            if (_cond1) {
// CHECK-NEXT:                _t0 = _cond0;
// CHECK-NEXT:                _cond0 = s[0] == 'a';
// CHECK-NEXT:            }
// CHECK-NEXT:        }
// CHECK-NEXT:        _cond2 = _cond1 && _cond0;
// CHECK-NEXT:        if (_cond2)
// CHECK-NEXT:            goto _label0;
// CHECK-NEXT:    }
// CHECK-NEXT:    {
// CHECK-NEXT:        if (_cond2)
// CHECK-NEXT:          _label0:
// CHECK-NEXT:           *_d_x += _d_y;
// CHECK-NEXT:        {
// CHECK-NEXT:            if (_cond1) {
// CHECK-NEXT:                _cond0 = _t0;
// CHECK-NEXT:                double _r_d0 = _d_cond0;
// CHECK-NEXT:                _d_cond0 = 0;
// CHECK-NEXT:            }
// CHECK-NEXT:        }
// CHECK-NEXT:    }
// CHECK-NEXT:}

// CHECK: void custom_max_pullback(const double &a, const double &b, double _d_y, double *_d_a, double *_d_b) {
// CHECK-NEXT:     bool _cond0 = a > b;
// CHECK-NEXT:     if (_cond0)
// CHECK-NEXT:         *_d_a += _d_y;
// CHECK-NEXT:     else
// CHECK-NEXT:         *_d_b += _d_y;
// CHECK-NEXT: }

// CHECK: void do_nothing_pullback(double *u, double *v, double *w, double _d_y, double *_d_u, double *_d_v, double *_d_w) {
// CHECK-NEXT:     _d_u[0] += _d_y;
// CHECK-NEXT: }

// CHECK: void multiply_pullback(double *a, double *b, double _d_y, double *_d_a, double *_d_b) {
// CHECK-NEXT:     {
// CHECK-NEXT:         _d_a[0] += _d_y * b[0];
// CHECK-NEXT:         _d_b[0] += a[0] * _d_y;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: void emptyFn_pullback(double &x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT: }

//CHECK: void recFun_pullback(double x, double y, double _d_y0, double *_d_x, double *_d_y) {
//CHECK-NEXT:     bool _cond0;
//CHECK-NEXT:     {
//CHECK-NEXT:     _cond0 = x > y;
//CHECK-NEXT:     if (_cond0)
//CHECK-NEXT:         goto _label0;
//CHECK-NEXT:     }
//CHECK-NEXT:     {
//CHECK-NEXT:         *_d_x += _d_y0 * y;
//CHECK-NEXT:         *_d_y += x * _d_y0;
//CHECK-NEXT:     }
//CHECK-NEXT:     if (_cond0)
//CHECK-NEXT:       _label0:
//CHECK-NEXT:         {
//CHECK-NEXT:             double _r0 = 0;
//CHECK-NEXT:             double _r1 = 0;
//CHECK-NEXT:             recFun_pullback(x - 1, y, _d_y0, &_r0, &_r1);
//CHECK-NEXT:             *_d_x += _r0;
//CHECK-NEXT:             *_d_y += _r1;
//CHECK-NEXT:         }
//CHECK-NEXT: }

//CHECK: void add_pullback(double a, double *b, double _d_y, double *_d_a) {
//CHECK-NEXT:     *_d_a += _d_y;
//CHECK-NEXT: }

//CHECK: void add_pullback(double a, double *b, double _d_y, double *_d_a, double *_d_b) {
//CHECK-NEXT:     {
//CHECK-NEXT:         *_d_a += _d_y;
//CHECK-NEXT:         _d_b[0] += _d_y;
//CHECK-NEXT:     }
//CHECK-NEXT: }

// CHECK: void sq_defined_later_pullback(double x, double _d_y, double *_d_x) {
// CHECK-NEXT:     {
// CHECK-NEXT:       *_d_x += _d_y * x;
// CHECK-NEXT:       *_d_x += x * _d_y;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: void templated_fn_pullback(double x, double _d_y, double *_d_x) {
// CHECK-NEXT:     *_d_x += _d_y;
// CHECK-NEXT: }

// CHECK: void weighted_sum_pullback(double *x, const double *w, double _d_y, double *_d_x) {
// CHECK-NEXT:     {
// CHECK-NEXT:         _d_x[0] += w[0] * _d_y;
// CHECK-NEXT:         _d_x[1] += w[1] * _d_y;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: void ptrRef_pullback(double *&ptr_ref, double _d_y, double **_d_ptr_ref) {
// CHECK-NEXT:     **_d_ptr_ref += _d_y;
// CHECK-NEXT: }
