// RUN: %cladclang %s -I%S/../../include -oUserDefinedTypes.out 2>&1 | FileCheck %s
// RUN: ./UserDefinedTypes.out | FileCheck -check-prefix=CHECK-EXEC %s

// CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

#include <complex>

#include "../TestUtils.h"
#include "../PrintOverloads.h"

std::pair<double, double> fn1(double i, double j) {
  std::pair<double, double> c(3, 5), d({7.00, 9.00});
  std::pair<double, double> e = d;
  c.first += i;
  c.second += 2 * i + j;
  d.first += 2 * i;
  d.second += i;
  c.first += d.first;
  c.second += d.second;
  return c;
}

// CHECK: std::pair<double, double> fn1_darg0(double i, double j) {
// CHECK-NEXT:     double _d_i = 1;
// CHECK-NEXT:     double _d_j = 0;
// CHECK-NEXT:     std::pair<double, double> _d_c(0, 0), _d_d({0., 0.});
// CHECK-NEXT:     std::pair<double, double> c(3, 5), d({7., 9.});
// CHECK-NEXT:     std::pair<double, double> _d_e = _d_d;
// CHECK-NEXT:     std::pair<double, double> e = d;
// CHECK-NEXT:     _d_c.first += _d_i;
// CHECK-NEXT:     c.first += i;
// CHECK-NEXT:     _d_c.second += 0 * i + 2 * _d_i + _d_j;
// CHECK-NEXT:     c.second += 2 * i + j;
// CHECK-NEXT:     _d_d.first += 0 * i + 2 * _d_i;
// CHECK-NEXT:     d.first += 2 * i;
// CHECK-NEXT:     _d_d.second += _d_i;
// CHECK-NEXT:     d.second += i;
// CHECK-NEXT:     _d_c.first += _d_d.first;
// CHECK-NEXT:     c.first += d.first;
// CHECK-NEXT:     _d_c.second += _d_d.second;
// CHECK-NEXT:     c.second += d.second;
// CHECK-NEXT:     return _d_c;
// CHECK-NEXT: }

std::pair<double, double> fn2(double i, double j) {
  std::pair<double, double> c, d;
  std::pair<double, double> e = d;
  return std::pair<double, double>();
}

// CHECK: std::pair<double, double> fn2_darg0(double i, double j) {
// CHECK-NEXT:     double _d_i = 1;
// CHECK-NEXT:     double _d_j = 0;
// CHECK-NEXT:     std::pair<double, double> _d_c, _d_d;
// CHECK-NEXT:     std::pair<double, double> c, d;
// CHECK-NEXT:     std::pair<double, double> _d_e = _d_d;
// CHECK-NEXT:     std::pair<double, double> e = d;
// CHECK-NEXT:     return std::pair<double, double>();
// CHECK-NEXT: }

std::pair<double, double> fn3(double i, double j) {
  std::pair<double, double> c, d;
  std::pair<double, double> e = d;
  return {};
}

// CHECK: std::pair<double, double> fn3_darg0(double i, double j) {
// CHECK-NEXT:     double _d_i = 1;
// CHECK-NEXT:     double _d_j = 0;
// CHECK-NEXT:     std::pair<double, double> _d_c, _d_d;
// CHECK-NEXT:     std::pair<double, double> c, d;
// CHECK-NEXT:     std::pair<double, double> _d_e = _d_d;
// CHECK-NEXT:     std::pair<double, double> e = d;
// CHECK-NEXT:     return {};
// CHECK-NEXT: }

std::pair<double, double> fn4(double i, double j) {
  std::pair<double, double> c, d;
  std::pair<double, double> e = d;
  return std::pair<double, double>(1, 3);
}

// CHECK: std::pair<double, double> fn4_darg0(double i, double j) {
// CHECK-NEXT:     double _d_i = 1;
// CHECK-NEXT:     double _d_j = 0;
// CHECK-NEXT:     std::pair<double, double> _d_c, _d_d;
// CHECK-NEXT:     std::pair<double, double> c, d;
// CHECK-NEXT:     std::pair<double, double> _d_e = _d_d;
// CHECK-NEXT:     std::pair<double, double> e = d;
// CHECK-NEXT:     return std::pair<double, double>(0, 0);
// CHECK-NEXT: }

template<typename T, unsigned N>
struct Tensor {
  T data[N] = {};
  void updateTo(T val) {
    for (int i=0; i<N; ++i)
      data[i] = val;
  }
  T sum() {
    T res = 0;
    for (int i=0; i<N; ++i)
      res += data[i];
    return res;
  }
};

template<typename T, unsigned N>
T sum(Tensor<T, N>& t) {
  T res = 0;
  for (int i=0; i<N; ++i)
    res += t.data[i];
  return res;
}

Tensor<double, 5> fn5(double i, double j) {
  Tensor<double, 5> T;
  for (int l=0; l<5; ++l) {
    T.data[l] = (l+1)*i;
  }
  return T;
} 

// CHECK: Tensor<double, 5> fn5_darg0(double i, double j) {
// CHECK-NEXT:     double _d_i = 1;
// CHECK-NEXT:     double _d_j = 0;
// CHECK-NEXT:     Tensor<double, 5> _d_T;
// CHECK-NEXT:     Tensor<double, 5> T;
// CHECK-NEXT:     {
// CHECK-NEXT:         int _d_l = 0;
// CHECK-NEXT:         for (int l = 0; l < 5; ++l) {
// CHECK-NEXT:             int _t0 = (l + 1);
// CHECK-NEXT:             _d_T.data[l] = (_d_l + 0) * i + _t0 * _d_i;
// CHECK-NEXT:             T.data[l] = _t0 * i;
// CHECK-NEXT:         }
// CHECK-NEXT:     }
// CHECK-NEXT:     return _d_T;
// CHECK-NEXT: }

using TensorD5 = Tensor<double, 5>;

double fn6(TensorD5 t, double i) {
  double res = 3*t.sum();
  t.updateTo(i*i);
  res += sum(t);
  return res;
}
// CHECK: clad::ValueAndPushforward<double, double> sum_pushforward(Tensor<double, 5> *_d_this) {
// CHECK-NEXT:     double _d_res = 0;
// CHECK-NEXT:     double res = 0;
// CHECK-NEXT:     {
// CHECK-NEXT:         int _d_i = 0;
// CHECK-NEXT:         for (int i = 0; i < 5U; ++i) {
// CHECK-NEXT:             _d_res += _d_this->data[i];
// CHECK-NEXT:             res += this->data[i];
// CHECK-NEXT:         }
// CHECK-NEXT:     }
// CHECK-NEXT:     return {res, _d_res};
// CHECK-NEXT: }

// CHECK: void updateTo_pushforward(double val, Tensor<double, 5> *_d_this, double _d_val) {
// CHECK-NEXT:     {
// CHECK-NEXT:         int _d_i = 0;
// CHECK-NEXT:         for (int i = 0; i < 5U; ++i) {
// CHECK-NEXT:             _d_this->data[i] = _d_val;
// CHECK-NEXT:             this->data[i] = val;
// CHECK-NEXT:         }
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: clad::ValueAndPushforward<double, double> sum_pushforward(Tensor<double, 5U> &t, Tensor<double, 5U> &_d_t) {
// CHECK-NEXT:     double _d_res = 0;
// CHECK-NEXT:     double res = 0;
// CHECK-NEXT:     {
// CHECK-NEXT:         int _d_i = 0;
// CHECK-NEXT:         for (int i = 0; i < 5U; ++i) {
// CHECK-NEXT:             _d_res += _d_t.data[i];
// CHECK-NEXT:             res += t.data[i];
// CHECK-NEXT:         }
// CHECK-NEXT:     }
// CHECK-NEXT:     return {res, _d_res};
// CHECK-NEXT: }

// CHECK: double fn6_darg1(TensorD5 t, double i) {
// CHECK-NEXT:     TensorD5 _d_t;
// CHECK-NEXT:     double _d_i = 1;
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t0 = t.sum_pushforward(&_d_t);
// CHECK-NEXT:     double &_t1 = _t0.value;
// CHECK-NEXT:     double _d_res = 0 * _t1 + 3 * _t0.pushforward;
// CHECK-NEXT:     double res = 3 * _t1;
// CHECK-NEXT:     t.updateTo_pushforward(i * i, &_d_t, _d_i * i + i * _d_i);
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t2 = sum_pushforward(t, _d_t);
// CHECK-NEXT:     _d_res += _t2.pushforward;
// CHECK-NEXT:     res += _t2.value;
// CHECK-NEXT:     return _d_res;
// CHECK-NEXT: }

TensorD5 fn7(double i, double j) {
  TensorD5 t;
  t.updateTo(7*i*j);
  return t;
}

// CHECK: TensorD5 fn7_darg0(double i, double j) {
// CHECK-NEXT:     double _d_i = 1;
// CHECK-NEXT:     double _d_j = 0;
// CHECK-NEXT:     TensorD5 _d_t;
// CHECK-NEXT:     TensorD5 t;
// CHECK-NEXT:     double _t0 = 7 * i;
// CHECK-NEXT:     t.updateTo_pushforward(_t0 * j, &_d_t, (0 * i + 7 * _d_i) * j + _t0 * _d_j);
// CHECK-NEXT:     return _d_t;
// CHECK-NEXT: }

using complexD = std::complex<double>;

complexD fn8(double i, TensorD5 t) {
  t.updateTo(i*i);
  complexD c;
  c.real(7*t.sum());
  c.imag(9*sum(t));
  return c;
}

// CHECK: void real_pushforward({{.*}} [[__val:.*]], std{{(::__1)?}}::complex<double> *_d_this, {{.*}} [[_d___val:[a-zA-Z_]*]]){{.*}} {
// CHECK-NEXT:     {{(__real)?}} _d_this->[[_M_value:.*]] = [[_d___val]];
// CHECK-NEXT:     {{(__real)?}} this->[[_M_value]] = [[__val]];
// CHECK-NEXT: }

// CHECK: void imag_pushforward({{.*}} [[__val:.*]], std{{(::__1)?}}::complex<double> *_d_this, {{.*}} [[_d___val:[a-zA-Z_]*]]){{.*}} {
// CHECK-NEXT:     {{(__imag)?}} _d_this->[[_M_value:.*]] = [[_d___val]];
// CHECK-NEXT:     {{(__imag)?}} this->[[_M_value]] = [[__val]];
// CHECK-NEXT: }


// CHECK: complexD fn8_darg0(double i, TensorD5 t) {
// CHECK-NEXT:     double _d_i = 1;
// CHECK-NEXT:     TensorD5 _d_t;
// CHECK-NEXT:     t.updateTo_pushforward(i * i, &_d_t, _d_i * i + i * _d_i);
// CHECK-NEXT:     complexD _d_c(0., 0.);
// CHECK-NEXT:     complexD c(0., 0.);
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t0 = t.sum_pushforward(&_d_t);
// CHECK-NEXT:     double &_t1 = _t0.value;
// CHECK-NEXT:     c.real_pushforward(7 * _t1, &_d_c, 0 * _t1 + 7 * _t0.pushforward);
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t2 = sum_pushforward(t, _d_t);
// CHECK-NEXT:     double &_t3 = _t2.value;
// CHECK-NEXT:     c.imag_pushforward(9 * _t3, &_d_c, 0 * _t3 + 9 * _t2.pushforward);
// CHECK-NEXT:     return _d_c;
// CHECK-NEXT: }

complexD fn9(double i, complexD c) {
  complexD r;
  c.real(i*i);
  c.imag(c.real()*c.real());
  r.real(i*c.real());
  r.imag(c.real()*c.imag());
  return r;
}

// CHECK: constexpr clad::ValueAndPushforward<double, double> real_pushforward(const std{{(::__1)?}}::complex<double> *_d_this){{.*}} {
// CHECK-NEXT:     return {{[{](__real )?}}this->[[_M_value:[a-zA-Z_]+]],{{( __real)?}} _d_this->[[_M_value:[a-zA-Z_]+]]};
// CHECK-NEXT: }

// CHECK: constexpr clad::ValueAndPushforward<double, double> imag_pushforward(const std{{(::__1)?}}::complex<double> *_d_this){{.*}} {
// CHECK-NEXT:     return {{[{](__imag )?}}this->[[_M_value:[a-zA-Z_]+]],{{( __imag)?}} _d_this->[[_M_value:[a-zA-Z_]+]]};
// CHECK-NEXT: }

// CHECK: complexD fn9_darg0(double i, complexD c) {
// CHECK-NEXT:     double _d_i = 1;
// CHECK-NEXT:     complexD _d_c;
// CHECK-NEXT:     complexD _d_r(0., 0.);
// CHECK-NEXT:     complexD r(0., 0.);
// CHECK-NEXT:     c.real_pushforward(i * i, &_d_c, _d_i * i + i * _d_i);
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t0 = c.real_pushforward(&_d_c);
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t1 = c.real_pushforward(&_d_c);
// CHECK-NEXT:     double &_t2 = _t0.value;
// CHECK-NEXT:     double &_t3 = _t1.value;
// CHECK-NEXT:     c.imag_pushforward(_t2 * _t3, &_d_c, _t0.pushforward * _t3 + _t2 * _t1.pushforward);
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t4 = c.real_pushforward(&_d_c);
// CHECK-NEXT:     double &_t5 = _t4.value;
// CHECK-NEXT:     r.real_pushforward(i * _t5, &_d_r, _d_i * _t5 + i * _t4.pushforward);
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t6 = c.real_pushforward(&_d_c);
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t7 = c.imag_pushforward(&_d_c);
// CHECK-NEXT:     double &_t8 = _t6.value;
// CHECK-NEXT:     double &_t9 = _t7.value;
// CHECK-NEXT:     r.imag_pushforward(_t8 * _t9, &_d_r, _t6.pushforward * _t9 + _t8 * _t7.pushforward);
// CHECK-NEXT:     return _d_r;
// CHECK-NEXT: }

template<unsigned N>
void print(const Tensor<double, N>& t) {
  for (int i=0; i<N; ++i) {
    test_utils::print(t.data[i]);
    if (i != N-1)
      printf(", ");
  }
}

int main() {
  INIT_DIFFERENTIATE(fn1, "i");
  INIT_DIFFERENTIATE(fn2, "i");
  INIT_DIFFERENTIATE(fn3, "i");
  INIT_DIFFERENTIATE(fn4, "i");
  INIT_DIFFERENTIATE(fn5, "i");
  INIT_DIFFERENTIATE(fn6, "i");
  INIT_DIFFERENTIATE(fn7, "i");
  INIT_DIFFERENTIATE(fn8, "i");
  INIT_DIFFERENTIATE(fn9, "i");

  TensorD5 t;
  t.updateTo(5);
  complexD c(3, 5);

  TEST_DIFFERENTIATE(fn1, 3, 5);  // CHECK-EXEC: {3.00, 3.00}
  TEST_DIFFERENTIATE(fn2, 3, 5);  // CHECK-EXEC: {0.00, 0.00}
  TEST_DIFFERENTIATE(fn3, 3, 5);  // CHECK-EXEC: {0.00, 0.00}
  TEST_DIFFERENTIATE(fn4, 3, 5);  // CHECK-EXEC: {0.00, 0.00}
  TEST_DIFFERENTIATE(fn5, 3, 5);  // CHECK-EXEC: {1.00, 2.00, 3.00, 4.00, 5.00}
  TEST_DIFFERENTIATE(fn6, t, 3);  // CHECK-EXEC: {30.00}
  TEST_DIFFERENTIATE(fn7, 3, 5);  // CHECK-EXEC: {35.00, 35.00, 35.00, 35.00, 35.00}
  TEST_DIFFERENTIATE(fn8, 3, t);  // CHECK-EXEC: {210.00, 270.00}
  TEST_DIFFERENTIATE(fn9, 3, c);  // CHECK-EXEC: {27.00, 1458.00}
}