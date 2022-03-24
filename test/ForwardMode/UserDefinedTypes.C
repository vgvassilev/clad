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

// CHECK: double fn6_darg1(TensorD5 t, double i) {
// CHECK-NEXT:     TensorD5 _d_t;
// CHECK-NEXT:     double _d_i = 1;
// CHECK-NEXT:     double _t0 = t.sum();
// CHECK-NEXT:     double _d_res = 0 * _t0 + 3 * t.sum_pushforward(&_d_t);
// CHECK-NEXT:     double res = 3 * _t0;
// CHECK-NEXT:     t.updateTo_pushforward(i * i, &_d_t, _d_i * i + i * _d_i);
// CHECK-NEXT:     t.updateTo(i * i);
// CHECK-NEXT:     _d_res += sum_pushforward(t, _d_t);
// CHECK-NEXT:     res += sum(t);
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
// CHECK-NEXT:     t.updateTo(_t0 * j);
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
// CHECK-NEXT:     t.updateTo(i * i);
// CHECK-NEXT:     complexD _d_c(0., 0.);
// CHECK-NEXT:     complexD c(0., 0.);
// CHECK-NEXT:     double _t0 = t.sum();
// CHECK-NEXT:     c.real_pushforward(7 * _t0, &_d_c, 0 * _t0 + 7 * t.sum_pushforward(&_d_t));
// CHECK-NEXT:     c.real(7 * _t0);
// CHECK-NEXT:     double _t1 = sum(t);
// CHECK-NEXT:     c.imag_pushforward(9 * _t1, &_d_c, 0 * _t1 + 9 * sum_pushforward(t, _d_t));
// CHECK-NEXT:     c.imag(9 * _t1);
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

// CHECK: constexpr double real_pushforward(const std{{(::__1)?}}::complex<double> *_d_this){{.*}} {
// CHECK-NEXT:     return{{( __real)?}} _d_this->[[_M_value:.*]];
// CHECK-NEXT: }

// CHECK: constexpr double imag_pushforward(const std{{(::__1)?}}::complex<double> *_d_this){{.*}} {
// CHECK-NEXT:     return{{( __imag)?}} _d_this->[[_M_value:.*]];
// CHECK-NEXT: }

// CHECK: complexD fn9_darg0(double i, complexD c) {
// CHECK-NEXT:     double _d_i = 1;
// CHECK-NEXT:     complexD _d_c;
// CHECK-NEXT:     complexD _d_r(0., 0.);
// CHECK-NEXT:     complexD r(0., 0.);
// CHECK-NEXT:     c.real_pushforward(i * i, &_d_c, _d_i * i + i * _d_i);
// CHECK-NEXT:     c.real(i * i);
// CHECK-NEXT:     double _t0 = c.real();
// CHECK-NEXT:     double _t1 = c.real();
// CHECK-NEXT:     c.imag_pushforward(_t0 * _t1, &_d_c, c.real_pushforward(&_d_c) * _t1 + _t0 * c.real_pushforward(&_d_c));
// CHECK-NEXT:     c.imag(_t0 * _t1);
// CHECK-NEXT:     double _t2 = c.real();
// CHECK-NEXT:     r.real_pushforward(i * _t2, &_d_r, _d_i * _t2 + i * c.real_pushforward(&_d_c));
// CHECK-NEXT:     r.real(i * _t2);
// CHECK-NEXT:     double _t3 = c.real();
// CHECK-NEXT:     double _t4 = c.imag();
// CHECK-NEXT:     r.imag_pushforward(_t3 * _t4, &_d_r, c.real_pushforward(&_d_c) * _t4 + _t3 * c.imag_pushforward(&_d_c));
// CHECK-NEXT:     r.imag(_t3 * _t4);
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