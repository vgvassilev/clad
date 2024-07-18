// RUN: %cladclang %s -I%S/../../include -oUserDefinedTypes.out | %filecheck %s
// RUN: ./UserDefinedTypes.out | %filecheck_exec %s

// XFAIL: asserts

// CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"
#include "clad/Differentiator/STLBuiltins.h"

#include <complex>
#include <numeric>
#include <vector>

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
// CHECK-NEXT:     std::pair<double, double> _d_c({0, 0}), _d_d({0., 0.});
// CHECK-NEXT:     std::pair<double, double> c({3, 5}), d({7., 9.});
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

  Tensor() : data() {}
  
  ~Tensor() {}

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

  Tensor& operator=(const Tensor& t) {
    for (unsigned i=0; i<N; ++i)
      data[i] = t.data[i];
    return *this;
  }

  T& operator[](std::size_t idx) {
    return data[idx];
  }

  const T& operator[](std::size_t idx) const {
    return data[idx];
  }

  void operator()(T val) {
    for (unsigned i=0; i<N; ++i)
      data[i] = val;
  }
  
  Tensor& operator++() {
    for (unsigned i=0; i<N; ++i)
      data[i]+=1;
    return *this;
  }
  
  Tensor operator++(int) {
    Tensor temp;
    for (unsigned i=0; i<N; ++i)
      temp.data[i] += 1;
    return temp;
  }
  
  Tensor& operator--() {
    for (unsigned i=0; i<N; ++i)
      data[i] += 1;
    return *this;
  }

  Tensor operator--(int) {
    Tensor temp;
    for (unsigned i=0; i<N; ++i)
      temp.data[i] -= 1;
    return temp;
  }

  Tensor* operator->() { return this; }
  Tensor* operator&() { return this; }
};

template<typename T, unsigned N>
Tensor<T, N> operator-(const Tensor<T, N>& t) {
  Tensor<T, N> res;
  for (unsigned i=0; i<N; ++i)
    res.data[i] = -t.data[i];
  return res;
};

template<typename T, unsigned N>
Tensor<T, N> operator+(const Tensor<T, N>& t) {
  Tensor<T, N> res;
  for (unsigned i=0; i<N; ++i)
    res.data[i] = +t.data[i];
  return res;
};

template<typename T, unsigned N>
Tensor<T, N> operator+(const Tensor<T, N>& a, const Tensor<T, N>& b) {
  Tensor<T, N> res;
  for (unsigned i=0; i<N; ++i)
    res.data[i] = a.data[i] + b.data[i];
  return res;
};

template<typename T, unsigned N>
Tensor<T, N> operator-(const Tensor<T, N>& a, const Tensor<T, N>& b) {
  Tensor<T, N> res;
  for (unsigned i=0; i<N; ++i)
    res.data[i] = a.data[i] - b.data[i];
  return res;
};

template<typename T, unsigned N>
Tensor<T, N> operator*(const Tensor<T, N>& a, const Tensor<T, N>& b) {
  Tensor<T, N> res;
  for (unsigned i=0; i<N; ++i)
    res.data[i] = a.data[i] * b.data[i];
  return res;
};

template<typename T, unsigned N>
Tensor<T, N> operator*(double val, const Tensor<T, N>& b) {
  Tensor<T, N> res;
  for (unsigned i=0; i<N; ++i)
    res.data[i] = val * b.data[i];
  return res;
};

template<typename T, unsigned N>
Tensor<T, N> operator/(const Tensor<T, N>& a, const Tensor<T, N>& b) {
  Tensor<T, N> res;
  for (unsigned i=0; i<N; ++i)
    res.data[i] = a.data[i] / b.data[i];
  return res;
};

template<typename T, unsigned N>
Tensor<T, N> operator^(const Tensor<T, N>& a, const Tensor<T, N>& b) {
  Tensor<T, N> res;
  for (unsigned i=0; i<N; ++i)
    res.data[i] = std::pow(a.data[i], b.data[i]);
  return res;
};

template<typename T, unsigned N>
Tensor<T, N>& operator+=(Tensor<T, N>& lhs, const Tensor<T, N>& rhs) {
  lhs = lhs + rhs;
  return lhs;
}

template<typename T, unsigned N>
Tensor<T, N>& operator-=(Tensor<T, N>& lhs, const Tensor<T, N>& rhs) {
  lhs = lhs - rhs;
  return lhs;
}

template<typename T, unsigned N>
Tensor<T, N>& operator*=(Tensor<T, N>& lhs, const Tensor<T, N>& rhs) {
  lhs = lhs * rhs;
  return lhs;
}

template<typename T, unsigned N>
Tensor<T, N>& operator/=(Tensor<T, N>& lhs, const Tensor<T, N>& rhs) {
  lhs = lhs / rhs;
  return lhs;
}

template<typename T, unsigned N>
Tensor<T, N>& operator^=(Tensor<T, N>& lhs, const Tensor<T, N>& rhs) {
  lhs = lhs / rhs;
  return lhs;
}

template<typename T, unsigned N>
Tensor<T, N> operator%(const Tensor<T, N>& a, const Tensor<T, N>& b) {
  return a;
};

template<typename T, unsigned N>
Tensor<T, N>& operator%=(Tensor<T, N>& a, const Tensor<T, N>& b) {
  return a;
};

template<typename T, unsigned N>
void operator~(const Tensor<T, N>& a) {};

template<typename T, unsigned N>
bool operator<(Tensor<T, N>& lhs, const Tensor<T, N>& rhs) {
  T lsum, rsum;
  lsum = rsum = 0;
  for (unsigned i=0; i<N; ++i) {
    lsum += lhs.data[i];
    rsum += lhs.data[i];
  }
  // FIXME: Add support for differentiating comparison operators.
  // return lsum < rsum;
  return 1;
}

template<typename T, unsigned N>
bool operator>(Tensor<T, N>& lhs, const Tensor<T, N>& rhs) {
  T lsum, rsum;
  lsum = rsum = 0;
  for (unsigned i=0; i<N; ++i) {
    lsum += lhs.data[i];
    rsum += lhs.data[i];
  }
  // FIXME: Add support for differentiating comparison operators.
  // return lsum > rsum;
  return 1;
}

template<typename T, unsigned N>
bool operator==(Tensor<T, N>& lhs, const Tensor<T, N>& rhs) {
  // FIXME: Add support for differentiating comparison operators.
  // return !(lhs < rhs) && !(lhs > rhs);
  return 1;
}

template<typename T, unsigned N>
bool operator<=(Tensor<T, N>& lhs, const Tensor<T, N>& rhs) {
  // FIXME: Add support for differentiating comparison operators.
  // return (lhs < rhs) || (lhs == rhs);
  return 1;
}

template<typename T, unsigned N>
bool operator>=(Tensor<T, N>& lhs, const Tensor<T, N>& rhs) {
  // FIXME: Add support for differentiating comparison operators.
  // return (lhs > rhs) || (lhs == rhs);
  return 1;
}

template<typename T, unsigned N>
bool operator!=(Tensor<T, N>& lhs, const Tensor<T, N>& rhs) {
  // FIXME: Add support for differentiating comparison operators.
  // return !(lhs == rhs);
  return 1;
}

template<typename T, unsigned N>
bool operator<<(Tensor<T, N>& lhs, const Tensor<T, N>& rhs) {
  // FIXME: Add support for differentiating comparison operators.
  // return !(lhs == rhs);
  return 1;
}

template<typename T, unsigned N>
bool operator>>(Tensor<T, N>& lhs, const Tensor<T, N>& rhs) {
  // FIXME: Add support for differentiating comparison operators.
  // return !(lhs == rhs);
  return 1;
}

template<typename T, unsigned N>
bool operator<<=(Tensor<T, N>& lhs, const Tensor<T, N>& rhs) {
  // FIXME: Add support for differentiating comparison operators.
  // return !(lhs == rhs);
  return 1;
}

template<typename T, unsigned N>
bool operator>>=(Tensor<T, N>& lhs, const Tensor<T, N>& rhs) {
  // FIXME: Add support for differentiating comparison operators.
  // return !(lhs == rhs);
  return 1;
}


template<typename T, unsigned N>
bool operator&&(Tensor<T, N>& lhs, const Tensor<T, N>& rhs) {
  // FIXME: Add support for differentiating comparison operators.
  // return !(lhs == rhs);
  return 1;
}

template<typename T, unsigned N>
bool operator||(Tensor<T, N>& lhs, const Tensor<T, N>& rhs) {
  // FIXME: Add support for differentiating comparison operators.
  // return !(lhs == rhs);
  return 1;
}

template<typename T, unsigned N>
bool operator&(Tensor<T, N>& lhs, const Tensor<T, N>& rhs) {
  // FIXME: Add support for differentiating comparison operators.
  // return !(lhs == rhs);
  return 1;
}

template<typename T, unsigned N>
bool operator|(Tensor<T, N>& lhs, const Tensor<T, N>& rhs) {
  // FIXME: Add support for differentiating comparison operators.
  // return !(lhs == rhs);
  return 1;
}

template<typename T, unsigned N>
bool operator&=(Tensor<T, N>& lhs, const Tensor<T, N>& rhs) {
  // FIXME: Add support for differentiating comparison operators.
  // return !(lhs == rhs);
  return 1;
}

template<typename T, unsigned N>
bool operator|=(Tensor<T, N>& lhs, const Tensor<T, N>& rhs) {
  // FIXME: Add support for differentiating comparison operators.
  // return !(lhs == rhs);
  return 1;
}

template<typename T, unsigned N>
void operator,(const Tensor<T, N>& lhs, const Tensor<T, N>& rhs) {}

template<typename T, unsigned N>
void operator!(const Tensor<T, N>& lhs) {}

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
// CHECK: clad::ValueAndPushforward<double, double> sum_pushforward(Tensor<double, 5> *_d_this);

// CHECK: void updateTo_pushforward(double val, Tensor<double, 5> *_d_this, double _d_val);

// CHECK: clad::ValueAndPushforward<double, double> sum_pushforward(Tensor<double, 5U> &t, Tensor<double, 5U> &_d_t);

// CHECK: double fn6_darg1(TensorD5 t, double i) {
// CHECK-NEXT:     TensorD5 _d_t;
// CHECK-NEXT:     double _d_i = 1;
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t0 = t.sum_pushforward(& _d_t);
// CHECK-NEXT:     double &_t1 = _t0.value;
// CHECK-NEXT:     double _d_res = 0 * _t1 + 3 * _t0.pushforward;
// CHECK-NEXT:     double res = 3 * _t1;
// CHECK-NEXT:     t.updateTo_pushforward(i * i, & _d_t, _d_i * i + i * _d_i);
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
// CHECK-NEXT:     t.updateTo_pushforward(_t0 * j, & _d_t, (0 * i + 7 * _d_i) * j + _t0 * _d_j);
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

// CHECK: void real_pushforward({{.*}} [[__val:.*]], std{{(::__1)?}}::complex<double> *_d_this, {{.*}} [[_d___val:[a-zA-Z_]*]]){{.*}};

// CHECK: void imag_pushforward({{.*}} [[__val:.*]], std{{(::__1)?}}::complex<double> *_d_this, {{.*}} [[_d___val:[a-zA-Z_]*]]){{.*}};

// CHECK: complexD fn8_darg0(double i, TensorD5 t) {
// CHECK-NEXT:     double _d_i = 1;
// CHECK-NEXT:     TensorD5 _d_t;
// CHECK-NEXT:     t.updateTo_pushforward(i * i, & _d_t, _d_i * i + i * _d_i);
// CHECK-NEXT:     complexD _d_c({0., 0.});
// CHECK-NEXT:     complexD c({0., 0.});
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t0 = t.sum_pushforward(& _d_t);
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

// CHECK: constexpr clad::ValueAndPushforward<double, double> real_pushforward(const std{{(::__1)?}}::complex<double> *_d_this){{.*}};

// CHECK: constexpr clad::ValueAndPushforward<double, double> imag_pushforward(const std{{(::__1)?}}::complex<double> *_d_this){{.*}};

// CHECK: complexD fn9_darg0(double i, complexD c) {
// CHECK-NEXT:     double _d_i = 1;
// CHECK-NEXT:     complexD _d_c;
// CHECK-NEXT:     complexD _d_r({0., 0.});
// CHECK-NEXT:     complexD r({0., 0.});
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

std::complex<double> fn10(double i, double j) {
  std::complex<double> c1, c2;
  c1.real(2 * i);
  c1.imag(5 * i);
  c2.real(5 * i);
  c2.imag(2 * i);
  c1 = c1 + c2;
  c1 += c2;
  return c1 + c1;
}

// CHECK: std::complex<double> fn10_darg0(double i, double j) {
// CHECK-NEXT:     double _d_i = 1;
// CHECK-NEXT:     double _d_j = 0;
// CHECK-NEXT:     std::complex<double> _d_c1({0., 0.}), _d_c2({0., 0.});
// CHECK-NEXT:     std::complex<double> c1({0., 0.}), c2({0., 0.});
// CHECK-NEXT:     c1.real_pushforward(2 * i, &_d_c1, 0 * i + 2 * _d_i);
// CHECK-NEXT:     c1.imag_pushforward(5 * i, &_d_c1, 0 * i + 5 * _d_i);
// CHECK-NEXT:     c2.real_pushforward(5 * i, &_d_c2, 0 * i + 5 * _d_i);
// CHECK-NEXT:     c2.imag_pushforward(2 * i, &_d_c2, 0 * i + 2 * _d_i);
// CHECK-NEXT:     clad::ValueAndPushforward<complex<double>, complex<double> > _t0 = operator_plus_pushforward(c1, c2, _d_c1, _d_c2);
// CHECK-NEXT:     clad::ValueAndPushforward<complex<double> &, complex<double> &> _t1 = c1.operator_equal_pushforward({{(static_cast<std(::__1)?::complex<double> &&>\(_t0.value\))|(_t0.value)}}, &_d_c1, {{(static_cast<std(::__1)?::complex<double> &&>\(_t0.pushforward\))|(_t0.pushforward)}});
// CHECK-NEXT:     clad::ValueAndPushforward<complex<double> &, complex<double> &> _t2 = c1.operator_plus_equal_pushforward(c2, &_d_c1, _d_c2);
// CHECK-NEXT:     clad::ValueAndPushforward<complex<double>, complex<double> > _t3 = operator_plus_pushforward(c1, c1, _d_c1, _d_c1);
// CHECK-NEXT:     return _t3.pushforward;
// CHECK-NEXT: }

TensorD5 fn11(double i, double j) {
  TensorD5 a, b;
  a(7*i);
  b(9*i);
  a[0] += 11*i;
  b[0] += 13*i;
  TensorD5 res1, res2;
  res1 = a + b + (a*b) + (-a) - b + a/a;
  TensorD5 one;
  one(1);
  res2 = (a+b)^(one);
  res1 += res2;
  res1 -= a*b;
  ++res1;
  ++res2;
  --res1;
  --res2;
  res1++;
  res2++;
  return res1;
}

// CHECK: void operator_call_pushforward(double val, Tensor<double, 5> *_d_this, double _d_val);

// CHECK: clad::ValueAndPushforward<double &, double &> operator_subscript_pushforward(std::size_t idx, Tensor<double, 5> *_d_this, std::size_t _d_idx);

// CHECK: clad::ValueAndPushforward<Tensor<double, 5U>, Tensor<double, 5U> > operator_plus_pushforward(const Tensor<double, 5U> &a, const Tensor<double, 5U> &b, const Tensor<double, 5U> &_d_a, const Tensor<double, 5U> &_d_b);

// CHECK: clad::ValueAndPushforward<Tensor<double, 5U>, Tensor<double, 5U> > operator_star_pushforward(const Tensor<double, 5U> &a, const Tensor<double, 5U> &b, const Tensor<double, 5U> &_d_a, const Tensor<double, 5U> &_d_b);

// CHECK: clad::ValueAndPushforward<Tensor<double, 5U>, Tensor<double, 5U> > operator_minus_pushforward(const Tensor<double, 5U> &t, const Tensor<double, 5U> &_d_t);

// CHECK: clad::ValueAndPushforward<Tensor<double, 5U>, Tensor<double, 5U> > operator_minus_pushforward(const Tensor<double, 5U> &a, const Tensor<double, 5U> &b, const Tensor<double, 5U> &_d_a, const Tensor<double, 5U> &_d_b);

// CHECK: clad::ValueAndPushforward<Tensor<double, 5U>, Tensor<double, 5U> > operator_slash_pushforward(const Tensor<double, 5U> &a, const Tensor<double, 5U> &b, const Tensor<double, 5U> &_d_a, const Tensor<double, 5U> &_d_b);

// CHECK: clad::ValueAndPushforward<Tensor<double, 5> &, Tensor<double, 5> &> operator_equal_pushforward(const Tensor<double, 5> &t, Tensor<double, 5> *_d_this, const Tensor<double, 5> &_d_t);

// CHECK: clad::ValueAndPushforward<Tensor<double, 5U>, Tensor<double, 5U> > operator_caret_pushforward(const Tensor<double, 5U> &a, const Tensor<double, 5U> &b, const Tensor<double, 5U> &_d_a, const Tensor<double, 5U> &_d_b);

// CHECK: clad::ValueAndPushforward<Tensor<double, 5U> &, Tensor<double, 5U> &> operator_plus_equal_pushforward(Tensor<double, 5U> &lhs, const Tensor<double, 5U> &rhs, Tensor<double, 5U> &_d_lhs, const Tensor<double, 5U> &_d_rhs);

// CHECK: clad::ValueAndPushforward<Tensor<double, 5U> &, Tensor<double, 5U> &> operator_minus_equal_pushforward(Tensor<double, 5U> &lhs, const Tensor<double, 5U> &rhs, Tensor<double, 5U> &_d_lhs, const Tensor<double, 5U> &_d_rhs);

// CHECK: clad::ValueAndPushforward<Tensor<double, 5> &, Tensor<double, 5> &> operator_plus_plus_pushforward(Tensor<double, 5> *_d_this);

// CHECK: clad::ValueAndPushforward<Tensor<double, 5> &, Tensor<double, 5> &> operator_minus_minus_pushforward(Tensor<double, 5> *_d_this);

// CHECK: clad::ValueAndPushforward<Tensor<double, 5>, Tensor<double, 5> > operator_plus_plus_pushforward(int param, Tensor<double, 5> *_d_this, int _d_param);

// CHECK: TensorD5 fn11_darg0(double i, double j) {
// CHECK-NEXT:     double _d_i = 1;
// CHECK-NEXT:     double _d_j = 0;
// CHECK-NEXT:     TensorD5 _d_a, _d_b;
// CHECK-NEXT:     TensorD5 a, b;
// CHECK-NEXT:     a.operator_call_pushforward(7 * i, & _d_a, 0 * i + 7 * _d_i);
// CHECK-NEXT:     b.operator_call_pushforward(9 * i, & _d_b, 0 * i + 9 * _d_i);
// CHECK-NEXT:     clad::ValueAndPushforward<double &, double &> _t0 = a.operator_subscript_pushforward(0, & _d_a, 0);
// CHECK-NEXT:     _t0.pushforward += 0 * i + 11 * _d_i;
// CHECK-NEXT:     _t0.value += 11 * i;
// CHECK-NEXT:     clad::ValueAndPushforward<double &, double &> _t1 = b.operator_subscript_pushforward(0, & _d_b, 0);
// CHECK-NEXT:     _t1.pushforward += 0 * i + 13 * _d_i;
// CHECK-NEXT:     _t1.value += 13 * i;
// CHECK-NEXT:     TensorD5 _d_res1, _d_res2;
// CHECK-NEXT:     TensorD5 res1, res2;
// CHECK-NEXT:     clad::ValueAndPushforward<Tensor<double, 5U>, Tensor<double, 5U> > _t2 = operator_plus_pushforward(a, b, _d_a, _d_b);
// CHECK-NEXT:     clad::ValueAndPushforward<Tensor<double, 5U>, Tensor<double, 5U> > _t3 = operator_star_pushforward(a, b, _d_a, _d_b);
// CHECK-NEXT:     clad::ValueAndPushforward<Tensor<double, 5U>, Tensor<double, 5U> > _t4 = operator_plus_pushforward(_t2.value, _t3.value, _t2.pushforward, _t3.pushforward);
// CHECK-NEXT:     clad::ValueAndPushforward<Tensor<double, 5U>, Tensor<double, 5U> > _t5 = operator_minus_pushforward(a, _d_a);
// CHECK-NEXT:     clad::ValueAndPushforward<Tensor<double, 5U>, Tensor<double, 5U> > _t6 = operator_plus_pushforward(_t4.value, _t5.value, _t4.pushforward, _t5.pushforward);
// CHECK-NEXT:     clad::ValueAndPushforward<Tensor<double, 5U>, Tensor<double, 5U> > _t7 = operator_minus_pushforward(_t6.value, b, _t6.pushforward, _d_b);
// CHECK-NEXT:     clad::ValueAndPushforward<Tensor<double, 5U>, Tensor<double, 5U> > _t8 = operator_slash_pushforward(a, a, _d_a, _d_a);
// CHECK-NEXT:     clad::ValueAndPushforward<Tensor<double, 5U>, Tensor<double, 5U> > _t9 = operator_plus_pushforward(_t7.value, _t8.value, _t7.pushforward, _t8.pushforward);
// CHECK-NEXT:     clad::ValueAndPushforward<Tensor<double, 5> &, Tensor<double, 5> &> _t10 = res1.operator_equal_pushforward(_t9.value, & _d_res1, _t9.pushforward);
// CHECK-NEXT:     TensorD5 _d_one;
// CHECK-NEXT:     TensorD5 one;
// CHECK-NEXT:     one.operator_call_pushforward(1, & _d_one, 0);
// CHECK-NEXT:     clad::ValueAndPushforward<Tensor<double, 5U>, Tensor<double, 5U> > _t11 = operator_plus_pushforward(a, b, _d_a, _d_b);
// CHECK-NEXT:     clad::ValueAndPushforward<Tensor<double, 5U>, Tensor<double, 5U> > _t12 = operator_caret_pushforward(_t11.value, one, _t11.pushforward, _d_one);
// CHECK-NEXT:     clad::ValueAndPushforward<Tensor<double, 5> &, Tensor<double, 5> &> _t13 = res2.operator_equal_pushforward(_t12.value, & _d_res2, _t12.pushforward);
// CHECK-NEXT:     clad::ValueAndPushforward<Tensor<double, 5U> &, Tensor<double, 5U> &> _t14 = operator_plus_equal_pushforward(res1, res2, _d_res1, _d_res2);
// CHECK-NEXT:     clad::ValueAndPushforward<Tensor<double, 5U>, Tensor<double, 5U> > _t15 = operator_star_pushforward(a, b, _d_a, _d_b);
// CHECK-NEXT:     clad::ValueAndPushforward<Tensor<double, 5U> &, Tensor<double, 5U> &> _t16 = operator_minus_equal_pushforward(res1, _t15.value, _d_res1, _t15.pushforward);
// CHECK-NEXT:     clad::ValueAndPushforward<Tensor<double, 5> &, Tensor<double, 5> &> _t17 = res1.operator_plus_plus_pushforward(& _d_res1);
// CHECK-NEXT:     clad::ValueAndPushforward<Tensor<double, 5> &, Tensor<double, 5> &> _t18 = res2.operator_plus_plus_pushforward(& _d_res2);
// CHECK-NEXT:     clad::ValueAndPushforward<Tensor<double, 5> &, Tensor<double, 5> &> _t19 = res1.operator_minus_minus_pushforward(& _d_res1);
// CHECK-NEXT:     clad::ValueAndPushforward<Tensor<double, 5> &, Tensor<double, 5> &> _t20 = res2.operator_minus_minus_pushforward(& _d_res2);
// CHECK-NEXT:     clad::ValueAndPushforward<Tensor<double, 5>, Tensor<double, 5> > _t21 = res1.operator_plus_plus_pushforward(0, & _d_res1, 0);
// CHECK-NEXT:     clad::ValueAndPushforward<Tensor<double, 5>, Tensor<double, 5> > _t22 = res2.operator_plus_plus_pushforward(0, & _d_res2, 0);
// CHECK-NEXT:     return _d_res1;
// CHECK-NEXT: }

TensorD5 fn12(double i, double j) {
  TensorD5 a, b;
  a(7*i);
  b(9*i);
  a[0] += 11*i;
  b[0] += 13*i;
  TensorD5 res1;
  TensorD5 one;
  one(1);
  res1(0);
  res1 += a;
  res1 *= b;
  res1 /= a;
  res1 -= b;
  res1 += a*a/a;
  res1[1] += 17*i;
  res1 ^= one;
  res1 = res1 = res1;
  return res1;
}

// CHECK: clad::ValueAndPushforward<Tensor<double, 5U> &, Tensor<double, 5U> &> operator_star_equal_pushforward(Tensor<double, 5U> &lhs, const Tensor<double, 5U> &rhs, Tensor<double, 5U> &_d_lhs, const Tensor<double, 5U> &_d_rhs);

// CHECK: clad::ValueAndPushforward<Tensor<double, 5U> &, Tensor<double, 5U> &> operator_slash_equal_pushforward(Tensor<double, 5U> &lhs, const Tensor<double, 5U> &rhs, Tensor<double, 5U> &_d_lhs, const Tensor<double, 5U> &_d_rhs);

// CHECK: clad::ValueAndPushforward<Tensor<double, 5U> &, Tensor<double, 5U> &> operator_caret_equal_pushforward(Tensor<double, 5U> &lhs, const Tensor<double, 5U> &rhs, Tensor<double, 5U> &_d_lhs, const Tensor<double, 5U> &_d_rhs);

// CHECK: TensorD5 fn12_darg0(double i, double j) {
// CHECK-NEXT:     double _d_i = 1;
// CHECK-NEXT:     double _d_j = 0;
// CHECK-NEXT:     TensorD5 _d_a, _d_b;
// CHECK-NEXT:     TensorD5 a, b;
// CHECK-NEXT:     a.operator_call_pushforward(7 * i, & _d_a, 0 * i + 7 * _d_i);
// CHECK-NEXT:     b.operator_call_pushforward(9 * i, & _d_b, 0 * i + 9 * _d_i);
// CHECK-NEXT:     clad::ValueAndPushforward<double &, double &> _t0 = a.operator_subscript_pushforward(0, & _d_a, 0);
// CHECK-NEXT:     _t0.pushforward += 0 * i + 11 * _d_i;
// CHECK-NEXT:     _t0.value += 11 * i;
// CHECK-NEXT:     clad::ValueAndPushforward<double &, double &> _t1 = b.operator_subscript_pushforward(0, & _d_b, 0);
// CHECK-NEXT:     _t1.pushforward += 0 * i + 13 * _d_i;
// CHECK-NEXT:     _t1.value += 13 * i;
// CHECK-NEXT:     TensorD5 _d_res1;
// CHECK-NEXT:     TensorD5 res1;
// CHECK-NEXT:     TensorD5 _d_one;
// CHECK-NEXT:     TensorD5 one;
// CHECK-NEXT:     one.operator_call_pushforward(1, & _d_one, 0);
// CHECK-NEXT:     res1.operator_call_pushforward(0, & _d_res1, 0);
// CHECK-NEXT:     clad::ValueAndPushforward<Tensor<double, 5U> &, Tensor<double, 5U> &> _t2 = operator_plus_equal_pushforward(res1, a, _d_res1, _d_a);
// CHECK-NEXT:     clad::ValueAndPushforward<Tensor<double, 5U> &, Tensor<double, 5U> &> _t3 = operator_star_equal_pushforward(res1, b, _d_res1, _d_b);
// CHECK-NEXT:     clad::ValueAndPushforward<Tensor<double, 5U> &, Tensor<double, 5U> &> _t4 = operator_slash_equal_pushforward(res1, a, _d_res1, _d_a);
// CHECK-NEXT:     clad::ValueAndPushforward<Tensor<double, 5U> &, Tensor<double, 5U> &> _t5 = operator_minus_equal_pushforward(res1, b, _d_res1, _d_b);
// CHECK-NEXT:     clad::ValueAndPushforward<Tensor<double, 5U>, Tensor<double, 5U> > _t6 = operator_star_pushforward(a, a, _d_a, _d_a);
// CHECK-NEXT:     clad::ValueAndPushforward<Tensor<double, 5U>, Tensor<double, 5U> > _t7 = operator_slash_pushforward(_t6.value, a, _t6.pushforward, _d_a);
// CHECK-NEXT:     clad::ValueAndPushforward<Tensor<double, 5U> &, Tensor<double, 5U> &> _t8 = operator_plus_equal_pushforward(res1, _t7.value, _d_res1, _t7.pushforward);
// CHECK-NEXT:     clad::ValueAndPushforward<double &, double &> _t9 = res1.operator_subscript_pushforward(1, & _d_res1, 0);
// CHECK-NEXT:     _t9.pushforward += 0 * i + 17 * _d_i;
// CHECK-NEXT:     _t9.value += 17 * i;
// CHECK-NEXT:     clad::ValueAndPushforward<Tensor<double, 5U> &, Tensor<double, 5U> &> _t10 = operator_caret_equal_pushforward(res1, one, _d_res1, _d_one);
// CHECK-NEXT:     clad::ValueAndPushforward<Tensor<double, 5> &, Tensor<double, 5> &> _t11 = res1.operator_equal_pushforward(res1, & _d_res1, _d_res1);
// CHECK-NEXT:     clad::ValueAndPushforward<Tensor<double, 5> &, Tensor<double, 5> &> _t12 = res1.operator_equal_pushforward(_t11.value, & _d_res1, _t11.pushforward);
// CHECK-NEXT:     return _d_res1;
// CHECK-NEXT: }

TensorD5 fn13(double i, double j) {
  TensorD5 a, b;
  a[0] = i*j;
  b[0] = i*i;
  a[0] += (a<b) + (a>b);
  b[0] += (a<=b) + (a>=b);
  a[0] += (a==b) + (a != b);
  a, b;
  !a;
  auto ap = &a;
  ap->data[1] = 7*i;
  auto b_data = b->data;
  b_data[1] = i*j;
  a = a % b;
  a %= b;
  ~a;
  a[2] = (a<<b) + (a>>b);
  b[2] = (a&&b) + (a||b);
  a <<= b;
  b >>= a;
  a[3] = (a&b) + (a|b);
  a |= b;
  b &= a;
  return a + b;
}

// CHECK: clad::ValueAndPushforward<bool, bool> operator_less_pushforward(Tensor<double, 5U> &lhs, const Tensor<double, 5U> &rhs, Tensor<double, 5U> &_d_lhs, const Tensor<double, 5U> &_d_rhs);

// CHECK: clad::ValueAndPushforward<bool, bool> operator_greater_pushforward(Tensor<double, 5U> &lhs, const Tensor<double, 5U> &rhs, Tensor<double, 5U> &_d_lhs, const Tensor<double, 5U> &_d_rhs);

// CHECK: clad::ValueAndPushforward<bool, bool> operator_less_equal_pushforward(Tensor<double, 5U> &lhs, const Tensor<double, 5U> &rhs, Tensor<double, 5U> &_d_lhs, const Tensor<double, 5U> &_d_rhs);

// CHECK: clad::ValueAndPushforward<bool, bool> operator_greater_equal_pushforward(Tensor<double, 5U> &lhs, const Tensor<double, 5U> &rhs, Tensor<double, 5U> &_d_lhs, const Tensor<double, 5U> &_d_rhs);

// CHECK: clad::ValueAndPushforward<bool, bool> operator_equal_equal_pushforward(Tensor<double, 5U> &lhs, const Tensor<double, 5U> &rhs, Tensor<double, 5U> &_d_lhs, const Tensor<double, 5U> &_d_rhs);

// CHECK: clad::ValueAndPushforward<bool, bool> operator_exclaim_equal_pushforward(Tensor<double, 5U> &lhs, const Tensor<double, 5U> &rhs, Tensor<double, 5U> &_d_lhs, const Tensor<double, 5U> &_d_rhs);

// CHECK: void operator_comma_pushforward(const Tensor<double, 5U> &lhs, const Tensor<double, 5U> &rhs, const Tensor<double, 5U> &_d_lhs, const Tensor<double, 5U> &_d_rhs);

// CHECK: void operator_exclaim_pushforward(const Tensor<double, 5U> &lhs, const Tensor<double, 5U> &_d_lhs);

// CHECK: clad::ValueAndPushforward<Tensor<double, 5> *, Tensor<double, 5> *> operator_amp_pushforward(Tensor<double, 5> *_d_this);

// CHECK: clad::ValueAndPushforward<Tensor<double, 5> *, Tensor<double, 5> *> operator_arrow_pushforward(Tensor<double, 5> *_d_this);

// CHECK: clad::ValueAndPushforward<Tensor<double, 5U>, Tensor<double, 5U> > operator_percent_pushforward(const Tensor<double, 5U> &a, const Tensor<double, 5U> &b, const Tensor<double, 5U> &_d_a, const Tensor<double, 5U> &_d_b);

// CHECK: clad::ValueAndPushforward<Tensor<double, 5U> &, Tensor<double, 5U> &> operator_percent_equal_pushforward(Tensor<double, 5U> &a, const Tensor<double, 5U> &b, Tensor<double, 5U> &_d_a, const Tensor<double, 5U> &_d_b);

// CHECK: void operator_tilde_pushforward(const Tensor<double, 5U> &a, const Tensor<double, 5U> &_d_a);

// CHECK: clad::ValueAndPushforward<bool, bool> operator_less_less_pushforward(Tensor<double, 5U> &lhs, const Tensor<double, 5U> &rhs, Tensor<double, 5U> &_d_lhs, const Tensor<double, 5U> &_d_rhs);

// CHECK: clad::ValueAndPushforward<bool, bool> operator_greater_greater_pushforward(Tensor<double, 5U> &lhs, const Tensor<double, 5U> &rhs, Tensor<double, 5U> &_d_lhs, const Tensor<double, 5U> &_d_rhs);

// CHECK: clad::ValueAndPushforward<bool, bool> operator_AmpAmp_pushforward(Tensor<double, 5U> &lhs, const Tensor<double, 5U> &rhs, Tensor<double, 5U> &_d_lhs, const Tensor<double, 5U> &_d_rhs);

// CHECK: clad::ValueAndPushforward<bool, bool> operator_pipe_pipe_pushforward(Tensor<double, 5U> &lhs, const Tensor<double, 5U> &rhs, Tensor<double, 5U> &_d_lhs, const Tensor<double, 5U> &_d_rhs);

// CHECK: clad::ValueAndPushforward<bool, bool> operator_less_less_equal_pushforward(Tensor<double, 5U> &lhs, const Tensor<double, 5U> &rhs, Tensor<double, 5U> &_d_lhs, const Tensor<double, 5U> &_d_rhs);

// CHECK: clad::ValueAndPushforward<bool, bool> operator_greater_greater_equal_pushforward(Tensor<double, 5U> &lhs, const Tensor<double, 5U> &rhs, Tensor<double, 5U> &_d_lhs, const Tensor<double, 5U> &_d_rhs);

// CHECK: clad::ValueAndPushforward<bool, bool> operator_amp_pushforward(Tensor<double, 5U> &lhs, const Tensor<double, 5U> &rhs, Tensor<double, 5U> &_d_lhs, const Tensor<double, 5U> &_d_rhs);

// CHECK: clad::ValueAndPushforward<bool, bool> operator_pipe_pushforward(Tensor<double, 5U> &lhs, const Tensor<double, 5U> &rhs, Tensor<double, 5U> &_d_lhs, const Tensor<double, 5U> &_d_rhs);

// CHECK: clad::ValueAndPushforward<bool, bool> operator_pipe_equal_pushforward(Tensor<double, 5U> &lhs, const Tensor<double, 5U> &rhs, Tensor<double, 5U> &_d_lhs, const Tensor<double, 5U> &_d_rhs);

// CHECK: clad::ValueAndPushforward<bool, bool> operator_amp_equal_pushforward(Tensor<double, 5U> &lhs, const Tensor<double, 5U> &rhs, Tensor<double, 5U> &_d_lhs, const Tensor<double, 5U> &_d_rhs);

// CHECK: TensorD5 fn13_darg0(double i, double j) {
// CHECK-NEXT:     double _d_i = 1;
// CHECK-NEXT:     double _d_j = 0;
// CHECK-NEXT:     TensorD5 _d_a, _d_b;
// CHECK-NEXT:     TensorD5 a, b;
// CHECK-NEXT:     clad::ValueAndPushforward<double &, double &> _t0 = a.operator_subscript_pushforward(0, & _d_a, 0);
// CHECK-NEXT:     _t0.pushforward = _d_i * j + i * _d_j;
// CHECK-NEXT:     _t0.value = i * j;
// CHECK-NEXT:     clad::ValueAndPushforward<double &, double &> _t1 = b.operator_subscript_pushforward(0, & _d_b, 0);
// CHECK-NEXT:     _t1.pushforward = _d_i * i + i * _d_i;
// CHECK-NEXT:     _t1.value = i * i;
// CHECK-NEXT:     clad::ValueAndPushforward<double &, double &> _t2 = a.operator_subscript_pushforward(0, & _d_a, 0);
// CHECK-NEXT:     clad::ValueAndPushforward<bool, bool> _t3 = operator_less_pushforward(a, b, _d_a, _d_b);
// CHECK-NEXT:     clad::ValueAndPushforward<bool, bool> _t4 = operator_greater_pushforward(a, b, _d_a, _d_b);
// CHECK-NEXT:     _t2.pushforward += _t3.pushforward + _t4.pushforward;
// CHECK-NEXT:     _t2.value += _t3.value + _t4.value;
// CHECK-NEXT:     clad::ValueAndPushforward<double &, double &> _t5 = b.operator_subscript_pushforward(0, & _d_b, 0);
// CHECK-NEXT:     clad::ValueAndPushforward<bool, bool> _t6 = operator_less_equal_pushforward(a, b, _d_a, _d_b);
// CHECK-NEXT:     clad::ValueAndPushforward<bool, bool> _t7 = operator_greater_equal_pushforward(a, b, _d_a, _d_b);
// CHECK-NEXT:     _t5.pushforward += _t6.pushforward + _t7.pushforward;
// CHECK-NEXT:     _t5.value += _t6.value + _t7.value;
// CHECK-NEXT:     clad::ValueAndPushforward<double &, double &> _t8 = a.operator_subscript_pushforward(0, & _d_a, 0);
// CHECK-NEXT:     clad::ValueAndPushforward<bool, bool> _t9 = operator_equal_equal_pushforward(a, b, _d_a, _d_b);
// CHECK-NEXT:     clad::ValueAndPushforward<bool, bool> _t10 = operator_exclaim_equal_pushforward(a, b, _d_a, _d_b);
// CHECK-NEXT:     _t8.pushforward += _t9.pushforward + _t10.pushforward;
// CHECK-NEXT:     _t8.value += _t9.value + _t10.value;
// CHECK-NEXT:     operator_comma_pushforward(a, b, _d_a, _d_b);
// CHECK-NEXT:     operator_exclaim_pushforward(a, _d_a);
// CHECK-NEXT:     clad::ValueAndPushforward<Tensor<double, 5> *, Tensor<double, 5> *> _t11 = a.operator_amp_pushforward(& _d_a);
// CHECK-NEXT:     Tensor<double, 5> *_d_ap = _t11.pushforward;
// CHECK-NEXT:     Tensor<double, 5> *ap = _t11.value;
// CHECK-NEXT:     _d_ap->data[1] = 0 * i + 7 * _d_i;
// CHECK-NEXT:     ap->data[1] = 7 * i;
// CHECK-NEXT:     clad::ValueAndPushforward<Tensor<double, 5> *, Tensor<double, 5> *> _t12 = b.operator_arrow_pushforward(& _d_b);
// CHECK-NEXT:     double *_d_b_data = _t12.pushforward->data;
// CHECK-NEXT:     double *b_data = _t12.value->data;
// CHECK-NEXT:     _d_b_data[1] = _d_i * j + i * _d_j;
// CHECK-NEXT:     b_data[1] = i * j;
// CHECK-NEXT:     clad::ValueAndPushforward<Tensor<double, 5U>, Tensor<double, 5U> > _t13 = operator_percent_pushforward(a, b, _d_a, _d_b);
// CHECK-NEXT:     clad::ValueAndPushforward<Tensor<double, 5> &, Tensor<double, 5> &> _t14 = a.operator_equal_pushforward(_t13.value, & _d_a, _t13.pushforward);
// CHECK-NEXT:     clad::ValueAndPushforward<Tensor<double, 5U> &, Tensor<double, 5U> &> _t15 = operator_percent_equal_pushforward(a, b, _d_a, _d_b);
// CHECK-NEXT:     operator_tilde_pushforward(a, _d_a);
// CHECK-NEXT:     clad::ValueAndPushforward<double &, double &> _t16 = a.operator_subscript_pushforward(2, & _d_a, 0);
// CHECK-NEXT:     clad::ValueAndPushforward<bool, bool> _t17 = operator_less_less_pushforward(a, b, _d_a, _d_b);
// CHECK-NEXT:     clad::ValueAndPushforward<bool, bool> _t18 = operator_greater_greater_pushforward(a, b, _d_a, _d_b);
// CHECK-NEXT:     _t16.pushforward = _t17.pushforward + _t18.pushforward;
// CHECK-NEXT:     _t16.value = _t17.value + _t18.value;
// CHECK-NEXT:     clad::ValueAndPushforward<double &, double &> _t19 = b.operator_subscript_pushforward(2, & _d_b, 0);
// CHECK-NEXT:     clad::ValueAndPushforward<bool, bool> _t20 = operator_AmpAmp_pushforward(a, b, _d_a, _d_b);
// CHECK-NEXT:     clad::ValueAndPushforward<bool, bool> _t21 = operator_pipe_pipe_pushforward(a, b, _d_a, _d_b);
// CHECK-NEXT:     _t19.pushforward = _t20.pushforward + _t21.pushforward;
// CHECK-NEXT:     _t19.value = _t20.value + _t21.value;
// CHECK-NEXT:     clad::ValueAndPushforward<bool, bool> _t22 = operator_less_less_equal_pushforward(a, b, _d_a, _d_b);
// CHECK-NEXT:     clad::ValueAndPushforward<bool, bool> _t23 = operator_greater_greater_equal_pushforward(b, a, _d_b, _d_a);
// CHECK-NEXT:     clad::ValueAndPushforward<double &, double &> _t24 = a.operator_subscript_pushforward(3, & _d_a, 0);
// CHECK-NEXT:     clad::ValueAndPushforward<bool, bool> _t25 = operator_amp_pushforward(a, b, _d_a, _d_b);
// CHECK-NEXT:     clad::ValueAndPushforward<bool, bool> _t26 = operator_pipe_pushforward(a, b, _d_a, _d_b);
// CHECK-NEXT:     _t24.pushforward = _t25.pushforward + _t26.pushforward;
// CHECK-NEXT:     _t24.value = _t25.value + _t26.value;
// CHECK-NEXT:     clad::ValueAndPushforward<bool, bool> _t27 = operator_pipe_equal_pushforward(a, b, _d_a, _d_b);
// CHECK-NEXT:     clad::ValueAndPushforward<bool, bool> _t28 = operator_amp_equal_pushforward(b, a, _d_b, _d_a);
// CHECK-NEXT:     clad::ValueAndPushforward<Tensor<double, 5U>, Tensor<double, 5U> > _t29 = operator_plus_pushforward(a, b, _d_a, _d_b);
// CHECK-NEXT:     return _t29.pushforward;
// CHECK-NEXT: }

using vectorD = std::vector<double>;

double fn14(double i, double j) {
  vectorD v;
  v.resize(5, 0);
  v[0] = 9 * i;
  v[1] = 11 * i;
  auto b = std::begin(v);
  auto e = std::end(v);
  auto res = std::accumulate(b, e, 0.00);
  return res;
}

// CHECK: double fn14_darg0(double i, double j) {
// CHECK-NEXT:     double _d_i = 1;
// CHECK-NEXT:     double _d_j = 0;
// CHECK-NEXT:     vectorD _d_v;
// CHECK-NEXT:     vectorD v;
// CHECK-NEXT:     clad::custom_derivatives::class_functions::resize_pushforward(&v, 5, 0, &_d_v, 0, 0);
// CHECK-NEXT:     {{.*}}ValueAndPushforward<{{.*}}, {{.*}}> _t0 = clad::custom_derivatives::class_functions::operator_subscript_pushforward(&v, 0, &_d_v, 0);
// CHECK-NEXT:     _t0.pushforward = 0 * i + 9 * _d_i;
// CHECK-NEXT:     _t0.value = 9 * i;
// CHECK-NEXT:     {{.*}}ValueAndPushforward<{{.*}}, {{.*}}> _t1 = clad::custom_derivatives::class_functions::operator_subscript_pushforward(&v, 1, &_d_v, 0);
// CHECK-NEXT:     _t1.pushforward = 0 * i + 11 * _d_i;
// CHECK-NEXT:     _t1.value = 11 * i;
// CHECK-NEXT:     clad::ValueAndPushforward<decltype({{.*}}.begin()), decltype({{.*}}.begin())> _t2 = begin_pushforward(v, _d_v);
// CHECK-NEXT:     {{.*}} _d_b = _t2.pushforward;
// CHECK-NEXT:     {{.*}} b = _t2.value;
// CHECK-NEXT:     clad::ValueAndPushforward<decltype({{.*}}.end()), decltype({{.*}}.end())> _t3 = end_pushforward(v, _d_v);
// CHECK-NEXT:     {{.*}} _d_e = _t3.pushforward;
// CHECK-NEXT:     {{.*}} e = _t3.value;
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t4 = accumulate_pushforward(b, e, 0., _d_b, _d_e, 0.);
// CHECK-NEXT:     double _d_res = _t4.pushforward;
// CHECK-NEXT:     double res = _t4.value;
// CHECK-NEXT:     return _d_res;
// CHECK-NEXT: }

using pairdd = std::pair<double, double>;
using pair_of_pairdd = std::pair<pairdd, pairdd>;

double fn15(pairdd u, pairdd v) {
  return u.first + 2*v.first;
}

// CHECK: double fn15_darg0_first(pairdd u, pairdd v) {
// CHECK-NEXT:     pairdd _d_u;
// CHECK-NEXT:     _d_u.first = 1;
// CHECK-NEXT:     pairdd _d_v;
// CHECK-NEXT:     double &_t0 = v.first;
// CHECK-NEXT:     return _d_u.first + 0 * _t0 + 2 * _d_v.first;
// CHECK-NEXT: }

double fn16(pair_of_pairdd u, pair_of_pairdd v) {
  return u.first.first + 2*v.second.second;
}

// CHECK: double fn16_darg1_second_second(pair_of_pairdd u, pair_of_pairdd v) {
// CHECK-NEXT:     pair_of_pairdd _d_u;
// CHECK-NEXT:     pair_of_pairdd _d_v;
// CHECK-NEXT:     _d_v.second.second = 1;
// CHECK-NEXT:     double &_t0 = v.second.second;
// CHECK-NEXT:     return _d_u.first.first + 0 * _t0 + 2 * _d_v.second.second;
// CHECK-NEXT: }


struct A {
  double mem;
  A(double p_mem = 0) : mem(p_mem) {}
};

struct B : public A {
  double mem;
  B(double p_mem = 0) : A(0), mem(p_mem) {}
};

double fn17(A a, B b) {
  return a.mem * b.mem;
}

// CHECK: double fn17_darg1_mem(A a, B b) {
// CHECK-NEXT:     A _d_a;
// CHECK-NEXT:     B _d_b;
// CHECK-NEXT:     _d_b.mem = 1;
// CHECK-NEXT:     double &_t0 = a.mem;
// CHECK-NEXT:     double &_t1 = b.mem;
// CHECK-NEXT:     return _d_a.mem * _t1 + _t0 * _d_b.mem;
// CHECK-NEXT: }

double fn18(double i, double j) {
  A v[2] = {2, 3};
  v[0] = 9 * i;
  return v[0].mem;
}

// CHECK: double fn18_darg0(double i, double j) {
// CHECK-NEXT:     double _d_i = 1;
// CHECK-NEXT:     double _d_j = 0;
// CHECK-NEXT:     A _d_v[2] = {0, 0};
// CHECK-NEXT:     A v[2] = {2, 3};
// CHECK-NEXT:     clad::ValueAndPushforward<A &, A &> _t0 = v[0].operator_equal_pushforward(9 * i, &_d_v[0], 0 * i + 9 * _d_i);
// CHECK-NEXT:     return _d_v[0].mem;
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
  INIT_DIFFERENTIATE(fn10, "i");
  INIT_DIFFERENTIATE(fn11, "i");
  INIT_DIFFERENTIATE(fn12, "i");
  INIT_DIFFERENTIATE(fn13, "i");
  INIT_DIFFERENTIATE(fn14, "i");
  INIT_DIFFERENTIATE(fn15, "u.first");
  INIT_DIFFERENTIATE(fn16, "v.second.second");
  INIT_DIFFERENTIATE(fn17, "b.mem");
  INIT_DIFFERENTIATE(fn18, "i");

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
  TEST_DIFFERENTIATE(fn10, 3, 5); // CHECK-EXEC: {24.00, 18.00}
  TEST_DIFFERENTIATE(fn11, 3, 5); // CHECK-EXEC: {40.00, 16.00, 16.00, 16.00, 16.00}
  TEST_DIFFERENTIATE(fn12, 3, 5); // CHECK-EXEC: {18.00, 24.00, 7.00, 7.00, 7.00}
  TEST_DIFFERENTIATE(fn13, 3, 5); // CHECK-EXEC: {11.00, 12.00, 0.00, 0.00, 0.00}
  TEST_DIFFERENTIATE(fn14, 3, 5); // CHECK-EXEC: {20.00}
  TEST_DIFFERENTIATE(fn15, pairdd(), pairdd());                   // CHECK-EXEC: {1.00}
  TEST_DIFFERENTIATE(fn16, pair_of_pairdd(), pair_of_pairdd());   // CHECK-EXEC: {2.00}
  TEST_DIFFERENTIATE(fn17, A(3.00), B(5.00));   // CHECK-EXEC: {3.00}
  TEST_DIFFERENTIATE(fn18, 7, 3);   // CHECK-EXEC: {9.00}

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

// CHECK: void real_pushforward({{.*}} [[__val:.*]], std{{(::__1)?}}::complex<double> *_d_this, {{.*}} [[_d___val:[a-zA-Z_]*]]){{.*}} {
// CHECK-NEXT:     {{(__real)?}} _d_this->[[_M_value:.*]] = [[_d___val]];
// CHECK-NEXT:     {{(__real)?}} this->[[_M_value]] = [[__val]];
// CHECK-NEXT: }

// CHECK: void imag_pushforward({{.*}} [[__val:.*]], std{{(::__1)?}}::complex<double> *_d_this, {{.*}} [[_d___val:[a-zA-Z_]*]]){{.*}} {
// CHECK-NEXT:     {{(__imag)?}} _d_this->[[_M_value:.*]] = [[_d___val]];
// CHECK-NEXT:     {{(__imag)?}} this->[[_M_value]] = [[__val]];
// CHECK-NEXT: }

// CHECK: constexpr clad::ValueAndPushforward<double, double> real_pushforward(const std{{(::__1)?}}::complex<double> *_d_this){{.*}} {
// CHECK-NEXT:     return {{[{](__real )?}}this->[[_M_value:[a-zA-Z_]+]],{{( __real)?}} _d_this->[[_M_value:[a-zA-Z_]+]]};
// CHECK-NEXT: }

// CHECK: constexpr clad::ValueAndPushforward<double, double> imag_pushforward(const std{{(::__1)?}}::complex<double> *_d_this){{.*}} {
// CHECK-NEXT:     return {{[{](__imag )?}}this->[[_M_value:[a-zA-Z_]+]],{{( __imag)?}} _d_this->[[_M_value:[a-zA-Z_]+]]};
// CHECK-NEXT: }

// CHECK: void operator_call_pushforward(double val, Tensor<double, 5> *_d_this, double _d_val) {
// CHECK-NEXT:     {
// CHECK-NEXT:         unsigned int _d_i = 0;
// CHECK-NEXT:         for (unsigned int i = 0; i < 5U; ++i) {
// CHECK-NEXT:             _d_this->data[i] = _d_val;
// CHECK-NEXT:             this->data[i] = val;
// CHECK-NEXT:         }
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: clad::ValueAndPushforward<double &, double &> operator_subscript_pushforward(std::size_t idx, Tensor<double, 5> *_d_this, std::size_t _d_idx) {
// CHECK-NEXT:     return {(double &)this->data[idx], (double &)_d_this->data[idx]};
// CHECK-NEXT: }

// CHECK: clad::ValueAndPushforward<Tensor<double, 5U>, Tensor<double, 5U> > operator_plus_pushforward(const Tensor<double, 5U> &a, const Tensor<double, 5U> &b, const Tensor<double, 5U> &_d_a, const Tensor<double, 5U> &_d_b) {
// CHECK-NEXT:     Tensor<double, 5U> _d_res;
// CHECK-NEXT:     Tensor<double, 5U> res;
// CHECK-NEXT:     {
// CHECK-NEXT:         unsigned int _d_i = 0;
// CHECK-NEXT:         for (unsigned int i = 0; i < 5U; ++i) {
// CHECK-NEXT:             _d_res.data[i] = _d_a.data[i] + _d_b.data[i];
// CHECK-NEXT:             res.data[i] = a.data[i] + b.data[i];
// CHECK-NEXT:         }
// CHECK-NEXT:     }
// CHECK-NEXT:     return {res, _d_res};
// CHECK-NEXT: }

// CHECK: clad::ValueAndPushforward<Tensor<double, 5U>, Tensor<double, 5U> > operator_star_pushforward(const Tensor<double, 5U> &a, const Tensor<double, 5U> &b, const Tensor<double, 5U> &_d_a, const Tensor<double, 5U> &_d_b) {
// CHECK-NEXT:     Tensor<double, 5U> _d_res;
// CHECK-NEXT:     Tensor<double, 5U> res;
// CHECK-NEXT:     {
// CHECK-NEXT:         unsigned int _d_i = 0;
// CHECK-NEXT:         for (unsigned int i = 0; i < 5U; ++i) {
// CHECK-NEXT:             const double _t0 = a.data[i];
// CHECK-NEXT:             const double _t1 = b.data[i];
// CHECK-NEXT:             _d_res.data[i] = _d_a.data[i] * _t1 + _t0 * _d_b.data[i];
// CHECK-NEXT:             res.data[i] = _t0 * _t1;
// CHECK-NEXT:         }
// CHECK-NEXT:     }
// CHECK-NEXT:     return {res, _d_res};
// CHECK-NEXT: }

// CHECK: clad::ValueAndPushforward<Tensor<double, 5U>, Tensor<double, 5U> > operator_minus_pushforward(const Tensor<double, 5U> &t, const Tensor<double, 5U> &_d_t) {
// CHECK-NEXT:     Tensor<double, 5U> _d_res;
// CHECK-NEXT:     Tensor<double, 5U> res;
// CHECK-NEXT:     {
// CHECK-NEXT:         unsigned int _d_i = 0;
// CHECK-NEXT:         for (unsigned int i = 0; i < 5U; ++i) {
// CHECK-NEXT:             _d_res.data[i] = -_d_t.data[i];
// CHECK-NEXT:             res.data[i] = -t.data[i];
// CHECK-NEXT:         }
// CHECK-NEXT:     }
// CHECK-NEXT:     return {res, _d_res};
// CHECK-NEXT: }

// CHECK: clad::ValueAndPushforward<Tensor<double, 5U>, Tensor<double, 5U> > operator_minus_pushforward(const Tensor<double, 5U> &a, const Tensor<double, 5U> &b, const Tensor<double, 5U> &_d_a, const Tensor<double, 5U> &_d_b) {
// CHECK-NEXT:     Tensor<double, 5U> _d_res;
// CHECK-NEXT:     Tensor<double, 5U> res;
// CHECK-NEXT:     {
// CHECK-NEXT:         unsigned int _d_i = 0;
// CHECK-NEXT:         for (unsigned int i = 0; i < 5U; ++i) {
// CHECK-NEXT:             _d_res.data[i] = _d_a.data[i] - _d_b.data[i];
// CHECK-NEXT:             res.data[i] = a.data[i] - b.data[i];
// CHECK-NEXT:         }
// CHECK-NEXT:     }
// CHECK-NEXT:     return {res, _d_res};
// CHECK-NEXT: }

// CHECK: clad::ValueAndPushforward<Tensor<double, 5U>, Tensor<double, 5U> > operator_slash_pushforward(const Tensor<double, 5U> &a, const Tensor<double, 5U> &b, const Tensor<double, 5U> &_d_a, const Tensor<double, 5U> &_d_b) {
// CHECK-NEXT:     Tensor<double, 5U> _d_res;
// CHECK-NEXT:     Tensor<double, 5U> res;
// CHECK-NEXT:     {
// CHECK-NEXT:         unsigned int _d_i = 0;
// CHECK-NEXT:         for (unsigned int i = 0; i < 5U; ++i) {
// CHECK-NEXT:             const double _t0 = a.data[i];
// CHECK-NEXT:             const double _t1 = b.data[i];
// CHECK-NEXT:             _d_res.data[i] = (_d_a.data[i] * _t1 - _t0 * _d_b.data[i]) / (_t1 * _t1);
// CHECK-NEXT:             res.data[i] = _t0 / _t1;
// CHECK-NEXT:         }
// CHECK-NEXT:     }
// CHECK-NEXT:     return {res, _d_res};
// CHECK-NEXT: }

// CHECK: clad::ValueAndPushforward<Tensor<double, 5> &, Tensor<double, 5> &> operator_equal_pushforward(const Tensor<double, 5> &t, Tensor<double, 5> *_d_this, const Tensor<double, 5> &_d_t) {
// CHECK-NEXT:     {
// CHECK-NEXT:         unsigned int _d_i = 0;
// CHECK-NEXT:         for (unsigned int i = 0; i < 5U; ++i) {
// CHECK-NEXT:             _d_this->data[i] = _d_t.data[i];
// CHECK-NEXT:             this->data[i] = t.data[i];
// CHECK-NEXT:         }
// CHECK-NEXT:     }
// CHECK-NEXT:     return {(Tensor<double, 5> &)*this, (Tensor<double, 5> &)*_d_this};
// CHECK-NEXT: }

// CHECK: clad::ValueAndPushforward<Tensor<double, 5U>, Tensor<double, 5U> > operator_caret_pushforward(const Tensor<double, 5U> &a, const Tensor<double, 5U> &b, const Tensor<double, 5U> &_d_a, const Tensor<double, 5U> &_d_b) {
// CHECK-NEXT:     Tensor<double, 5U> _d_res;
// CHECK-NEXT:     Tensor<double, 5U> res;
// CHECK-NEXT:     {
// CHECK-NEXT:         unsigned int _d_i = 0;
// CHECK-NEXT:         for (unsigned int i = 0; i < 5U; ++i) {
// CHECK-NEXT:             ValueAndPushforward<decltype(::std::pow(double(), double())), decltype(::std::pow(double(), double()))> _t0 = clad::custom_derivatives::pow_pushforward(a.data[i], b.data[i], _d_a.data[i], _d_b.data[i]);
// CHECK-NEXT:             _d_res.data[i] = _t0.pushforward;
// CHECK-NEXT:             res.data[i] = _t0.value;
// CHECK-NEXT:         }
// CHECK-NEXT:     }
// CHECK-NEXT:     return {res, _d_res};
// CHECK-NEXT: }

// CHECK: clad::ValueAndPushforward<Tensor<double, 5U> &, Tensor<double, 5U> &> operator_plus_equal_pushforward(Tensor<double, 5U> &lhs, const Tensor<double, 5U> &rhs, Tensor<double, 5U> &_d_lhs, const Tensor<double, 5U> &_d_rhs) {
// CHECK-NEXT:     clad::ValueAndPushforward<Tensor<double, 5U>, Tensor<double, 5U> > _t0 = operator_plus_pushforward(lhs, rhs, _d_lhs, _d_rhs);
// CHECK-NEXT:     clad::ValueAndPushforward<Tensor<double, 5> &, Tensor<double, 5> &> _t1 = lhs.operator_equal_pushforward(_t0.value, & _d_lhs, _t0.pushforward);
// CHECK-NEXT:     return {(Tensor<double, 5U> &)lhs, (Tensor<double, 5U> &)_d_lhs};
// CHECK-NEXT: }

// CHECK: clad::ValueAndPushforward<Tensor<double, 5U> &, Tensor<double, 5U> &> operator_minus_equal_pushforward(Tensor<double, 5U> &lhs, const Tensor<double, 5U> &rhs, Tensor<double, 5U> &_d_lhs, const Tensor<double, 5U> &_d_rhs) {
// CHECK-NEXT:     clad::ValueAndPushforward<Tensor<double, 5U>, Tensor<double, 5U> > _t0 = operator_minus_pushforward(lhs, rhs, _d_lhs, _d_rhs);
// CHECK-NEXT:     clad::ValueAndPushforward<Tensor<double, 5> &, Tensor<double, 5> &> _t1 = lhs.operator_equal_pushforward(_t0.value, & _d_lhs, _t0.pushforward);
// CHECK-NEXT:     return {(Tensor<double, 5U> &)lhs, (Tensor<double, 5U> &)_d_lhs};
// CHECK-NEXT: }

// CHECK: clad::ValueAndPushforward<Tensor<double, 5> &, Tensor<double, 5> &> operator_plus_plus_pushforward(Tensor<double, 5> *_d_this) {
// CHECK-NEXT:     {
// CHECK-NEXT:         unsigned int _d_i = 0;
// CHECK-NEXT:         for (unsigned int i = 0; i < 5U; ++i) {
// CHECK-NEXT:             _d_this->data[i] += 0;
// CHECK-NEXT:             this->data[i] += 1;
// CHECK-NEXT:         }
// CHECK-NEXT:     }
// CHECK-NEXT:     return {(Tensor<double, 5> &)*this, (Tensor<double, 5> &)*_d_this};
// CHECK-NEXT: }

// CHECK: clad::ValueAndPushforward<Tensor<double, 5> &, Tensor<double, 5> &> operator_minus_minus_pushforward(Tensor<double, 5> *_d_this) {
// CHECK-NEXT:     {
// CHECK-NEXT:         unsigned int _d_i = 0;
// CHECK-NEXT:         for (unsigned int i = 0; i < 5U; ++i) {
// CHECK-NEXT:             _d_this->data[i] += 0;
// CHECK-NEXT:             this->data[i] += 1;
// CHECK-NEXT:         }
// CHECK-NEXT:     }
// CHECK-NEXT:     return {(Tensor<double, 5> &)*this, (Tensor<double, 5> &)*_d_this};
// CHECK-NEXT: }

// CHECK: clad::ValueAndPushforward<Tensor<double, 5>, Tensor<double, 5> > operator_plus_plus_pushforward(int param, Tensor<double, 5> *_d_this, int _d_param) {
// CHECK-NEXT:     Tensor<double, 5> _d_temp;
// CHECK-NEXT:     Tensor<double, 5> temp;
// CHECK-NEXT:     {
// CHECK-NEXT:         unsigned int _d_i = 0;
// CHECK-NEXT:         for (unsigned int i = 0; i < 5U; ++i) {
// CHECK-NEXT:             _d_temp.data[i] += 0;
// CHECK-NEXT:             temp.data[i] += 1;
// CHECK-NEXT:         }
// CHECK-NEXT:     }
// CHECK-NEXT:     return {temp, _d_temp};
// CHECK-NEXT: }

// CHECK: clad::ValueAndPushforward<Tensor<double, 5U> &, Tensor<double, 5U> &> operator_star_equal_pushforward(Tensor<double, 5U> &lhs, const Tensor<double, 5U> &rhs, Tensor<double, 5U> &_d_lhs, const Tensor<double, 5U> &_d_rhs) {
// CHECK-NEXT:     clad::ValueAndPushforward<Tensor<double, 5U>, Tensor<double, 5U> > _t0 = operator_star_pushforward(lhs, rhs, _d_lhs, _d_rhs);
// CHECK-NEXT:     clad::ValueAndPushforward<Tensor<double, 5> &, Tensor<double, 5> &> _t1 = lhs.operator_equal_pushforward(_t0.value, & _d_lhs, _t0.pushforward);
// CHECK-NEXT:     return {(Tensor<double, 5U> &)lhs, (Tensor<double, 5U> &)_d_lhs};
// CHECK-NEXT: }

// CHECK: clad::ValueAndPushforward<Tensor<double, 5U> &, Tensor<double, 5U> &> operator_slash_equal_pushforward(Tensor<double, 5U> &lhs, const Tensor<double, 5U> &rhs, Tensor<double, 5U> &_d_lhs, const Tensor<double, 5U> &_d_rhs) {
// CHECK-NEXT:     clad::ValueAndPushforward<Tensor<double, 5U>, Tensor<double, 5U> > _t0 = operator_slash_pushforward(lhs, rhs, _d_lhs, _d_rhs);
// CHECK-NEXT:     clad::ValueAndPushforward<Tensor<double, 5> &, Tensor<double, 5> &> _t1 = lhs.operator_equal_pushforward(_t0.value, & _d_lhs, _t0.pushforward);
// CHECK-NEXT:     return {(Tensor<double, 5U> &)lhs, (Tensor<double, 5U> &)_d_lhs};
// CHECK-NEXT: }

// CHECK: clad::ValueAndPushforward<Tensor<double, 5U> &, Tensor<double, 5U> &> operator_caret_equal_pushforward(Tensor<double, 5U> &lhs, const Tensor<double, 5U> &rhs, Tensor<double, 5U> &_d_lhs, const Tensor<double, 5U> &_d_rhs) {
// CHECK-NEXT:     clad::ValueAndPushforward<Tensor<double, 5U>, Tensor<double, 5U> > _t0 = operator_slash_pushforward(lhs, rhs, _d_lhs, _d_rhs);
// CHECK-NEXT:     clad::ValueAndPushforward<Tensor<double, 5> &, Tensor<double, 5> &> _t1 = lhs.operator_equal_pushforward(_t0.value, & _d_lhs, _t0.pushforward);
// CHECK-NEXT:     return {(Tensor<double, 5U> &)lhs, (Tensor<double, 5U> &)_d_lhs};
// CHECK-NEXT: }

// CHECK: clad::ValueAndPushforward<bool, bool> operator_less_pushforward(Tensor<double, 5U> &lhs, const Tensor<double, 5U> &rhs, Tensor<double, 5U> &_d_lhs, const Tensor<double, 5U> &_d_rhs) {
// CHECK-NEXT:     double _d_lsum, _d_rsum;
// CHECK-NEXT:     double lsum, rsum;
// CHECK-NEXT:     _d_lsum = _d_rsum = 0;
// CHECK-NEXT:     lsum = rsum = 0;
// CHECK-NEXT:     {
// CHECK-NEXT:         unsigned int _d_i = 0;
// CHECK-NEXT:         for (unsigned int i = 0; i < 5U; ++i) {
// CHECK-NEXT:             _d_lsum += _d_lhs.data[i];
// CHECK-NEXT:             lsum += lhs.data[i];
// CHECK-NEXT:             _d_rsum += _d_lhs.data[i];
// CHECK-NEXT:             rsum += lhs.data[i];
// CHECK-NEXT:         }
// CHECK-NEXT:     }
// CHECK-NEXT:     return {(bool)1, (bool)0};
// CHECK-NEXT: }

// CHECK: clad::ValueAndPushforward<bool, bool> operator_greater_pushforward(Tensor<double, 5U> &lhs, const Tensor<double, 5U> &rhs, Tensor<double, 5U> &_d_lhs, const Tensor<double, 5U> &_d_rhs) {
// CHECK-NEXT:     double _d_lsum, _d_rsum;
// CHECK-NEXT:     double lsum, rsum;
// CHECK-NEXT:     _d_lsum = _d_rsum = 0;
// CHECK-NEXT:     lsum = rsum = 0;
// CHECK-NEXT:     {
// CHECK-NEXT:         unsigned int _d_i = 0;
// CHECK-NEXT:         for (unsigned int i = 0; i < 5U; ++i) {
// CHECK-NEXT:             _d_lsum += _d_lhs.data[i];
// CHECK-NEXT:             lsum += lhs.data[i];
// CHECK-NEXT:             _d_rsum += _d_lhs.data[i];
// CHECK-NEXT:             rsum += lhs.data[i];
// CHECK-NEXT:         }
// CHECK-NEXT:     }
// CHECK-NEXT:     return {(bool)1, (bool)0};
// CHECK-NEXT: }

// CHECK: clad::ValueAndPushforward<bool, bool> operator_less_equal_pushforward(Tensor<double, 5U> &lhs, const Tensor<double, 5U> &rhs, Tensor<double, 5U> &_d_lhs, const Tensor<double, 5U> &_d_rhs) {
// CHECK-NEXT:     return {(bool)1, (bool)0};
// CHECK-NEXT: }

// CHECK: clad::ValueAndPushforward<bool, bool> operator_greater_equal_pushforward(Tensor<double, 5U> &lhs, const Tensor<double, 5U> &rhs, Tensor<double, 5U> &_d_lhs, const Tensor<double, 5U> &_d_rhs) {
// CHECK-NEXT:     return {(bool)1, (bool)0};
// CHECK-NEXT: }

// CHECK: clad::ValueAndPushforward<bool, bool> operator_equal_equal_pushforward(Tensor<double, 5U> &lhs, const Tensor<double, 5U> &rhs, Tensor<double, 5U> &_d_lhs, const Tensor<double, 5U> &_d_rhs) {
// CHECK-NEXT:     return {(bool)1, (bool)0};
// CHECK-NEXT: }

// CHECK: clad::ValueAndPushforward<bool, bool> operator_exclaim_equal_pushforward(Tensor<double, 5U> &lhs, const Tensor<double, 5U> &rhs, Tensor<double, 5U> &_d_lhs, const Tensor<double, 5U> &_d_rhs) {
// CHECK-NEXT:     return {(bool)1, (bool)0};
// CHECK-NEXT: }

// CHECK: void operator_comma_pushforward(const Tensor<double, 5U> &lhs, const Tensor<double, 5U> &rhs, const Tensor<double, 5U> &_d_lhs, const Tensor<double, 5U> &_d_rhs) {
// CHECK-NEXT: }

// CHECK: void operator_exclaim_pushforward(const Tensor<double, 5U> &lhs, const Tensor<double, 5U> &_d_lhs) {
// CHECK-NEXT: }

// CHECK: clad::ValueAndPushforward<Tensor<double, 5> *, Tensor<double, 5> *> operator_amp_pushforward(Tensor<double, 5> *_d_this) {
// CHECK-NEXT:     return {this, _d_this};
// CHECK-NEXT: }

// CHECK: clad::ValueAndPushforward<Tensor<double, 5> *, Tensor<double, 5> *> operator_arrow_pushforward(Tensor<double, 5> *_d_this) {
// CHECK-NEXT:     return {this, _d_this};
// CHECK-NEXT: }

// CHECK: clad::ValueAndPushforward<Tensor<double, 5U>, Tensor<double, 5U> > operator_percent_pushforward(const Tensor<double, 5U> &a, const Tensor<double, 5U> &b, const Tensor<double, 5U> &_d_a, const Tensor<double, 5U> &_d_b) {
// CHECK-NEXT:     return {a, _d_a};
// CHECK-NEXT: }

// CHECK: clad::ValueAndPushforward<Tensor<double, 5U> &, Tensor<double, 5U> &> operator_percent_equal_pushforward(Tensor<double, 5U> &a, const Tensor<double, 5U> &b, Tensor<double, 5U> &_d_a, const Tensor<double, 5U> &_d_b) {
// CHECK-NEXT:     return {(Tensor<double, 5U> &)a, (Tensor<double, 5U> &)_d_a};
// CHECK-NEXT: }

// CHECK: void operator_tilde_pushforward(const Tensor<double, 5U> &a, const Tensor<double, 5U> &_d_a) {
// CHECK-NEXT: }

// CHECK: clad::ValueAndPushforward<bool, bool> operator_less_less_pushforward(Tensor<double, 5U> &lhs, const Tensor<double, 5U> &rhs, Tensor<double, 5U> &_d_lhs, const Tensor<double, 5U> &_d_rhs) {
// CHECK-NEXT:     return {(bool)1, (bool)0};
// CHECK-NEXT: }

// CHECK: clad::ValueAndPushforward<bool, bool> operator_greater_greater_pushforward(Tensor<double, 5U> &lhs, const Tensor<double, 5U> &rhs, Tensor<double, 5U> &_d_lhs, const Tensor<double, 5U> &_d_rhs) {
// CHECK-NEXT:     return {(bool)1, (bool)0};
// CHECK-NEXT: }

// CHECK: clad::ValueAndPushforward<bool, bool> operator_AmpAmp_pushforward(Tensor<double, 5U> &lhs, const Tensor<double, 5U> &rhs, Tensor<double, 5U> &_d_lhs, const Tensor<double, 5U> &_d_rhs) {
// CHECK-NEXT:     return {(bool)1, (bool)0};
// CHECK-NEXT: }

// CHECK: clad::ValueAndPushforward<bool, bool> operator_pipe_pipe_pushforward(Tensor<double, 5U> &lhs, const Tensor<double, 5U> &rhs, Tensor<double, 5U> &_d_lhs, const Tensor<double, 5U> &_d_rhs) {
// CHECK-NEXT:     return {(bool)1, (bool)0};
// CHECK-NEXT: }

// CHECK: clad::ValueAndPushforward<bool, bool> operator_less_less_equal_pushforward(Tensor<double, 5U> &lhs, const Tensor<double, 5U> &rhs, Tensor<double, 5U> &_d_lhs, const Tensor<double, 5U> &_d_rhs) {
// CHECK-NEXT:     return {(bool)1, (bool)0};
// CHECK-NEXT: }

// CHECK: clad::ValueAndPushforward<bool, bool> operator_greater_greater_equal_pushforward(Tensor<double, 5U> &lhs, const Tensor<double, 5U> &rhs, Tensor<double, 5U> &_d_lhs, const Tensor<double, 5U> &_d_rhs) {
// CHECK-NEXT:     return {(bool)1, (bool)0};
// CHECK-NEXT: }

// CHECK: clad::ValueAndPushforward<bool, bool> operator_amp_pushforward(Tensor<double, 5U> &lhs, const Tensor<double, 5U> &rhs, Tensor<double, 5U> &_d_lhs, const Tensor<double, 5U> &_d_rhs) {
// CHECK-NEXT:     return {(bool)1, (bool)0};
// CHECK-NEXT: }

// CHECK: clad::ValueAndPushforward<bool, bool> operator_pipe_pushforward(Tensor<double, 5U> &lhs, const Tensor<double, 5U> &rhs, Tensor<double, 5U> &_d_lhs, const Tensor<double, 5U> &_d_rhs) {
// CHECK-NEXT:     return {(bool)1, (bool)0};
// CHECK-NEXT: }

// CHECK: clad::ValueAndPushforward<bool, bool> operator_pipe_equal_pushforward(Tensor<double, 5U> &lhs, const Tensor<double, 5U> &rhs, Tensor<double, 5U> &_d_lhs, const Tensor<double, 5U> &_d_rhs) {
// CHECK-NEXT:     return {(bool)1, (bool)0};
// CHECK-NEXT: }

// CHECK: clad::ValueAndPushforward<bool, bool> operator_amp_equal_pushforward(Tensor<double, 5U> &lhs, const Tensor<double, 5U> &rhs, Tensor<double, 5U> &_d_lhs, const Tensor<double, 5U> &_d_rhs) {
// CHECK-NEXT:     return {(bool)1, (bool)0};
// CHECK-NEXT: }
}