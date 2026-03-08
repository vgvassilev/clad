// RUN: %cladclang %s -I%S/../../include -oUserDefinedTypes.out -Xclang -verify 2>&1 | %filecheck %s
// RUN: ./UserDefinedTypes.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr -Xclang -plugin-arg-clad -Xclang -enable-va %s -I%S/../../include -oUserDefinedTypes.out
// RUN: ./UserDefinedTypes.out | %filecheck_exec %s
// XFAIL: valgrind

#include "clad/Differentiator/Differentiator.h"

#include "clad/Differentiator/STLBuiltins.h"

#include <utility>
#include <complex>

#include "../TestUtils.h"
#include "../PrintOverloads.h"

using pairdd = std::pair<double, double>;

double fn1(pairdd p, double i) {
    double res = p.first + 2*p.second + 3*i;
    return res;
}

// CHECK: void fn1_grad(pairdd p, double i, pairdd *_d_p, double *_d_i) {
// CHECK-NEXT:     double _d_res = 0.;
// CHECK-NEXT:     double res = p.first + 2 * p.second + 3 * i;
// CHECK-NEXT:     _d_res += 1;
// CHECK-NEXT:     {
// CHECK-NEXT:         (*_d_p).first += _d_res;
// CHECK-NEXT:         (*_d_p).second += 2 * _d_res;
// CHECK-NEXT:         *_d_i += 3 * _d_res;
// CHECK-NEXT:     }
// CHECK-NEXT: }

struct Tangent {
  Tangent() {}
  double data[5] = {};
  void updateTo(double d) {
    for (int i = 0; i < 5; ++i)
      data[i] = d;
  }

  double someMemFn(double i, double j) {
    return data[0] * i + data[1] * j + 3 * data[2] + data[3] * data[4];
  }
  double someMemFn2(double i, double j) const {
      return data[0]*i + data[1]*i*j;
  }
};

double sum(Tangent& t) {
    double res=0;
    for (int i=0; i<5; ++i)
        res += t.data[i];
    return res;
}

// CHECK: void sum_pullback(Tangent &t, double _d_y, Tangent *_d_t) {
// CHECK-NEXT:     int _d_i = 0;
// CHECK-NEXT:     int i = 0;
// CHECK-NEXT:     double _d_res = 0.;
// CHECK-NEXT:     double res = 0;
// CHECK-NEXT:     unsigned {{int|long|long long}} _t0 = 0;
// CHECK-NEXT:     for (i = 0; i < 5; ++i) {
// CHECK-NEXT:         _t0++;
// CHECK-NEXT:         res += t.data[i];
// CHECK-NEXT:     }
// CHECK-NEXT:     _d_res += _d_y;
// CHECK-NEXT:     for (; _t0; _t0--) {
// CHECK-NEXT:         --i;
// CHECK-NEXT:         double _r_d0 = _d_res;
// CHECK-NEXT:         (*_d_t).data[i] += _r_d0;
// CHECK-NEXT:     }
// CHECK-NEXT: }

double sum(double *data) {
    double res = 0;
    for (int i=0; i<5; ++i)
        res += data[i];
    return res;
}

// CHECK: void sum_pullback(double *data, double _d_y, double *_d_data) {
// CHECK-NEXT:     int _d_i = 0;
// CHECK-NEXT:     int i = 0;
// CHECK-NEXT:     double _d_res = 0.;
// CHECK-NEXT:     double res = 0;
// CHECK-NEXT:     unsigned {{int|long|long long}} _t0 = 0;
// CHECK-NEXT:     for (i = 0; i < 5; ++i) {
// CHECK-NEXT:         _t0++;
// CHECK-NEXT:         res += data[i];
// CHECK-NEXT:     }
// CHECK-NEXT:     _d_res += _d_y;
// CHECK-NEXT:     for (; _t0; _t0--) {
// CHECK-NEXT:         --i;
// CHECK-NEXT:         double _r_d0 = _d_res;
// CHECK-NEXT:         _d_data[i] += _r_d0;
// CHECK-NEXT:     }
// CHECK-NEXT: }

double fn2(Tangent t, double i) {
    double res = sum(t);
    res += sum(t.data) + i + 2*t.data[0];
    return res;
}

// CHECK: void fn2_grad(Tangent t, double i, Tangent *_d_t, double *_d_i) {
// CHECK-NEXT:     double _d_res = 0.;
// CHECK-NEXT:     double res = sum(t);
// CHECK-NEXT:     res += sum(t.data) + i + 2 * t.data[0];
// CHECK-NEXT:     _d_res += 1;
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r_d0 = _d_res;
// CHECK-NEXT:         sum_pullback(t.data, _r_d0, (*_d_t).data);
// CHECK-NEXT:         *_d_i += _r_d0;
// CHECK-NEXT:         (*_d_t).data[0] += 2 * _r_d0;
// CHECK-NEXT:     }
// CHECK-NEXT:     sum_pullback(t, _d_res, _d_t);
// CHECK-NEXT: }

double fn3(double i, double j) {
    Tangent t;
    t.data[0] = 2*i;
    t.data[1] = 5*i + 3*j;
    return sum(t);
}

// CHECK: void fn3_grad(double i, double j, double *_d_i, double *_d_j) {
// CHECK-NEXT:     Tangent t;
// CHECK-NEXT:     Tangent _d_t;
// CHECK-NEXT:     t.data[0] = 2 * i;
// CHECK-NEXT:     t.data[1] = 5 * i + 3 * j;
// CHECK-NEXT:     sum_pullback(t, 1, &_d_t);
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r_d1 = _d_t.data[1];
// CHECK-NEXT:         _d_t.data[1] = 0.;
// CHECK-NEXT:         *_d_i += 5 * _r_d1;
// CHECK-NEXT:         *_d_j += 3 * _r_d1;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r_d0 = _d_t.data[0];
// CHECK-NEXT:         _d_t.data[0] = 0.;
// CHECK-NEXT:         *_d_i += 2 * _r_d0;
// CHECK-NEXT:     }
// CHECK-NEXT: }

double fn4(double i, double j) {
    pairdd p(1, 3);
    pairdd q{7, 5};
    return p.first*i + p.second*j + q.first*i + q.second*j;
}

// CHECK: void fn4_grad(double i, double j, double *_d_i, double *_d_j) {
// CHECK-NEXT:     pairdd p(1, 3);
// CHECK-NEXT:     pairdd _d_p(0, 0);
// CHECK-NEXT:     pairdd q{7, 5};
// CHECK-NEXT:     pairdd _d_q{0, 0};
// CHECK-NEXT:     {
// CHECK-NEXT:         _d_p.first += 1 * i;
// CHECK-NEXT:         *_d_i += p.first * 1;
// CHECK-NEXT:         _d_p.second += 1 * j;
// CHECK-NEXT:         *_d_j += p.second * 1;
// CHECK-NEXT:         _d_q.first += 1 * i;
// CHECK-NEXT:         *_d_i += q.first * 1;
// CHECK-NEXT:         _d_q.second += 1 * j;
// CHECK-NEXT:         *_d_j += q.second * 1;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: void someMemFn_grad(double i, double j, Tangent *_d_this, double *_d_i, double *_d_j) {
// CHECK-NEXT:     {
// CHECK-NEXT:         _d_this->data[0] += 1 * i;
// CHECK-NEXT:         *_d_i += this->data[0] * 1;
// CHECK-NEXT:         _d_this->data[1] += 1 * j;
// CHECK-NEXT:         *_d_j += this->data[1] * 1;
// CHECK-NEXT:         _d_this->data[2] += 3 * 1;
// CHECK-NEXT:         _d_this->data[3] += 1 * this->data[4];
// CHECK-NEXT:         _d_this->data[4] += this->data[3] * 1;
// CHECK-NEXT:     }
// CHECK-NEXT: }

double fn5(const Tangent& t, double i) {
    return t.someMemFn2(i, i);
}

// CHECK: void someMemFn2_pullback(double i, double j, double _d_y, Tangent *_d_this, double *_d_i, double *_d_j) const {
// CHECK-NEXT:     {
// CHECK-NEXT:         _d_this->data[0] += _d_y * i;
// CHECK-NEXT:         *_d_i += this->data[0] * _d_y;
// CHECK-NEXT:         _d_this->data[1] += _d_y * j * i;
// CHECK-NEXT:         *_d_i += this->data[1] * _d_y * j;
// CHECK-NEXT:         *_d_j += this->data[1] * i * _d_y;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: void fn5_grad(const Tangent &t, double i, Tangent *_d_t, double *_d_i) {
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r0 = 0.;
// CHECK-NEXT:         double _r1 = 0.;
// CHECK-NEXT:         t.someMemFn2_pullback(i, i, 1, _d_t, &_r0, &_r1);
// CHECK-NEXT:         *_d_i += _r0;
// CHECK-NEXT:         *_d_i += _r1;
// CHECK-NEXT:     }
// CHECK-NEXT: }

using dcomplex = std::complex<double>;

double fn6(dcomplex c, double i) {
    c.real(5*i);
    double res = c.real() + 3*c.imag() + 6*i;
    res += 4*c.real();
    return res;
}
// CHECK: void real_pullback({{.*}} [[__val:.*]], std{{(::__1)?}}::complex<double> *_d_this, {{.*}} *[[_d___val:[a-zA-Z_]*]]){{.*}} {
// CHECK-NEXT:     double _t0 ={{( __real)?}} this->[[_M_value:.*]];
// CHECK-NEXT:     {{(__real)?}} this->[[_M_value:.*]] = [[__val]];
// CHECK-NEXT:     {
// CHECK-NEXT:         {{(__real)?}} this->[[_M_value:.*]] = _t0;
// CHECK-NEXT:         double _r_d0 ={{( __real)?}} _d_this->[[_M_value]];
// CHECK-NEXT:         {{(__real)?}} _d_this->[[_M_value]] = 0.;
// CHECK-NEXT:         *[[_d___val]] += _r_d0;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: constexpr void real_pullback(double _d_y, std{{(::__1)?}}::complex<double> *_d_this){{.*}} {
// CHECK-NEXT:     {{(__real)?}} _d_this->{{.*}} += _d_y;
// CHECK-NEXT: }

// CHECK: constexpr void imag_pullback(double _d_y, std{{(::__1)?}}::complex<double> *_d_this){{.*}} {
// CHECK-NEXT:     {{(__imag)?}} _d_this->{{.*}} += _d_y;
// CHECK-NEXT: }

// CHECK: void fn6_grad(dcomplex c, double i, dcomplex *_d_c, double *_d_i) {
// CHECK-NEXT:     c.real(5 * i);
// CHECK-NEXT:     double _t0 = c.imag();
// CHECK-NEXT:     double _d_res = 0.;
// CHECK-NEXT:     double res = c.real() + 3 * _t0 + 6 * i;
// CHECK-NEXT:     double _t1 = c.real();
// CHECK-NEXT:     res += 4 * _t1;
// CHECK-NEXT:     _d_res += 1;
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r_d0 = _d_res;
// CHECK-NEXT:         c.real_pullback(4 * _r_d0, _d_c);
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         c.real_pullback(_d_res, _d_c);
// CHECK-NEXT:         c.imag_pullback(3 * _d_res, _d_c);
// CHECK-NEXT:         *_d_i += 6 * _d_res;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r0 = 0.;
// CHECK-NEXT:         c.real_pullback(5 * i, _d_c, &_r0);
// CHECK-NEXT:         *_d_i += 5 * _r0;
// CHECK-NEXT:     }
// CHECK-NEXT: }

double fn7(dcomplex c1, dcomplex c2) {
    c1.real(c2.imag() + 5*c2.real());
    return c1.real() + 3*c1.imag();
}

// CHECK: void fn7_grad(dcomplex c1, dcomplex c2, dcomplex *_d_c1, dcomplex *_d_c2) {
// CHECK-NEXT:     double _t0 = c2.real();
// CHECK-NEXT:     c1.real(c2.imag() + 5 * _t0);
// CHECK-NEXT:     double _t1 = c1.imag();
// CHECK-NEXT:     {
// CHECK-NEXT:         c1.real_pullback(1, _d_c1);
// CHECK-NEXT:         c1.imag_pullback(3 * 1, _d_c1);
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r0 = 0.;
// CHECK-NEXT:         c1.real_pullback(c2.imag() + 5 * _t0, _d_c1, &_r0);
// CHECK-NEXT:         c2.imag_pullback(_r0, _d_c2);
// CHECK-NEXT:         c2.real_pullback(5 * _r0, _d_c2);
// CHECK-NEXT:     }
// CHECK-NEXT: }

double fn8(Tangent t, dcomplex c) {
  t.updateTo(c.real());
  return sum(t);
}

// CHECK: void updateTo_pullback(double d, Tangent *_d_this, double *_d_d) {
// CHECK-NEXT:     int _d_i = 0;
// CHECK-NEXT:     int i = 0;
// CHECK-NEXT:     clad::tape<double> _t1 = {};
// CHECK-NEXT:     unsigned {{int|long|long long}} _t0 = 0;
// CHECK-NEXT:     for (i = 0; i < 5; ++i) {
// CHECK-NEXT:         _t0++;
// CHECK-NEXT:         clad::push(_t1, this->data[i]);
// CHECK-NEXT:         this->data[i] = d;
// CHECK-NEXT:     }
// CHECK-NEXT:     for (; _t0; _t0--) {
// CHECK-NEXT:         --i;
// CHECK-NEXT:         this->data[i] = clad::pop(_t1);
// CHECK-NEXT:         double _r_d0 = _d_this->data[i];
// CHECK-NEXT:         _d_this->data[i] = 0.;
// CHECK-NEXT:         *_d_d += _r_d0;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: void fn8_grad(Tangent t, dcomplex c, Tangent *_d_t, dcomplex *_d_c) {
// CHECK-NEXT:     t.updateTo(c.real());
// CHECK-NEXT:     sum_pullback(t, 1, _d_t);
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r0 = 0.;
// CHECK-NEXT:         t.updateTo_pullback(c.real(), _d_t, &_r0);
// CHECK-NEXT:         c.real_pullback(_r0, _d_c);
// CHECK-NEXT:     }
// CHECK-NEXT: }

double fn9(Tangent t, dcomplex c) {
  double res = 0;
  for (int i=0; i<5; ++i) {
    res += c.real() + 2*c.imag();
  }
  res += sum(t);
  return res;
}

// CHECK: void fn9_grad(Tangent t, dcomplex c, Tangent *_d_t, dcomplex *_d_c) {
// CHECK-NEXT:     int _d_i = 0;
// CHECK-NEXT:     int i = 0;
// CHECK-NEXT:     clad::tape<double> _t1 = {};
// CHECK-NEXT:     double _d_res = 0.;
// CHECK-NEXT:     double res = 0;
// CHECK-NEXT:     unsigned {{int|long|long long}} _t0 = 0;
// CHECK-NEXT:     for (i = 0; i < 5; ++i) {
// CHECK-NEXT:         _t0++;
// CHECK-NEXT:         res += c.real() + 2 * clad::push(_t1, c.imag());
// CHECK-NEXT:     }
// CHECK-NEXT:     res += sum(t);
// CHECK-NEXT:     _d_res += 1;
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r_d1 = _d_res;
// CHECK-NEXT:         sum_pullback(t, _r_d1, _d_t);
// CHECK-NEXT:     }
// CHECK-NEXT:     for (; _t0; _t0--) {
// CHECK-NEXT:         {
// CHECK-NEXT:             double _r_d0 = _d_res;
// CHECK-NEXT:             c.real_pullback(_r_d0, _d_c);
// CHECK-NEXT:             c.imag_pullback(2 * _r_d0, _d_c);
// CHECK-NEXT:         }
// CHECK-NEXT:     }
// CHECK-NEXT: }

template <typename T>
struct A {
  using PtrType = T*;
};

double fn10(double x, double y) {
  A<double>::PtrType ptr = &x;
  ptr[0] += 6;
  return *ptr;
}

// CHECK: void fn10_grad(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:     A<double>::PtrType _d_ptr = _d_x;
// CHECK-NEXT:     A<double>::PtrType ptr = &x;
// CHECK-NEXT:     ptr[0] += 6;
// CHECK-NEXT:     *_d_ptr += 1;
// CHECK-NEXT:     double _r_d0 = _d_ptr[0];
// CHECK-NEXT: }

double operator+(const double& x, const Tangent& t) {
  return x + t.data[0];
}

// CHECK: void operator_plus_pullback(const double &x, const Tangent &t, double _d_y, double *_d_x, Tangent *_d_t) {
// CHECK-NEXT:     {
// CHECK-NEXT:         *_d_x += _d_y;
// CHECK-NEXT:         (*_d_t).data[0] += _d_y;
// CHECK-NEXT:     }
// CHECK-NEXT: }

double fn11(double x, double y) {
  Tangent t;
  t.data[0] = -y;
  return x + t;
}

// CHECK: void fn11_grad(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:     Tangent t;
// CHECK-NEXT:     Tangent _d_t;
// CHECK-NEXT:     t.data[0] = -y;
// CHECK-NEXT:     operator_plus_pullback(x, t, 1, _d_x, &_d_t);
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r_d0 = _d_t.data[0];
// CHECK-NEXT:         _d_t.data[0] = 0.;
// CHECK-NEXT:         *_d_y += -_r_d0;
// CHECK-NEXT:     }
// CHECK-NEXT: }

struct MyStruct{
  double a;
  double b;
}; 

MyStruct fn12(MyStruct s) {  // expected-warning {{clad::gradient only supports differentiation functions of real return types. Return stmt ignored}}
  s = {2 * s.a, 2 * s.b + 2};
  return s;
}

// CHECK: inline constexpr clad::ValueAndAdjoint<MyStruct &, MyStruct &> operator_equal_reverse_forw(MyStruct &&arg, MyStruct *_d_this, MyStruct &&_d_arg, clad::restore_tracker &_tracker0) noexcept {
// CHECK-NEXT:    _tracker0.store(this->a);
// CHECK-NEXT:    this->a = static_cast<MyStruct &&>(arg).a;
// CHECK-NEXT:    _tracker0.store(this->b);
// CHECK-NEXT:    this->b = static_cast<MyStruct &&>(arg).b;
// CHECK-NEXT:    return {*this, *_d_this};
// CHECK-NEXT:}

// CHECK: inline constexpr void operator_equal_pullback(MyStruct &&arg, MyStruct *_d_this, MyStruct *_d_arg) noexcept {
// CHECK-NEXT:    double _t0 = this->a;
// CHECK-NEXT:    this->a = static_cast<MyStruct &&>(arg).a;
// CHECK-NEXT:    double _t1 = this->b;
// CHECK-NEXT:    this->b = static_cast<MyStruct &&>(arg).b;
// CHECK-NEXT:    {
// CHECK-NEXT:        this->b = _t1;
// CHECK-NEXT:        double _r_d1 = _d_this->b;
// CHECK-NEXT:        _d_this->b = 0.;
// CHECK-NEXT:        (*_d_arg).b += _r_d1;
// CHECK-NEXT:    }
// CHECK-NEXT:    {
// CHECK-NEXT:        this->a = _t0;
// CHECK-NEXT:        double _r_d0 = _d_this->a;
// CHECK-NEXT:        _d_this->a = 0.;
// CHECK-NEXT:        (*_d_arg).a += _r_d0;
// CHECK-NEXT:    }
// CHECK-NEXT:}

// CHECK: void fn12_grad(MyStruct s, MyStruct *_d_s) {
// CHECK-NEXT:     clad::restore_tracker _tracker0 = {};
// CHECK-NEXT:     s.operator_equal_reverse_forw({2 * s.a, 2 * s.b + 2}, _d_s, {0., 0.}, _tracker0);
// CHECK-NEXT:    {
// CHECK-NEXT:        _tracker0.restore();
// CHECK-NEXT:        MyStruct _r0 = {0., 0.};
// CHECK-NEXT:        s.operator_equal_pullback({2 * s.a, 2 * s.b + 2}, _d_s, &_r0);
// CHECK-NEXT:        (*_d_s).a += 2 * _r0.a;
// CHECK-NEXT:        (*_d_s).b += 2 * _r0.b;
// CHECK-NEXT:    }
// CHECK-NEXT:}

typedef int Fint;
typedef union Findex
{
    struct
    {
        Fint j, k, l;
    };
    Fint dim[3];
} Findex;

void fn13(double *x, double *y, int size)
{
    Findex p;

    for (p.j = 0; p.j < size; p.j += 1)
    {
        y[p.j] = 2.0 * x[p.j];
    }
}

// CHECK: void fn13_grad_0_1(double *x, double *y, int size, double *_d_x, double *_d_y) {
// CHECK-NEXT: int _d_size = 0;
// CHECK-NEXT: clad::tape<Fint> _t1 = {};
// CHECK-NEXT: Findex _d_p = {};
// CHECK-NEXT: Findex p;
// CHECK-NEXT: unsigned {{int|long|long long}} _t0 = 0;
// CHECK-NEXT: for (p.j = 0; p.j < size; clad::push(_t1, p.j) , (p.j += 1)) {
// CHECK-NEXT:     _t0++;
// CHECK-NEXT:     y[p.j] = 2. * x[p.j];
// CHECK-NEXT: }
// CHECK-NEXT: {
// CHECK-NEXT:     for (; _t0; _t0--) {
// CHECK-NEXT:         {
// CHECK-NEXT:             p.j = clad::pop(_t1);
// CHECK-NEXT:             Fint _r_d1 = _d_p.j;
// CHECK-NEXT:         }
// CHECK-NEXT:         {
// CHECK-NEXT:             double _r_d2 = _d_y[p.j];
// CHECK-NEXT:             _d_y[p.j] = 0.;
// CHECK-NEXT:             _d_x[p.j] += 2. * _r_d2;
// CHECK-NEXT:         }
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         Fint _r_d0 = _d_p.j;
// CHECK-NEXT:        _d_p.j = 0;
// CHECK-NEXT:     }
// CHECK-NEXT: }
// CHECK-NEXT:}

double fn14(double x, double y) {
  MyStruct s = {2 * y, 3 * x + 2};
  return s.a * s.b;
}

// CHECK: void fn14_grad(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:     MyStruct _d_s = {0., 0.};
// CHECK-NEXT:     MyStruct s = {2 * y, 3 * x + 2};
// CHECK-NEXT:     {
// CHECK-NEXT:         _d_s.a += 1 * s.b;
// CHECK-NEXT:         _d_s.b += s.a * 1;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         *_d_y += 2 * _d_s.a;
// CHECK-NEXT:         *_d_x += 3 * _d_s.b;
// CHECK-NEXT:     }
// CHECK-NEXT:}

template <typename T, std::size_t N>
struct SimpleArray {
    T elements[N]; // Aggregate initialization
};

namespace clad {
namespace custom_derivatives {
namespace class_functions {
template<::std::size_t N>
::clad::ValueAndAdjoint<SimpleArray<double, N>, SimpleArray<double, N>>
constructor_reverse_forw(::clad::Tag<SimpleArray<double, N>>) {  // expected-note {{'constructor_reverse_forw<2}}{{' is defined here}}
  SimpleArray<double, N> a;
  SimpleArray<double, N> d_a;
  return {a, d_a};
}
}}}

double fn15(double x, double y) {
  SimpleArray<double, 2> arr; // expected-warning {{'SimpleArray<double, 2>' is aggregate type and its constructor does not require user-defined forward sweep function}}
  return arr.elements[0];
}

// CHECK:void fn15_grad(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:    ::clad::ValueAndAdjoint<SimpleArray<double, {{2U|2UL|2ULL}}>, SimpleArray<double, {{2U|2UL|2ULL}}> > _t0 = clad::custom_derivatives::class_functions::constructor_reverse_forw(clad::Tag<SimpleArray<double, 2> >());
// CHECK-NEXT:    SimpleArray<double, 2> _d_arr = _t0.adjoint;
// CHECK-NEXT:    SimpleArray<double, 2> arr(_t0.value);
// CHECK-NEXT:    _d_arr.elements[0] += 1;
// CHECK-NEXT:}

class SimpleFunctions1 {
public:
  SimpleFunctions1() noexcept : x(0), y(0) {}
  SimpleFunctions1(double px) : x(px), y(0) {}
  SimpleFunctions1(double p_x, double p_y) noexcept : x(p_x), y(p_y) {}
  double x;
  double y;
  double mem_fn_1(double i, double j) { return (x + y) * i + i * j * j; }
  double mem_fn(double i, double j) { return (x + y) * i + i * j; }
  SimpleFunctions1 operator+(const SimpleFunctions1& other) const {
    SimpleFunctions1 res(x + other.x, y + other.y);
    return res;
  }
  SimpleFunctions1 operator*(const SimpleFunctions1& rhs) const {
    return {this->x * rhs.x, this->y * rhs.y};
  }
};

// CHECK: static void constructor_pullback(double p_x, double p_y, SimpleFunctions1 *_d_this, double *_d_p_x, double *_d_p_y) noexcept {
// CHECK-NEXT:    {
// CHECK-NEXT:        *_d_p_y += _d_this->y;
// CHECK-NEXT:        _d_this->y = 0.;
// CHECK-NEXT:    }
// CHECK-NEXT:    {
// CHECK-NEXT:        *_d_p_x += _d_this->x;
// CHECK-NEXT:        _d_this->x = 0.;
// CHECK-NEXT:    }
// CHECK-NEXT:}

// CHECK: static inline constexpr void constructor_pullback(SimpleFunctions1 &&arg, SimpleFunctions1 *_d_this, SimpleFunctions1 *_d_arg) noexcept {
// CHECK-NEXT:      {
// CHECK-NEXT:          (*_d_arg).y += _d_this->y;
// CHECK-NEXT:          _d_this->y = 0.;
// CHECK-NEXT:      }
// CHECK-NEXT:      {
// CHECK-NEXT:          (*_d_arg).x += _d_this->x;
// CHECK-NEXT:          _d_this->x = 0.;
// CHECK-NEXT:      }
// CHECK-NEXT:  }

// CHECK:  void operator_plus_pullback(const SimpleFunctions1 &other, SimpleFunctions1 _d_y, SimpleFunctions1 *_d_this, SimpleFunctions1 *_d_other) const {
// CHECK-NEXT:      SimpleFunctions1 res(this->x + other.x, this->y + other.y);
// CHECK-NEXT:      SimpleFunctions1 _d_res(0., 0.);
// CHECK-NEXT:      SimpleFunctions1::constructor_pullback(std::move(res), &_d_y, &_d_res);
// CHECK-NEXT:      {
// CHECK-NEXT:          double _r0 = 0.;
// CHECK-NEXT:          double _r1 = 0.;
// CHECK-NEXT:          SimpleFunctions1::constructor_pullback(this->x + other.x, this->y + other.y, &_d_res, &_r0, &_r1);
// CHECK-NEXT:          _d_this->x += _r0;
// CHECK-NEXT:          (*_d_other).x += _r0;
// CHECK-NEXT:          _d_this->y += _r1;
// CHECK-NEXT:          (*_d_other).y += _r1;
// CHECK-NEXT:      }
// CHECK-NEXT:  }

double fn16(double i, double j) {
  SimpleFunctions1 obj1(2, 3);
  SimpleFunctions1 obj2(3, 5);
  return (obj1 + obj2).mem_fn_1(i, j);
}

// CHECK: void fn16_grad(double i, double j, double *_d_i, double *_d_j) {
// CHECK-NEXT:    SimpleFunctions1 obj1(2, 3);
// CHECK-NEXT:    SimpleFunctions1 _d_obj1(0, 0);
// CHECK-NEXT:    SimpleFunctions1 obj2(3, 5);
// CHECK-NEXT:    SimpleFunctions1 _d_obj2(0, 0);
// CHECK-NEXT:    {
// CHECK-NEXT:        SimpleFunctions1 _r0 = {};
// CHECK-NEXT:        double _r1 = 0.;
// CHECK-NEXT:        double _r2 = 0.;
// CHECK-NEXT:        (obj1 + obj2).mem_fn_1_pullback(i, j, 1, &_r0, &_r1, &_r2);
// CHECK-NEXT:        obj1.operator_plus_pullback(obj2, _r0, &_d_obj1, &_d_obj2);
// CHECK-NEXT:        *_d_i += _r1;
// CHECK-NEXT:        *_d_j += _r2;
// CHECK-NEXT:    }
// CHECK-NEXT:}

// CHECK: void mem_fn_pullback(double i, double j, double _d_y, SimpleFunctions1 *_d_this, double *_d_i, double *_d_j) {
// CHECK-NEXT:    {
// CHECK-NEXT:        _d_this->x += _d_y * i;
// CHECK-NEXT:        _d_this->y += _d_y * i;
// CHECK-NEXT:        *_d_i += (this->x + this->y) * _d_y;
// CHECK-NEXT:        *_d_i += _d_y * j;
// CHECK-NEXT:        *_d_j += i * _d_y;
// CHECK-NEXT:    }
// CHECK-NEXT:}

double fn17(double i, double j) {
    SimpleFunctions1 sf(3, 5);
    return sf.mem_fn(i, j);
}

// CHECK: void fn17_grad(double i, double j, double *_d_i, double *_d_j) {
// CHECK-NEXT:    SimpleFunctions1 sf(3, 5);
// CHECK-NEXT:    SimpleFunctions1 _d_sf(0, 0);
// CHECK-NEXT:    {
// CHECK-NEXT:        double _r0 = 0.;
// CHECK-NEXT:        double _r1 = 0.;
// CHECK-NEXT:        sf.mem_fn_pullback(i, j, 1, &_d_sf, &_r0, &_r1);
// CHECK-NEXT:        *_d_i += _r0;
// CHECK-NEXT:        *_d_j += _r1;
// CHECK-NEXT:    }
// CHECK-NEXT:}

double fn18(double i, double j) {
    SimpleFunctions1 sf(3 * i, 5 * j);
    return sf.mem_fn(i, j);
}

// CHECK:  void fn18_grad(double i, double j, double *_d_i, double *_d_j) {
// CHECK-NEXT:      SimpleFunctions1 sf(3 * i, 5 * j);
// CHECK-NEXT:      SimpleFunctions1 _d_sf(0., 0.);
// CHECK-NEXT:      {
// CHECK-NEXT:          double _r2 = 0.;
// CHECK-NEXT:          double _r3 = 0.;
// CHECK-NEXT:          sf.mem_fn_pullback(i, j, 1, &_d_sf, &_r2, &_r3);
// CHECK-NEXT:          *_d_i += _r2;
// CHECK-NEXT:          *_d_j += _r3;
// CHECK-NEXT:      }
// CHECK-NEXT:      {
// CHECK-NEXT:          double _r0 = 0.;
// CHECK-NEXT:          double _r1 = 0.;
// CHECK-NEXT:          SimpleFunctions1::constructor_pullback(3 * i, 5 * j, &_d_sf, &_r0, &_r1);
// CHECK-NEXT:          *_d_i += 3 * _r0;
// CHECK-NEXT:          *_d_j += 5 * _r1;
// CHECK-NEXT:      }
// CHECK-NEXT:  }

double fn19(double i, double j) {
    SimpleFunctions1 sf1(3, 5);
    SimpleFunctions1 sf2(i, j);
    return (sf1 * sf2).mem_fn(i, j);
}

// CHECK: void operator_star_pullback(const SimpleFunctions1 &rhs, SimpleFunctions1 _d_y, SimpleFunctions1 *_d_this, SimpleFunctions1 *_d_rhs) const {
// CHECK-NEXT:    {
// CHECK-NEXT:        double _r0 = 0.;
// CHECK-NEXT:        double _r1 = 0.;
// CHECK-NEXT:        SimpleFunctions1::constructor_pullback(this->x * rhs.x, this->y * rhs.y, &_d_y, &_r0, &_r1);
// CHECK-NEXT:        _d_this->x += _r0 * rhs.x;
// CHECK-NEXT:        (*_d_rhs).x += this->x * _r0;
// CHECK-NEXT:        _d_this->y += _r1 * rhs.y;
// CHECK-NEXT:        (*_d_rhs).y += this->y * _r1;
// CHECK-NEXT:    }
// CHECK-NEXT:}

// CHECK:  void fn19_grad(double i, double j, double *_d_i, double *_d_j) {
// CHECK-NEXT:      SimpleFunctions1 sf1(3, 5);
// CHECK-NEXT:      SimpleFunctions1 _d_sf1(0, 0);
// CHECK-NEXT:      SimpleFunctions1 sf2(i, j);
// CHECK-NEXT:      SimpleFunctions1 _d_sf2(0., 0.);
// CHECK-NEXT:      {
// CHECK-NEXT:          SimpleFunctions1 _r2 = {};
// CHECK-NEXT:          double _r3 = 0.;
// CHECK-NEXT:          double _r4 = 0.;
// CHECK-NEXT:          (sf1 * sf2).mem_fn_pullback(i, j, 1, &_r2, &_r3, &_r4);
// CHECK-NEXT:          sf1.operator_star_pullback(sf2, _r2, &_d_sf1, &_d_sf2);
// CHECK-NEXT:          *_d_i += _r3;
// CHECK-NEXT:          *_d_j += _r4;
// CHECK-NEXT:      }
// CHECK-NEXT:      {
// CHECK-NEXT:          double _r0 = 0.;
// CHECK-NEXT:          double _r1 = 0.;
// CHECK-NEXT:          SimpleFunctions1::constructor_pullback(i, j, &_d_sf2, &_r0, &_r1);
// CHECK-NEXT:          *_d_i += _r0;
// CHECK-NEXT:          *_d_j += _r1;
// CHECK-NEXT:      }
// CHECK-NEXT:  }

void fn20(MyStruct s) {
  s = {2 * s.a, 2 * s.b + 2};
}

// CHECK: void fn20_grad(MyStruct s, MyStruct *_d_s) {
// CHECK-NEXT:     clad::restore_tracker _tracker0 = {};
// CHECK-NEXT:     s.operator_equal_reverse_forw({2 * s.a, 2 * s.b + 2}, _d_s, {0., 0.}, _tracker0);
// CHECK-NEXT:    {
// CHECK-NEXT:        _tracker0.restore();
// CHECK-NEXT:        MyStruct _r0 = {0., 0.};
// CHECK-NEXT:        s.operator_equal_pullback({2 * s.a, 2 * s.b + 2}, _d_s, &_r0);
// CHECK-NEXT:        (*_d_s).a += 2 * _r0.a;
// CHECK-NEXT:        (*_d_s).b += 2 * _r0.b;
// CHECK-NEXT:    }
// CHECK-NEXT:}

class Identity {
    public:
    double operator()(double u) { return u + 1; }
};

// CHECK:  void operator_call_pullback(double u, double _d_y, Identity *_d_this, double *_d_u) {
// CHECK-NEXT:    *_d_u += _d_y;
// CHECK-NEXT:}


double fn21(double x, double y) {
    Identity di{};
    double val = di(x);
    return val * val;
}

// CHECK:  void fn21_grad(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:      Identity _d_di = {};
// CHECK-NEXT:      Identity di{};
// CHECK-NEXT:      double _d_val = 0.;
// CHECK-NEXT:      double val = di(x);
// CHECK-NEXT:      {
// CHECK-NEXT:          _d_val += 1 * val;
// CHECK-NEXT:          _d_val += val * 1;
// CHECK-NEXT:      }
// CHECK-NEXT:      {
// CHECK-NEXT:          double _r0 = 0.;
// CHECK-NEXT:          di.operator_call_pullback(x, _d_val, &_d_di, &_r0);
// CHECK-NEXT:          *_d_x += _r0;
// CHECK-NEXT:      }
// CHECK-NEXT:  }

struct StructNoDefConstr {
  StructNoDefConstr(int){}
};

double fn22(double x){
  StructNoDefConstr t{0};
  return x;
}

// CHECK:  void fn22_grad(double x, double *_d_x) {
// CHECK-NEXT:      StructNoDefConstr t(0);
// CHECK-NEXT:      StructNoDefConstr _d_t(0);
// CHECK-NEXT:      *_d_x += 1;
// CHECK-NEXT:  }

class B {
public:
  double data = 0;
};

double add(B b, double u) {
    return b.data + u;
}

// CHECK:  void add_pullback(B b, double u, double _d_y, B *_d_b, double *_d_u) {
// CHECK-NEXT:      {
// CHECK-NEXT:          (*_d_b).data += _d_y;
// CHECK-NEXT:          *_d_u += _d_y;
// CHECK-NEXT:      }
// CHECK-NEXT:  }

double fn23(double u, double v) {
    B b;
    b.data = v;
    double res = 0;
    res = add(b, u);
    return res;
}

// CHECK: static inline constexpr void constructor_pullback(const B &arg, B *_d_this, B *_d_arg) noexcept {
// CHECK-NEXT:      {
// CHECK-NEXT:          (*_d_arg).data += _d_this->data;
// CHECK-NEXT:          _d_this->data = 0.;
// CHECK-NEXT:      }
// CHECK-NEXT:  }

// CHECK:  void fn23_grad(double u, double v, double *_d_u, double *_d_v) {
// CHECK-NEXT:      B _d_b = {0.};
// CHECK-NEXT:      B b;
// CHECK-NEXT:      b.data = v;
// CHECK-NEXT:      double _d_res = 0.;
// CHECK-NEXT:      double res = 0;
// CHECK-NEXT:      res = add(b, u);
// CHECK-NEXT:      _d_res += 1;
// CHECK-NEXT:      {
// CHECK-NEXT:          double _r_d1 = _d_res;
// CHECK-NEXT:          _d_res = 0.;
// CHECK-NEXT:          B _r0 = _d_b;
// CHECK-NEXT:          double _r1 = 0.;
// CHECK-NEXT:          add_pullback(b, u, _r_d1, &_r0, &_r1);
// CHECK-NEXT:          constructor_pullback(b, &_r0, &_d_b);
// CHECK-NEXT:          *_d_u += _r1;
// CHECK-NEXT:      }
// CHECK-NEXT:      {
// CHECK-NEXT:          double _r_d0 = _d_b.data;
// CHECK-NEXT:          _d_b.data = 0.;
// CHECK-NEXT:          *_d_v += _r_d0;
// CHECK-NEXT:      }
// CHECK-NEXT:  }

struct Vector3 {
    double x, y, z;
    Vector3(double px = 0, double py = 0, double pz = 0) : x(px), y(py), z(pz) {}
    Vector3 operator- () {
      return {-x, -y, -z};
    }
};

// CHECK: static void constructor_pullback(double px, double py, double pz, Vector3 *_d_this, double *_d_px, double *_d_py, double *_d_pz) {
// CHECK-NEXT:      {
// CHECK-NEXT:          *_d_pz += _d_this->z;
// CHECK-NEXT:          _d_this->z = 0.;
// CHECK-NEXT:      }
// CHECK-NEXT:      {
// CHECK-NEXT:          *_d_py += _d_this->y;
// CHECK-NEXT:          _d_this->y = 0.;
// CHECK-NEXT:      }
// CHECK-NEXT:      {
// CHECK-NEXT:          *_d_px += _d_this->x;
// CHECK-NEXT:          _d_this->x = 0.;
// CHECK-NEXT:      }
// CHECK-NEXT:  }

// CHECK:  void operator_star_pullback(double a, const Vector3 &v, Vector3 _d_y, double *_d_a, Vector3 *_d_v) {
// CHECK-NEXT:      {
// CHECK-NEXT:          double _r0 = 0.;
// CHECK-NEXT:          double _r1 = 0.;
// CHECK-NEXT:          double _r2 = 0.;
// CHECK-NEXT:          Vector3::constructor_pullback(a * v.x, a * v.y, a * v.z, &_d_y, &_r0, &_r1, &_r2);
// CHECK-NEXT:          *_d_a += _r0 * v.x;
// CHECK-NEXT:          (*_d_v).x += a * _r0;
// CHECK-NEXT:          *_d_a += _r1 * v.y;
// CHECK-NEXT:          (*_d_v).y += a * _r1;
// CHECK-NEXT:          *_d_a += _r2 * v.z;
// CHECK-NEXT:          (*_d_v).z += a * _r2;
// CHECK-NEXT:      }
// CHECK-NEXT:  }

const Vector3 operator*(double a, const Vector3& v) {
  return {a * v.x, a * v.y, a * v.z};
}

double fn24(double x, double y) {
  Vector3 v(x, x, y);
  Vector3 w = 2*v;
  return w.x;
}

// CHECK:  void fn24_grad(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:      Vector3 v(x, x, y);
// CHECK-NEXT:      Vector3 _d_v(0., 0., 0.);
// CHECK-NEXT:      Vector3 w = 2 * v;
// CHECK-NEXT:      Vector3 _d_w;
// CHECK-NEXT:      _d_w.x += 1;
// CHECK-NEXT:      {
// CHECK-NEXT:          double _r3 = 0.;
// CHECK-NEXT:          operator_star_pullback(2, v, _d_w, &_r3, &_d_v);
// CHECK-NEXT:      }
// CHECK-NEXT:      {
// CHECK-NEXT:          double _r0 = 0.;
// CHECK-NEXT:          double _r1 = 0.;
// CHECK-NEXT:          double _r2 = 0.;
// CHECK-NEXT:          Vector3::constructor_pullback(x, x, y, &_d_v, &_r0, &_r1, &_r2);
// CHECK-NEXT:          *_d_x += _r0;
// CHECK-NEXT:          *_d_x += _r1;
// CHECK-NEXT:          *_d_y += _r2;
// CHECK-NEXT:      }
// CHECK-NEXT:  }

// CHECK:  void operator_minus_pullback(Vector3 _d_y, Vector3 *_d_this) {
// CHECK-NEXT:      {
// CHECK-NEXT:          double _r0 = 0.;
// CHECK-NEXT:          double _r1 = 0.;
// CHECK-NEXT:          double _r2 = 0.;
// CHECK-NEXT:          Vector3::constructor_pullback(-this->x, -this->y, -this->z, &_d_y, &_r0, &_r1, &_r2);
// CHECK-NEXT:          _d_this->x += -_r0;
// CHECK-NEXT:          _d_this->y += -_r1;
// CHECK-NEXT:          _d_this->z += -_r2;
// CHECK-NEXT:      }
// CHECK-NEXT:  }

double fn25(double x, double y) {
  Vector3 v{x, x, y};
  Vector3 w = -v;
  return w.x;
}

// CHECK:  void fn25_grad(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:      Vector3 v{x, x, y};
// CHECK-NEXT:      Vector3 _d_v{0., 0., 0.};
// CHECK-NEXT:      Vector3 w = - v;
// CHECK-NEXT:      Vector3 _d_w;
// CHECK-NEXT:      _d_w.x += 1;
// CHECK-NEXT:      v.operator_minus_pullback(_d_w, &_d_v);
// CHECK-NEXT:      {
// CHECK-NEXT:          double _r0 = 0.;
// CHECK-NEXT:          double _r1 = 0.;
// CHECK-NEXT:          double _r2 = 0.;
// CHECK-NEXT:          Vector3::constructor_pullback(x, x, y, &_d_v, &_r0, &_r1, &_r2);
// CHECK-NEXT:          *_d_x += _r0;
// CHECK-NEXT:          *_d_x += _r1;
// CHECK-NEXT:          *_d_y += _r2;
// CHECK-NEXT:      }
// CHECK-NEXT:  }

struct ptrClass {
  double* ptr = nullptr;
  ptrClass() = default;
  ptrClass(double* mptr): ptr(mptr) {}
  double& operator*() {
    return *ptr;
  }
};

namespace clad {
namespace custom_derivatives {
namespace class_functions {
::clad::ValueAndAdjoint<ptrClass, ptrClass>
constructor_reverse_forw(::clad::Tag<ptrClass>, double* mptr, double* d_mptr) elidable_reverse_forw;
}}}

// CHECK:  clad::ValueAndAdjoint<double &, double &> operator_star_reverse_forw(ptrClass *_d_this) {
// CHECK-NEXT:      return {*this->ptr, *_d_this->ptr};
// CHECK-NEXT:  }

// CHECK:  void operator_star_pullback(ptrClass *_d_this) {
// CHECK-NEXT:  }

double fn26(double x, double y) {
  ptrClass p(&x);
  return *p;
}

// CHECK:  void fn26_grad(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:      ptrClass p(&x);
// CHECK-NEXT:      ptrClass _d_p(_d_x);
// CHECK-NEXT:      clad::ValueAndAdjoint<double &, double &> _t0 = p.operator_star_reverse_forw(&_d_p);
// CHECK-NEXT:      _t0.adjoint += 1;
// CHECK-NEXT:  }

struct MyStructWrapper {
  MyStruct val;
};

// CHECK:  inline constexpr clad::ValueAndAdjoint<MyStructWrapper &, MyStructWrapper &> operator_equal_reverse_forw(MyStructWrapper &&arg, MyStructWrapper *_d_this, MyStructWrapper &&_d_arg, clad::restore_tracker &_tracker0) noexcept {
// CHECK-NEXT:      this->val.operator_equal_reverse_forw(static_cast<MyStructWrapper &&>(arg).val, &_d_this->val, std::move(_d_arg.val), _tracker0);
// CHECK-NEXT:      return {*this, *_d_this};
// CHECK-NEXT:  }

// CHECK:  inline constexpr void operator_equal_pullback(MyStructWrapper &&arg, MyStructWrapper *_d_this, MyStructWrapper *_d_arg) noexcept {
// CHECK-NEXT:      clad::restore_tracker _tracker0 = {};
// CHECK-NEXT:      this->val.operator_equal_reverse_forw(static_cast<MyStructWrapper &&>(arg).val, &_d_this->val, std::move((*_d_arg).val), _tracker0);
// CHECK-NEXT:      {
// CHECK-NEXT:          _tracker0.restore();
// CHECK-NEXT:          this->val.operator_equal_pullback(static_cast<MyStructWrapper &&>(arg).val, &_d_this->val, &(*_d_arg).val);
// CHECK-NEXT:      }
// CHECK-NEXT:  }

double fn27(double x, double y) {
  MyStructWrapper s;
  s = {2 * y, 3 * x + 2};
  return s.val.a * s.val.b;
}

// CHECK:  void fn27_grad(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:      MyStructWrapper _d_s = {{.*0., 0..*}};
// CHECK-NEXT:      MyStructWrapper s;
// CHECK-NEXT:      clad::restore_tracker _tracker0 = {};
// CHECK-NEXT:      s.operator_equal_reverse_forw({{.*2 \* y, 3 \* x \+ 2.*}}, &_d_s, {{.*0., 0..*}}, _tracker0);
// CHECK-NEXT:      {
// CHECK-NEXT:          _d_s.val.a += 1 * s.val.b;
// CHECK-NEXT:          _d_s.val.b += s.val.a * 1;
// CHECK-NEXT:      }
// CHECK-NEXT:      {
// CHECK-NEXT:          _tracker0.restore();
// CHECK-NEXT:          MyStructWrapper _r0 = {{.*0., 0..*}};
// CHECK-NEXT:          s.operator_equal_pullback({{.*2 \* y, 3 \* x \+ 2.*}}, &_d_s, &_r0);
// CHECK-NEXT:          *_d_y += 2 * _r0.val.a;
// CHECK-NEXT:          *_d_x += 3 * _r0.val.b;
// CHECK-NEXT:      }
// CHECK-NEXT:  }

double vecSum(const Vector3& v) {
  return v.x + v.y + v.z;
}

// CHECK:  void vecSum_pullback(const Vector3 &v, double _d_y, Vector3 *_d_v) {
// CHECK-NEXT:      {
// CHECK-NEXT:          (*_d_v).x += _d_y;
// CHECK-NEXT:          (*_d_v).y += _d_y;
// CHECK-NEXT:          (*_d_v).z += _d_y;
// CHECK-NEXT:      }
// CHECK-NEXT:  }

double fn28(double x, double y) {
  double z = vecSum({x, y, x});
  return z;
}

// CHECK:  void fn28_grad(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:      double _d_z = 0.;
// CHECK-NEXT:      double z = vecSum({x, y, x});
// CHECK-NEXT:      _d_z += 1;
// CHECK-NEXT:      {
// CHECK-NEXT:          Vector3 _r0 = {0., 0., 0.};
// CHECK-NEXT:          vecSum_pullback({x, y, x}, _d_z, &_r0);
// CHECK-NEXT:          double _r1 = 0.;
// CHECK-NEXT:          double _r2 = 0.;
// CHECK-NEXT:          double _r3 = 0.;
// CHECK-NEXT:          Vector3::constructor_pullback(x, y, x, &_r0, &_r1, &_r2, &_r3);
// CHECK-NEXT:          *_d_x += _r1;
// CHECK-NEXT:          *_d_y += _r2;
// CHECK-NEXT:          *_d_x += _r3;
// CHECK-NEXT:      }
// CHECK-NEXT:  }

struct Session {
   float factors[5]{9., 8., 7., 6., 5.};
};

float fn29(float *input, Session const *session) {
   Session const &sess = session[0];
   float buffer[5]{0., 0., 0., 0., 0};
   const size_t n = 5;
   for (size_t id = 0; id < n; id++) {
      buffer[id] = sess.factors[id] * input[id];
   }
   float out = 0.0;
   for (size_t id = 0; id < n; id++) {
      out += input[id];
   }
   return out; // sum(input)
}

// CHECK:  void fn29_grad_0(float *input, const Session *session, float *_d_input) {
// CHECK-NEXT:      size_t _d_id = {{0U|0UL|0ULL}};
// CHECK-NEXT:      size_t id = {{0U|0UL|0ULL}};
// CHECK-NEXT:      size_t _d_id0 = {{0U|0UL|0ULL}};
// CHECK-NEXT:      size_t id0 = {{0U|0UL|0ULL}};
// CHECK-NEXT:      const Session &sess = session[0];
// CHECK-NEXT:      float _d_buffer[5] = {0};
// CHECK-NEXT:      float buffer[5]{0., 0., 0., 0., 0};
// CHECK-NEXT:      size_t _d_n = {{0U|0UL|0ULL}};
// CHECK-NEXT:      const size_t n = 5;
// CHECK-NEXT:      unsigned {{int|long|long long}} _t0 = 0;
// CHECK-NEXT:      for (id = 0; id < n; id++) {
// CHECK-NEXT:          _t0++;
// CHECK-NEXT:          buffer[id] = sess.factors[id] * input[id];
// CHECK-NEXT:      }
// CHECK-NEXT:      float _d_out = 0.F;
// CHECK-NEXT:      float out = 0.;
// CHECK-NEXT:      unsigned {{int|long|long long}} _t1 = 0;
// CHECK-NEXT:      for (id0 = 0; id0 < n; id0++) {
// CHECK-NEXT:          _t1++;
// CHECK-NEXT:          out += input[id0];
// CHECK-NEXT:      }
// CHECK-NEXT:      _d_out += 1;
// CHECK-NEXT:      for (; _t1; _t1--) {
// CHECK-NEXT:          id0--;
// CHECK-NEXT:          {
// CHECK-NEXT:              float _r_d1 = _d_out;
// CHECK-NEXT:              _d_input[id0] += _r_d1;
// CHECK-NEXT:          }
// CHECK-NEXT:      }
// CHECK-NEXT:      for (; _t0; _t0--) {
// CHECK-NEXT:          id--;
// CHECK-NEXT:          {
// CHECK-NEXT:              float _r_d0 = _d_buffer[id];
// CHECK-NEXT:              _d_buffer[id] = 0.F;
// CHECK-NEXT:              _d_input[id] += sess.factors[id] * _r_d0;
// CHECK-NEXT:          }
// CHECK-NEXT:      }
// CHECK-NEXT:  }

MyStruct& objRef(MyStruct& s) {
  return s;
}

// CHECK:  clad::ValueAndAdjoint<MyStruct &, MyStruct &> objRef_reverse_forw(MyStruct &s, MyStruct &_d_s) {
// CHECK-NEXT:      return {s, _d_s};
// CHECK-NEXT:  }

// CHECK:  void objRef_pullback(MyStruct &s, MyStruct *_d_s) {
// CHECK-NEXT:  }

double fn30(double x, double y) {
  MyStruct a{x, y};
  return objRef(a).b;
}

// CHECK:  void fn30_grad(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:      MyStruct _d_a = {0., 0.};
// CHECK-NEXT:      MyStruct a{x, y};
// CHECK-NEXT:      clad::ValueAndAdjoint<MyStruct &, MyStruct &> _t0 = objRef_reverse_forw(a, _d_a);
// CHECK-NEXT:      _t0.adjoint.b += 1;
// CHECK-NEXT:      {
// CHECK-NEXT:          *_d_x += _d_a.a;
// CHECK-NEXT:          *_d_y += _d_a.b;
// CHECK-NEXT:      }
// CHECK-NEXT:  }

struct PtrAndValAggr {
   double x;
   double* ptr;
};

double fn31(double x, double y) {
  PtrAndValAggr s = {x, &y};
  return s.x + *s.ptr;
}

// CHECK:  void fn31_grad(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:      PtrAndValAggr _d_s = {0., _d_y};
// CHECK-NEXT:      PtrAndValAggr s = {x, &y};
// CHECK-NEXT:      {
// CHECK-NEXT:          _d_s.x += 1;
// CHECK-NEXT:          *_d_s.ptr += 1;
// CHECK-NEXT:      }
// CHECK-NEXT:      *_d_x += _d_s.x;
// CHECK-NEXT:  }

double fn32(double x, double y) {
  PtrAndValAggr s[2] = {{x, &x}, {y, &y}};
  return s[0].x + *s[1].ptr;
}

// CHECK:  void fn32_grad(double x, double y, double *_d_x, double *_d_y) {
//// Note: {{ is represented with {{[{][{]}} because of regex
// CHECK-NEXT:      PtrAndValAggr _d_s[2] = {{[{][{]}}0., _d_x}, {0., _d_y{{[}][}]}};
// CHECK-NEXT:      PtrAndValAggr s[2] = {{[{][{]}}x, &x}, {y, &y{{[}][}]}};
// CHECK-NEXT:      {
// CHECK-NEXT:          _d_s[0].x += 1;
// CHECK-NEXT:          *_d_s[1].ptr += 1;
// CHECK-NEXT:      }
// CHECK-NEXT:      {
// CHECK-NEXT:          *_d_x += _d_s[0].x;
// CHECK-NEXT:          *_d_y += _d_s[1].x;
// CHECK-NEXT:      }
// CHECK-NEXT:  }

class PIBase {
  double pi = 3.14;
public:
  double getPi() {return pi; }
};
struct PI : public PIBase {
  double getFourPi() {
    double tmp1 = static_cast<PIBase*>(this)->getPi();
    double tmp2 = reinterpret_cast<PIBase*>(this)->getPi();
    double tmp3 = dynamic_cast<PIBase*>(this)->getPi();
    double tmp4 = ((PIBase*)this)->getPi();
    return 2 * (tmp1 + tmp2 + tmp3 + tmp4);
  }
};
double fn33(double x) {
  PI pi;
  return x + pi.getFourPi();
}

// CHECK: void getPi_pullback(double _d_y, PIBase *_d_this) {
// CHECK-NEXT:    _d_this->pi += _d_y;
// CHECK-NEXT:}
// CHECK-NEXT: void getFourPi_pullback(double _d_y, PI *_d_this) {
// CHECK-NEXT:     double _d_tmp1 = 0.;
// CHECK-NEXT:     double tmp1 = static_cast<PIBase *>(this)->getPi();
// CHECK-NEXT:     double _d_tmp2 = 0.;
// CHECK-NEXT:     double tmp2 = reinterpret_cast<PIBase *>(this)->getPi();
// CHECK-NEXT:     double _d_tmp3 = 0.;
// CHECK-NEXT:     double tmp3 = dynamic_cast<PIBase *>(this)->getPi();
// CHECK-NEXT:     double _d_tmp4 = 0.;
// CHECK-NEXT:     double tmp4 = ((PIBase *)this)->getPi();
// CHECK-NEXT:     {
// CHECK-NEXT:         _d_tmp1 += 2 * _d_y;
// CHECK-NEXT:         _d_tmp2 += 2 * _d_y;
// CHECK-NEXT:         _d_tmp3 += 2 * _d_y;
// CHECK-NEXT:         _d_tmp4 += 2 * _d_y;
// CHECK-NEXT:     }
// CHECK-NEXT:     (PIBase *)this->getPi_pullback(_d_tmp4, (PIBase *)_d_this);
// CHECK-NEXT:     dynamic_cast<PIBase *>(this)->getPi_pullback(_d_tmp3, _d_this);
// CHECK-NEXT:     reinterpret_cast<PIBase *>(this)->getPi_pullback(_d_tmp2, _d_this);
// CHECK-NEXT:     static_cast<PIBase *>(this)->getPi_pullback(_d_tmp1, _d_this);
// CHECK-NEXT: }
// CHECK-NEXT:void fn33_grad(double x, double *_d_x) {
// CHECK-NEXT:    PI _d_pi = {};
// CHECK-NEXT:    PI pi;
// CHECK-NEXT:    {
// CHECK-NEXT:        *_d_x += 1;
// CHECK-NEXT:        pi.getFourPi_pullback(1, &_d_pi);
// CHECK-NEXT:    }
// CHECK-NEXT:}

struct otherStruct {
  double val;
  double* ptr;
};

struct structToConvert {
  double data = 0;
  operator double& () {
    return data;
  }

  // CHECK: clad::ValueAndAdjoint<otherStruct, otherStruct> conversion_operator_reverse_forw(clad::Tag<otherStruct>, structToConvert *_d_this) {
  // CHECK-NEXT:    return {{[{][{]}}2 * this->data, &this->data}, {0., &_d_this->data{{[}][}]}};
  // CHECK-NEXT:}

  // CHECK: void conversion_operator_pullback(otherStruct _d_y, structToConvert *_d_this) {
  // CHECK-NEXT:    _d_this->data += 2 * _d_y.val;
  // CHECK-NEXT:}

  operator otherStruct () {
    return {2 * data, &data};
  }

  // CHECK: clad::ValueAndAdjoint<double &, double &> conversion_operator_reverse_forw(clad::Tag<double &>, structToConvert *_d_this) {
  // CHECK-NEXT:    return {this->data, _d_this->data};
  // CHECK-NEXT:}

  // CHECK: void conversion_operator_pullback(structToConvert *_d_this) {
  // CHECK-NEXT:}
};

double fn34(double x, double y) {
  structToConvert obj_x{x};
  structToConvert obj_y{y};
  otherStruct conv = (otherStruct)obj_y;
  return (obj_x + 1.) + conv.val;
} // x + 1. + 2 * y

// CHECK: void fn34_grad(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:    structToConvert _d_obj_x = {0.};
// CHECK-NEXT:    structToConvert obj_x{x};
// CHECK-NEXT:    structToConvert _d_obj_y = {0.};
// CHECK-NEXT:    structToConvert obj_y{y};
// CHECK-NEXT:    clad::ValueAndAdjoint<otherStruct, otherStruct> _t0 = obj_y.conversion_operator_reverse_forw(clad::Tag<otherStruct>(), &_d_obj_y);
// CHECK-NEXT:    otherStruct _d_conv = (otherStruct)_t0.adjoint;
// CHECK-NEXT:    otherStruct conv = (otherStruct)_t0.value;
// CHECK-NEXT:    clad::ValueAndAdjoint<double &, double &> _t1 = obj_x.conversion_operator_reverse_forw(clad::Tag<double &>(), &_d_obj_x);
// CHECK-NEXT:    {
// CHECK-NEXT:        _t1.adjoint += 1;
// CHECK-NEXT:        _d_conv.val += 1;
// CHECK-NEXT:    }
// CHECK-NEXT:    obj_y.conversion_operator_pullback(_d_conv, &_d_obj_y);
// CHECK-NEXT:    *_d_y += _d_obj_y.data;
// CHECK-NEXT:    *_d_x += _d_obj_x.data;
// CHECK-NEXT:}

double fn35(double x, double y) {
    double& ref = *(PtrAndValAggr){x, &y}.ptr;
    return ref;
}

// CHECK:  void fn35_grad(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:      double &_d_ref = *(PtrAndValAggr){0., _d_y}.ptr;
// CHECK-NEXT:      double &ref = *(PtrAndValAggr){x, &y}.ptr;
// CHECK-NEXT:      _d_ref += 1;
// CHECK-NEXT:  }

class PrivClass {
private:
  size_t n_;
public:
  PrivClass(size_t n) : n_(n) {}

  double multiply(double x) {
    return n_*x;
  }
};

// CHECK:  static void constructor_pullback(size_t n, PrivClass *_d_this, size_t *_d_n) {
// CHECK-NEXT:      {
// CHECK-NEXT:          *_d_n += _d_this->n_;
// CHECK-NEXT:          _d_this->n_ = {{0U|0UL|0ULL}};
// CHECK-NEXT:      }
// CHECK-NEXT:  }

// CHECK:  void multiply_pullback(double x, double _d_y, PrivClass *_d_this, double *_d_x) {
// CHECK-NEXT:      {
// CHECK-NEXT:          _d_this->n_ += _d_y * x;
// CHECK-NEXT:          *_d_x += this->n_ * _d_y;
// CHECK-NEXT:      }
// CHECK-NEXT:  }

double fn36(double x, size_t n){
  PrivClass E(n);
  return E.multiply(x); // (x*n)
}

// CHECK:  void fn36_grad_0(double x, size_t n, double *_d_x) {
// CHECK-NEXT:      size_t _d_n = {{0U|0UL|0ULL}};
// CHECK-NEXT:      PrivClass E(n);
// CHECK-NEXT:      PrivClass _d_E(0{{.*}});
// CHECK-NEXT:      {
// CHECK-NEXT:          double _r1 = 0.;
// CHECK-NEXT:          E.multiply_pullback(x, 1, &_d_E, &_r1);
// CHECK-NEXT:          *_d_x += _r1;
// CHECK-NEXT:      }
// CHECK-NEXT:      {
// CHECK-NEXT:          size_t _r0 = {{0U|0UL|0ULL}};
// CHECK-NEXT:          PrivClass::constructor_pullback(n, &_d_E, &_r0);
// CHECK-NEXT:          _d_n += _r0;
// CHECK-NEXT:      }
// CHECK-NEXT:  }
void print(const Tangent& t) {
  for (int i = 0; i < 5; ++i) {
    printf("%.2f", t.data[i]);
    if (i != 4)
      printf(", ");
  }
}

void print(const MyStruct& s) {
  printf("{%.2f, %.2f}\n", s.a, s.b);
}

template <typename T>
void printArray(T* arr, int size) {
  printf("{");
  for (int i = 0; i < size; ++i) {
    printf("%.2f", arr[i]);
    if (i != size - 1)
      printf(", ");
  }
  printf("}\n");
}

int main() {
    pairdd p(3, 5), d_p;
    double i = 3, d_i, d_j;
    Tangent t, d_t;
    dcomplex c1, c2, d_c1, d_c2;
    auto memFn1 = &Tangent::someMemFn;

    INIT_GRADIENT(fn1);
    INIT_GRADIENT(fn2);
    INIT_GRADIENT(fn3);
    INIT_GRADIENT(fn4);
    INIT_GRADIENT(memFn1);
    INIT_GRADIENT(fn5);
    INIT_GRADIENT(fn6);
    INIT_GRADIENT(fn7);
    INIT_GRADIENT(fn8);
    INIT_GRADIENT(fn9);
    INIT_GRADIENT(fn10);
    INIT_GRADIENT(fn11);
    
    TEST_GRADIENT(fn1, /*numOfDerivativeArgs=*/2, p, i, &d_p, &d_i);    // CHECK-EXEC: {1.00, 2.00, 3.00}
    TEST_GRADIENT(fn2, /*numOfDerivativeArgs=*/2, t, i, &d_t, &d_i);    // CHECK-EXEC: {4.00, 2.00, 2.00, 2.00, 2.00, 1.00}
    TEST_GRADIENT(fn3, /*numOfDerivativeArgs=*/2, 3, 5, &d_i, &d_j);    // CHECK-EXEC: {7.00, 3.00}
    TEST_GRADIENT(fn4, /*numOfDerivativeArgs=*/2, 3, 5, &d_i, &d_j);    // CHECK-EXEC: {8.00, 8.00}
    t.updateTo(5);
    TEST_GRADIENT(memFn1, /*numOfDerivativeArgs=*/3, t, 3, 5, &d_t, &d_i, &d_j);   // CHECK-EXEC: {3.00, 5.00, 3.00, 5.00, 5.00, 5.00, 5.00}
    t.updateTo(5);
    TEST_GRADIENT(fn5, /*numOfDerivativeArgs=*/2, t, 3, &d_t, &d_i);    // CHECK-EXEC: {3.00, 9.00, 0.00, 0.00, 0.00, 35.00}
    TEST_GRADIENT(fn6, /*numOfDerivativeArgs=*/2, c1, 3, &d_c1, &d_i);  // CHECK-EXEC: {0.00, 3.00, 31.00}
    TEST_GRADIENT(fn7, /*numOfDerivativeArgs=*/2, c1, c2, &d_c1, &d_c2);// CHECK-EXEC: {0.00, 3.00, 5.00, 1.00}
    TEST_GRADIENT(fn8, /*numOfDerivativeArgs=*/2, t, c1, &d_t, &d_c1);  // CHECK-EXEC: {0.00, 0.00, 0.00, 0.00, 0.00, 5.00, 0.00}
    TEST_GRADIENT(fn9, /*numOfDerivativeArgs=*/2, t, c1, &d_t, &d_c1);  // CHECK-EXEC: {1.00, 1.00, 1.00, 1.00, 1.00, 5.00, 10.00}
    TEST_GRADIENT(fn10, /*numOfDerivativeArgs=*/2, 5, 10, &d_i, &d_j);  // CHECK-EXEC: {1.00, 0.00}
    TEST_GRADIENT(fn11, /*numOfDerivativeArgs=*/2, 3, -14, &d_i, &d_j);  // CHECK-EXEC: {1.00, -1.00}
    MyStruct s = {1.0, 2.0}, d_s = {1.0, 1.0};
    auto fn12_test = clad::gradient(fn12); // expected-note {{use clad::jacobian to compute derivatives of multiple real outputs w.r.t. multiple real inputs}}
    fn12_test.execute(s, &d_s);
    print(d_s); // CHECK-EXEC: {2.00, 2.00}

    auto fn13_test = clad::gradient(fn13, "x, y");
    double x[3] = {1.0, 2.0, 3.0}, y[3] = {0.0, 0.0, 0.0};
    double d_x[3] = {0.0, 0.0, 0.0}, d_y[3] = {1.0, 1.0, 1.0};
    int size = 3;
    fn13_test.execute(x, y, 3, d_x, d_y);
    printArray(d_x, size); // CHECK-EXEC: {2.00, 2.00, 2.00}
    printArray(d_y, size); // CHECK-EXEC: {0.00, 0.00, 0.00}

    INIT_GRADIENT(fn14);
    TEST_GRADIENT(fn14, /*numOfDerivativeArgs=*/2, 3, 5, &d_i, &d_j);    // CHECK-EXEC: {30.00, 22.00}

    INIT_GRADIENT(fn15);
    
    INIT_GRADIENT(fn16);
    TEST_GRADIENT(fn16, /*numOfDerivativeArgs=*/2, 2, 3, &d_i, &d_j);    // CHECK-EXEC: {22.00, 12.00}
    
    INIT_GRADIENT(fn17);
    TEST_GRADIENT(fn17, /*numOfDerivativeArgs=*/2, 2, 3, &d_i, &d_j);    // CHECK-EXEC: {11.00, 2.00}
    
    INIT_GRADIENT(fn18);
    TEST_GRADIENT(fn18, /*numOfDerivativeArgs=*/2, 2, 3, &d_i, &d_j);    // CHECK-EXEC: {30.00, 12.00}
    
    INIT_GRADIENT(fn19);
    TEST_GRADIENT(fn19, /*numOfDerivativeArgs=*/2, 2, 3, &d_i, &d_j);    // CHECK-EXEC: {30.00, 12.00}

    s = {1.0, 2.0}, d_s = {1.0, 1.0};
    auto fn20_test = clad::gradient(fn20);
    fn20_test.execute(s, &d_s);
    print(d_s); // CHECK-EXEC: {2.00, 2.00}

    INIT_GRADIENT(fn21);
    TEST_GRADIENT(fn21, /*numOfDerivativeArgs=*/2, 3, 2, &d_i, &d_j);    // CHECK-EXEC: {8.00, 0.00}

    INIT_GRADIENT(fn22);

    INIT_GRADIENT(fn23);
    TEST_GRADIENT(fn23, /*numOfDerivativeArgs=*/2, 3, 2, &d_i, &d_j);    // CHECK-EXEC: {1.00, 1.00}


    INIT_GRADIENT(fn24);
    TEST_GRADIENT(fn24, /*numOfDerivativeArgs=*/2, 2, 3, &d_i, &d_j);    // CHECK-EXEC: {2.00, 0.00}

    INIT_GRADIENT(fn25);
    TEST_GRADIENT(fn25, /*numOfDerivativeArgs=*/2, 2, 3, &d_i, &d_j);    // CHECK-EXEC: {-1.00, 0.00}

    INIT_GRADIENT(fn26);
    TEST_GRADIENT(fn26, /*numOfDerivativeArgs=*/2, 2, 3, &d_i, &d_j);    // CHECK-EXEC: {1.00, 0.00}

    INIT_GRADIENT(fn27);
    TEST_GRADIENT(fn27, /*numOfDerivativeArgs=*/2, 3, 5, &d_i, &d_j);    // CHECK-EXEC: {30.00, 22.00}

    INIT_GRADIENT(fn28);
    TEST_GRADIENT(fn28, /*numOfDerivativeArgs=*/2, 3, 5, &d_i, &d_j);    // CHECK-EXEC: {2.00, 1.00}

    auto fn29_grad = clad::gradient(fn29, "input");
    Session session;
    float input[]{3., 4., 1., 2., 5};
    float dout[]{0., 0., 0., 0., 0};
    fn29_grad.execute(input, &session, dout);
    printArray(dout, 5);  // CHECK-EXEC: {1.00, 1.00, 1.00, 1.00, 1.00}

    INIT_GRADIENT(fn30);
    TEST_GRADIENT(fn30, /*numOfDerivativeArgs=*/2, 3, 5, &d_i, &d_j);    // CHECK-EXEC: {0.00, 1.00}

    INIT_GRADIENT(fn31);
    TEST_GRADIENT(fn31, /*numOfDerivativeArgs=*/2, 3, 5, &d_i, &d_j);    // CHECK-EXEC: {1.00, 1.00}

    INIT_GRADIENT(fn32);
    TEST_GRADIENT(fn32, /*numOfDerivativeArgs=*/2, 3, 5, &d_i, &d_j);    // CHECK-EXEC: {1.00, 1.00}

    INIT_GRADIENT(fn33);
    TEST_GRADIENT(fn33, /*numOfDerivativeArgs=*/1, 3, &d_i);    // CHECK-EXEC: {1.00}

    INIT_GRADIENT(fn34);
    TEST_GRADIENT(fn34, /*numOfDerivativeArgs=*/2, -5, 6, &d_i, &d_j);    // CHECK-EXEC: {1.00, 2.00}

    INIT_GRADIENT(fn35);
    TEST_GRADIENT(fn35, /*numOfDerivativeArgs=*/2, -5, 6, &d_i, &d_j);    // CHECK-EXEC: {0.00, 1.00}

    auto fn36_grad = clad::gradient(fn36, "x");
    d_i = 0;
    fn36_grad.execute(i, 5, &d_i);
    printf("{%.2f}\n", d_i);    // CHECK-EXEC: {5.00}
}
