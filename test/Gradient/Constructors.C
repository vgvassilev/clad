// RUN: %cladclang %s -I%S/../../include -oConstructors.out -Xclang -verify 2>&1 | %filecheck %s
// RUN: ./Constructors.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr -Xclang -plugin-arg-clad -Xclang -enable-va %s -I%S/../../include -oConstructors.out
// RUN: ./Constructors.out | %filecheck_exec %s
// XFAIL: valgrind

#include "clad/Differentiator/Differentiator.h"
#include "clad/Differentiator/STLBuiltins.h"

#include <utility>
#include <complex>

#include "../TestUtils.h"
#include "../PrintOverloads.h"


struct argByVal {
    double x, y;
    argByVal(double val) {
        x = val;
        val *= val;
        y = val;
    }
};

double fn1(double x, double y) {
    argByVal g(x);
    y = x;
    return y + g.y;
} // x + x^2

// CHECK: static clad::ValueAndAdjoint<argByVal, argByVal> constructor_reverse_forw(clad::Tag<argByVal>, double val, double _d_val) {
// CHECK-NEXT:     argByVal *_this = (argByVal *)malloc(sizeof(argByVal));
// CHECK-NEXT:     argByVal *_d_this = (argByVal *)malloc(sizeof(argByVal));
// CHECK-NEXT:     memset(_d_this, 0, sizeof(argByVal));
// CHECK-NEXT:     _this->x = val;
// CHECK-NEXT:     val *= val;
// CHECK-NEXT:     _this->y = val;
// CHECK-NEXT:     return {*_this, *_d_this};
// CHECK-NEXT: }

// CHECK:  static void constructor_pullback(double val, argByVal *_d_this, double *_d_val) {
// CHECK-NEXT:      argByVal *_this = (argByVal *)malloc(sizeof(argByVal));
// CHECK-NEXT:      _this->x = val;
// CHECK-NEXT:      double _t0 = val;
// CHECK-NEXT:      val *= val;
// CHECK-NEXT:      _this->y = val;
// CHECK-NEXT:      {
// CHECK-NEXT:          double _r_d2 = _d_this->y;
// CHECK-NEXT:          _d_this->y = 0.;
// CHECK-NEXT:          *_d_val += _r_d2;
// CHECK-NEXT:      }
// CHECK-NEXT:      {
// CHECK-NEXT:          val = _t0;
// CHECK-NEXT:          double _r_d1 = *_d_val;
// CHECK-NEXT:          *_d_val = 0.;
// CHECK-NEXT:          *_d_val += _r_d1 * val;
// CHECK-NEXT:          *_d_val += val * _r_d1;
// CHECK-NEXT:      }
// CHECK-NEXT:      {
// CHECK-NEXT:          double _r_d0 = _d_this->x;
// CHECK-NEXT:          _d_this->x = 0.;
// CHECK-NEXT:          *_d_val += _r_d0;
// CHECK-NEXT:      }
// CHECK-NEXT:      free(_this);
// CHECK-NEXT:  }

// CHECK:  void fn1_grad(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:      clad::ValueAndAdjoint<argByVal, argByVal> _t0 = argByVal::constructor_reverse_forw(clad::Tag<argByVal>(), x, 0.);
// CHECK-NEXT:      argByVal g(_t0.value);
// CHECK-NEXT:      argByVal _d_g(_t0.adjoint);
// CHECK-NEXT:      y = x;
// CHECK-NEXT:      {
// CHECK-NEXT:          *_d_y += 1;
// CHECK-NEXT:          _d_g.y += 1;
// CHECK-NEXT:      }
// CHECK-NEXT:      {
// CHECK-NEXT:          *_d_x += *_d_y;
// CHECK-NEXT:          *_d_y = 0.;
// CHECK-NEXT:      }
// CHECK-NEXT:      {
// CHECK-NEXT:          double _r0 = 0.;
// CHECK-NEXT:          argByVal::constructor_pullback(x, &_d_g, &_r0);
// CHECK-NEXT:          *_d_x += _r0;
// CHECK-NEXT:      }
// CHECK-NEXT:  }

struct S1{
  double p;
  double d;
  S1(double x) : p(x), d([](){return 12.;}()) {} // expected-warning {{direct lambda calls are not supported, ignored}} // expected-warning {{direct lambda calls are not supported, ignored}}
};

struct S2{
  double p;
  double i;
  double d;
  S2(double x) : p(x), i(1.), d([&](){i *= 32; return 12.;}()) {} // expected-warning {{direct lambda calls are not supported, ignored}} // expected-warning {{direct lambda calls are not supported, ignored}}
};

struct S3{
  double p;
  S3(double x) {
    p = x * x;
  }
};

struct S4{
  double i = 9;
  double p;
  S4(double x) : p(x) {}
};

struct S5{
  double i;
  S5(double x) 
    try { // expected-warning {{statement kind 'CXXTryStmt' is not supported}} // expected-warning {{statement kind 'CXXTryStmt' is not supported}}
      i = x;
    } catch(...) {
      printf("caught\n");
    }
};

double fn2(double u, double v) {
  S1 s1(u);
  S2 s2(v);
  S3 s3(v);
  S5 s5(u);
  return 1;
}

// CHECK:  static clad::ValueAndAdjoint<S1, S1> constructor_reverse_forw(clad::Tag<S1>, double x, double _d_x) {
// CHECK-NEXT:      S1 *_this = (S1 *)malloc(sizeof(S1));
// CHECK-NEXT:      S1 *_d_this = (S1 *)malloc(sizeof(S1));
// CHECK-NEXT:      memset(_d_this, 0, sizeof(S1));
// CHECK-NEXT:      _this->p = x;
// CHECK-NEXT:      _this->d = 0.;
// CHECK-NEXT:      return {*_this, *_d_this};
// CHECK-NEXT:  }

// CHECK:  static void constructor_pullback(double x, S1 *_d_this, double *_d_x) {
// CHECK-NEXT:      _d_this->d = 0.;
// CHECK-NEXT:      {
// CHECK-NEXT:          *_d_x += _d_this->p;
// CHECK-NEXT:          _d_this->p = 0.;
// CHECK-NEXT:      }
// CHECK-NEXT:  }

// CHECK: static clad::ValueAndAdjoint<S2, S2> constructor_reverse_forw(clad::Tag<S2>, double x, double _d_x) {
// CHECK-NEXT:     S2 *_this = (S2 *)malloc(sizeof(S2));
// CHECK-NEXT:     S2 *_d_this = (S2 *)malloc(sizeof(S2));
// CHECK-NEXT:     memset(_d_this, 0, sizeof(S2));
// CHECK-NEXT:     _this->p = x;
// CHECK-NEXT:     _this->i = 1.;
// CHECK-NEXT:     _this->d = 0.;
// CHECK-NEXT:     return {*_this, *_d_this};
// CHECK-NEXT: }

// CHECK:  static void constructor_pullback(double x, S2 *_d_this, double *_d_x) {
// CHECK-NEXT:      S2 *_this = (S2 *)malloc(sizeof(S2));
// CHECK-NEXT:      _this->p = x;
// CHECK-NEXT:      _this->i = 1.;
// CHECK-NEXT:      _this->d = 0.;
// CHECK-NEXT:      _d_this->d = 0.;
// CHECK-NEXT:      _d_this->i = 0.;
// CHECK-NEXT:      {
// CHECK-NEXT:          *_d_x += _d_this->p;
// CHECK-NEXT:          _d_this->p = 0.;
// CHECK-NEXT:      }
// CHECK-NEXT:      free(_this);
// CHECK-NEXT:  }

// CHECK: static clad::ValueAndAdjoint<S3, S3> constructor_reverse_forw(clad::Tag<S3>, double x, double _d_x) {
// CHECK-NEXT:     S3 *_this = (S3 *)malloc(sizeof(S3));
// CHECK-NEXT:     S3 *_d_this = (S3 *)malloc(sizeof(S3));
// CHECK-NEXT:     memset(_d_this, 0, sizeof(S3));
// CHECK-NEXT:     _this->p = x * x;
// CHECK-NEXT:     return {*_this, *_d_this};
// CHECK-NEXT: }

// CHECK:  static void constructor_pullback(double x, S3 *_d_this, double *_d_x) {
// CHECK-NEXT:      S3 *_this = (S3 *)malloc(sizeof(S3));
// CHECK-NEXT:      _this->p = x * x;
// CHECK-NEXT:      {
// CHECK-NEXT:          double _r_d0 = _d_this->p;
// CHECK-NEXT:          _d_this->p = 0.;
// CHECK-NEXT:          *_d_x += _r_d0 * x;
// CHECK-NEXT:          *_d_x += x * _r_d0;
// CHECK-NEXT:      }
// CHECK-NEXT:      free(_this);
// CHECK-NEXT:  }

// CHECK: static clad::ValueAndAdjoint<S5, S5> constructor_reverse_forw(clad::Tag<S5>, double x, double _d_x) {
// CHECK-NEXT:     S5 *_this = (S5 *)malloc(sizeof(S5));
// CHECK-NEXT:     S5 *_d_this = (S5 *)malloc(sizeof(S5));
// CHECK-NEXT:     memset(_d_this, 0, sizeof(S5));
// CHECK-NEXT:     return {*_this, *_d_this};
// CHECK-NEXT: }

// CHECK:  static void constructor_pullback(double x, S5 *_d_this, double *_d_x) {
// CHECK-NEXT:      S5 *_this = (S5 *)malloc(sizeof(S5));
// CHECK-NEXT:      free(_this);
// CHECK-NEXT:  }

// CHECK:  void fn2_grad(double u, double v, double *_d_u, double *_d_v) {
// CHECK-NEXT:      clad::ValueAndAdjoint<S1, S1> _t0 = S1::constructor_reverse_forw(clad::Tag<S1>(), u, 0.);
// CHECK-NEXT:      S1 s1(_t0.value);
// CHECK-NEXT:      S1 _d_s1(_t0.adjoint);
// CHECK-NEXT:      clad::ValueAndAdjoint<S2, S2> _t1 = S2::constructor_reverse_forw(clad::Tag<S2>(), v, 0.);
// CHECK-NEXT:      S2 s2(_t1.value);
// CHECK-NEXT:      S2 _d_s2(_t1.adjoint);
// CHECK-NEXT:      clad::ValueAndAdjoint<S3, S3> _t2 = S3::constructor_reverse_forw(clad::Tag<S3>(), v, 0.);
// CHECK-NEXT:      S3 s3(_t2.value);
// CHECK-NEXT:      S3 _d_s3(_t2.adjoint);
// CHECK-NEXT:      clad::ValueAndAdjoint<S5, S5> _t3 = S5::constructor_reverse_forw(clad::Tag<S5>(), u, 0.);
// CHECK-NEXT:      S5 s5(_t3.value);
// CHECK-NEXT:      S5 _d_s5(_t3.adjoint);
// CHECK-NEXT:    {
// CHECK-NEXT:        double _r3 = 0.;
// CHECK-NEXT:        S5::constructor_pullback(u, &_d_s5, &_r3);
// CHECK-NEXT:        *_d_u += _r3;
// CHECK-NEXT:    }
// CHECK-NEXT:    {
// CHECK-NEXT:        double _r2 = 0.;
// CHECK-NEXT:        S3::constructor_pullback(v, &_d_s3, &_r2);
// CHECK-NEXT:        *_d_v += _r2;
// CHECK-NEXT:    }
// CHECK-NEXT:    {
// CHECK-NEXT:        double _r1 = 0.;
// CHECK-NEXT:        S2::constructor_pullback(v, &_d_s2, &_r1);
// CHECK-NEXT:        *_d_v += _r1;
// CHECK-NEXT:    }
// CHECK-NEXT:    {
// CHECK-NEXT:        double _r0 = 0.;
// CHECK-NEXT:        S1::constructor_pullback(u, &_d_s1, &_r0);
// CHECK-NEXT:        *_d_u += _r0;
// CHECK-NEXT:    }
// CHECK-NEXT:  }

double fn3(double u, double v) {
  S4 s(u);
  return s.i * s.p;
}

// CHECK: static clad::ValueAndAdjoint<S4, S4> constructor_reverse_forw(clad::Tag<S4>, double x, double _d_x) {
// CHECK-NEXT:    S4 *_this = (S4 *)malloc(sizeof(S4));
// CHECK-NEXT:    S4 *_d_this = (S4 *)malloc(sizeof(S4));
// CHECK-NEXT:    memset(_d_this, 0, sizeof(S4));
// CHECK-NEXT:    _this->i = 9;
// CHECK-NEXT:    _this->p = x;
// CHECK-NEXT:    return {*_this, *_d_this};
// CHECK-NEXT:}

// CHECK: static void constructor_pullback(double x, S4 *_d_this, double *_d_x) {
// CHECK-NEXT:    {
// CHECK-NEXT:        *_d_x += _d_this->p;
// CHECK-NEXT:        _d_this->p = 0.;
// CHECK-NEXT:    }
// CHECK-NEXT:    _d_this->i = 0.;
// CHECK-NEXT:}

// CHECK: void fn3_grad(double u, double v, double *_d_u, double *_d_v) {
// CHECK-NEXT:    clad::ValueAndAdjoint<S4, S4> _t0 = S4::constructor_reverse_forw(clad::Tag<S4>(), u, 0.);
// CHECK-NEXT:    S4 s(_t0.value);
// CHECK-NEXT:    S4 _d_s(_t0.adjoint);
// CHECK-NEXT:    {
// CHECK-NEXT:        _d_s.i += 1 * s.p;
// CHECK-NEXT:        _d_s.p += s.i * 1;
// CHECK-NEXT:    }
// CHECK-NEXT:    {
// CHECK-NEXT:        double _r0 = 0.;
// CHECK-NEXT:        S4::constructor_pullback(u, &_d_s, &_r0);
// CHECK-NEXT:        *_d_u += _r0;
// CHECK-NEXT:    }
// CHECK-NEXT:}

class SimpleFunctions1 {
public:
  SimpleFunctions1() noexcept : x(0), y(0) {}
  SimpleFunctions1(double px) : x(px), y(0) {}
  double x;
  double y;
};

double operator+(const double& val, const SimpleFunctions1& a) {
  return a.x + val;
}

// CHECK:  void operator_plus_pullback(const double &val, const SimpleFunctions1 &a, double _d_y, double *_d_val, SimpleFunctions1 *_d_a) {
// CHECK-NEXT:      {
// CHECK-NEXT:          (*_d_a).x += _d_y;
// CHECK-NEXT:          *_d_val += _d_y;
// CHECK-NEXT:      }
// CHECK-NEXT:  }

double fn4(double i, double j) {
  return 2 + SimpleFunctions1(i);
}

// CHECK: static void constructor_pullback(double px, SimpleFunctions1 *_d_this, double *_d_px) {
// CHECK-NEXT:      _d_this->y = 0.;
// CHECK-NEXT:      {
// CHECK-NEXT:          *_d_px += _d_this->x;
// CHECK-NEXT:          _d_this->x = 0.;
// CHECK-NEXT:      }
// CHECK-NEXT:  }

// CHECK:  void fn4_grad(double i, double j, double *_d_i, double *_d_j) {
// CHECK-NEXT:      {
// CHECK-NEXT:          double _r0 = 0.;
// CHECK-NEXT:          SimpleFunctions1 _r1 = 0.;
// CHECK-NEXT:          operator_plus_pullback(2, SimpleFunctions1(i), 1, &_r0, &_r1);
// CHECK-NEXT:          double _r2 = 0.;
// CHECK-NEXT:          SimpleFunctions1::constructor_pullback(i, &_r1, &_r2);
// CHECK-NEXT:          *_d_i += _r2;
// CHECK-NEXT:      }
// CHECK-NEXT:  }

struct argByValWrapper : public argByVal {
  double z;
  argByValWrapper(double v) : argByVal(v) {}
  argByValWrapper(double v, double u) : argByValWrapper(v) {
    z = y * u;
  }
  argByValWrapper(double v, bool) : argByVal(v) {
    z = x * y;
  }
};

double fn5(double x, double y) {
    argByValWrapper g(x);
    y = x;
    return y + g.y;
} // x + x^2

// CHECK:  static void constructor_pullback(double v, argByValWrapper *_d_this, double *_d_v) {
// CHECK-NEXT:      clad::ValueAndAdjoint<argByVal, argByVal> _t0 = argByVal::constructor_reverse_forw(clad::Tag<argByVal>(), v, 0.);
// CHECK-NEXT:      {
// CHECK-NEXT:          double _r0 = 0.;
// CHECK-NEXT:          argByVal::constructor_pullback(v, _d_this, &_r0);
// CHECK-NEXT:          *_d_v += _r0;
// CHECK-NEXT:      }
// CHECK-NEXT:  }

// CHECK:  void fn5_grad(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:      argByValWrapper g(x);
// CHECK-NEXT:      argByValWrapper _d_g(0.);
// CHECK-NEXT:      y = x;
// CHECK-NEXT:      {
// CHECK-NEXT:          *_d_y += 1;
// CHECK-NEXT:          _d_g.y += 1;
// CHECK-NEXT:      }
// CHECK-NEXT:      {
// CHECK-NEXT:          *_d_x += *_d_y;
// CHECK-NEXT:          *_d_y = 0.;
// CHECK-NEXT:      }
// CHECK-NEXT:      {
// CHECK-NEXT:          double _r0 = 0.;
// CHECK-NEXT:          argByValWrapper::constructor_pullback(x, &_d_g, &_r0);
// CHECK-NEXT:          *_d_x += _r0;
// CHECK-NEXT:      }
// CHECK-NEXT:  }

double fn6(double x, double y) {
    argByValWrapper g(x, y);
    return g.z;
} // x^2 * y

// CHECK: static clad::ValueAndAdjoint<argByValWrapper, argByValWrapper> constructor_reverse_forw(clad::Tag<argByValWrapper>, double v, double u, double _d_v, double _d_u) {
// CHECK-NEXT:     argByValWrapper *_this = new argByValWrapper(v);
// CHECK-NEXT:     argByValWrapper *_d_this = (argByValWrapper *)malloc(sizeof(argByValWrapper));
// CHECK-NEXT:     memset(_d_this, 0, sizeof(argByValWrapper));
// CHECK-NEXT:     _this->z = _this->y * u;
// CHECK-NEXT:     return {*_this, *_d_this};
// CHECK-NEXT: }

// CHECK:  static void constructor_pullback(double v, double u, argByValWrapper *_d_this, double *_d_v, double *_d_u) {
// CHECK-NEXT:      argByValWrapper *_this = new argByValWrapper(v);
// CHECK-NEXT:      _this->z = _this->y * u;
// CHECK-NEXT:      {
// CHECK-NEXT:          double _r_d0 = _d_this->z;
// CHECK-NEXT:          _d_this->z = 0.;
// CHECK-NEXT:          _d_this->y += _r_d0 * u;
// CHECK-NEXT:          *_d_u += _this->y * _r_d0;
// CHECK-NEXT:      }
// CHECK-NEXT:      {
// CHECK-NEXT:          double _r0 = 0.;
// CHECK-NEXT:          argByValWrapper::constructor_pullback(v, _d_this, &_r0);
// CHECK-NEXT:          *_d_v += _r0;
// CHECK-NEXT:      }
// CHECK-NEXT:      free(_this);
// CHECK-NEXT:  }

// CHECK:  void fn6_grad(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:      clad::ValueAndAdjoint<argByValWrapper, argByValWrapper> _t0 = argByValWrapper::constructor_reverse_forw(clad::Tag<argByValWrapper>(), x, y, 0., 0.);
// CHECK-NEXT:      argByValWrapper g(_t0.value);
// CHECK-NEXT:      argByValWrapper _d_g(_t0.adjoint);
// CHECK-NEXT:      _d_g.z += 1;
// CHECK-NEXT:      {
// CHECK-NEXT:          double _r0 = 0.;
// CHECK-NEXT:          double _r1 = 0.;
// CHECK-NEXT:          argByValWrapper::constructor_pullback(x, y, &_d_g, &_r0, &_r1);
// CHECK-NEXT:          *_d_x += _r0;
// CHECK-NEXT:          *_d_y += _r1;
// CHECK-NEXT:      }
// CHECK-NEXT:  }

double fn7(double x, double y) {
    argByValWrapper g(x, false);
    return g.z;
} // x^3

// CHECK: static clad::ValueAndAdjoint<argByValWrapper, argByValWrapper> constructor_reverse_forw(clad::Tag<argByValWrapper>, double v, bool arg, double _d_v, bool _d_arg) {
// CHECK-NEXT:     argByValWrapper *_this = (argByValWrapper *)malloc(sizeof(argByValWrapper));
// CHECK-NEXT:     argByValWrapper *_d_this = (argByValWrapper *)malloc(sizeof(argByValWrapper));
// CHECK-NEXT:     memset(_d_this, 0, sizeof(argByValWrapper));
// CHECK-NEXT:     clad::ValueAndAdjoint<argByVal, argByVal> _t0 = argByVal::constructor_reverse_forw(clad::Tag<argByVal>(), v, 0.);
// CHECK-NEXT:     new (static_cast<argByVal *>(_this)) argByVal(_t0.value);
// CHECK-NEXT:     _this->z = _this->x * _this->y;
// CHECK-NEXT:     return {*_this, *_d_this};
// CHECK-NEXT: }

// CHECK:  static void constructor_pullback(double v, bool arg, argByValWrapper *_d_this, double *_d_v, bool *_d_arg) {
// CHECK-NEXT:      argByValWrapper *_this = (argByValWrapper *)malloc(sizeof(argByValWrapper));
// CHECK-NEXT:      clad::ValueAndAdjoint<argByVal, argByVal> _t0 = argByVal::constructor_reverse_forw(clad::Tag<argByVal>(), v, 0.);
// CHECK-NEXT:      new (static_cast<argByVal *>(_this)) argByVal(_t0.value);
// CHECK-NEXT:      _this->z = _this->x * _this->y;
// CHECK-NEXT:      {
// CHECK-NEXT:          double _r_d0 = _d_this->z;
// CHECK-NEXT:          _d_this->z = 0.;
// CHECK-NEXT:          _d_this->x += _r_d0 * _this->y;
// CHECK-NEXT:          _d_this->y += _this->x * _r_d0;
// CHECK-NEXT:      }
// CHECK-NEXT:      {
// CHECK-NEXT:          double _r0 = 0.;
// CHECK-NEXT:          argByVal::constructor_pullback(v, _d_this, &_r0);
// CHECK-NEXT:          *_d_v += _r0;
// CHECK-NEXT:      }
// CHECK-NEXT:      free(_this);
// CHECK-NEXT:  }


// CHECK:  void fn7_grad(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:      clad::ValueAndAdjoint<argByValWrapper, argByValWrapper> _t0 = argByValWrapper::constructor_reverse_forw(clad::Tag<argByValWrapper>(), x, false, 0., false);
// CHECK-NEXT:      argByValWrapper g(_t0.value);
// CHECK-NEXT:      argByValWrapper _d_g(_t0.adjoint);
// CHECK-NEXT:      _d_g.z += 1;
// CHECK-NEXT:      {
// CHECK-NEXT:          double _r0 = 0.;
// CHECK-NEXT:          bool _r1 = false;
// CHECK-NEXT:          argByValWrapper::constructor_pullback(x, false, &_d_g, &_r0, &_r1);
// CHECK-NEXT:          *_d_x += _r0;
// CHECK-NEXT:      }
// CHECK-NEXT:  }

double fn8(double u, double v) {
  std::pair<double, double> p(u,v);
  return p.first + p.second;
}

// CHECK: constructor_pullback(double &__{{u1|x}}, double &__{{u2|y}}, std::pair<double, double> *_d_this, double *_d_{{u1|x}}, double *_d_{{u2|y}})
// CHECK-SAME: {
// CHECK-NEXT:     std::pair<double, double> *_this = (std::pair<double, double> *)malloc(sizeof(std::pair<double, double>));
// CHECK:          _this->first = __{{u1|x}};
// CHECK-NEXT:     _this->second = __{{u2|y}};
// CHECK:          {
// CHECK-NEXT:         *_d_{{u2|y}} += _d_this->second;
// CHECK-NEXT:         _d_this->second = 0.;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         *_d_{{u1|x}} += _d_this->first;
// CHECK-NEXT:         _d_this->first = 0.;
// CHECK-NEXT:     }
// CHECK-NEXT:     free(_this);
// CHECK-NEXT: }

// CHECK:  void fn8_grad(double u, double v, double *_d_u, double *_d_v) {
// CHECK-NEXT:      double _t0 = u;
// CHECK-NEXT:      double _t1 = v;
// CHECK-NEXT:      std::pair<double, double> p(u, v);
// CHECK-NEXT:      std::pair<double, double> _d_p(*_d_u, *_d_v);
// CHECK-NEXT:      {
// CHECK-NEXT:          _d_p.first += 1;
// CHECK-NEXT:          _d_p.second += 1;
// CHECK-NEXT:      }
// CHECK-NEXT:      {
// CHECK-NEXT:          u = _t0;
// CHECK-NEXT:          v = _t1;
// CHECK-NEXT:          std::pair<double, double>::constructor_pullback(u, v, &_d_p, _d_u, _d_v);
// CHECK-NEXT:      }
// CHECK-NEXT:  }

struct S {
  double* a;
};

double func2(S s) {
  return s.a[2];
}

// CHECK:  void func2_pullback(S s, double _d_y, S *_d_s) {
// CHECK-NEXT:      (*_d_s).a[2] += _d_y;
// CHECK-NEXT:  }

double fn9(S& s) {
  double r = func2(s);
  return r;
}

// CHECK:  void fn9_grad(S &s, S *_d_s) {
// CHECK-NEXT:      double _d_r = 0.;
// CHECK-NEXT:      double r = func2(s);
// CHECK-NEXT:      _d_r += 1;
// CHECK-NEXT:      {
// CHECK-NEXT:          S _r0 = (*_d_s);
// CHECK-NEXT:          func2_pullback(s, _d_r, &_r0);
// CHECK-NEXT:      }
// CHECK-NEXT:  }

struct arrWrapper {
    double arr[2];
};

// CHECK:  static inline constexpr void constructor_pullback(const arrWrapper &arg, arrWrapper *_d_this, arrWrapper *_d_arg) noexcept {
// CHECK-NEXT:      for (unsigned {{int|long}} i = 0; i < 2; ++i)
// CHECK-NEXT:          (*_d_arg).arr[i] += _d_this->arr[i];
// CHECK-NEXT:  }

double fn10(double x, double y) {
    arrWrapper a = {x, y};
    arrWrapper b = a;
    return b.arr[0] + b.arr[1];
}

// CHECK:  void fn10_grad(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:      arrWrapper _d_a = {{.*0.*}};
// CHECK-NEXT:      arrWrapper a = {{.*x, y.*}};
// CHECK-NEXT:      arrWrapper _d_b = _d_a;
// CHECK-NEXT:      arrWrapper b = a;
// CHECK-NEXT:      {
// CHECK-NEXT:          _d_b.arr[0] += 1;
// CHECK-NEXT:          _d_b.arr[1] += 1;
// CHECK-NEXT:      }
// CHECK-NEXT:      arrWrapper::constructor_pullback(a, &_d_b, &_d_a);
// CHECK-NEXT:      {
// CHECK-NEXT:          *_d_x += _d_a.arr[0];
// CHECK-NEXT:          *_d_y += _d_a.arr[1];
// CHECK-NEXT:      }
// CHECK-NEXT:  }

struct arr2DWrapper {
    double arr[1][2];
};

// CHECK:  static inline constexpr void constructor_pullback(const arr2DWrapper &arg, arr2DWrapper *_d_this, arr2DWrapper *_d_arg) noexcept {
// CHECK-NEXT:      for (unsigned {{int|long}} i = 0; i < 1; ++i)
// CHECK-NEXT:          for (unsigned {{int|long}} i0 = 0; i0 < 2; ++i0)
// CHECK-NEXT:              (*_d_arg).arr[i][i0] += _d_this->arr[i][i0];
// CHECK-NEXT:  }

double fn11(double x, double y) {
    arr2DWrapper a = {x, y};
    arr2DWrapper b = a;
    return b.arr[0][0] + b.arr[0][1];
}

// CHECK:  void fn11_grad(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:      arr2DWrapper _d_a = {{.*0.*}};
// CHECK-NEXT:      arr2DWrapper a = {{.*x, y.*}};
// CHECK-NEXT:      arr2DWrapper _d_b = _d_a;
// CHECK-NEXT:      arr2DWrapper b = a;
// CHECK-NEXT:      {
// CHECK-NEXT:          _d_b.arr[0][0] += 1;
// CHECK-NEXT:          _d_b.arr[0][1] += 1;
// CHECK-NEXT:      }
// CHECK-NEXT:      arr2DWrapper::constructor_pullback(a, &_d_b, &_d_a);
// CHECK-NEXT:      {
// CHECK-NEXT:          *_d_x += _d_a.arr[0][0];
// CHECK-NEXT:          *_d_y += _d_a.arr[0][1];
// CHECK-NEXT:      }
// CHECK-NEXT:  }

struct cust_double {
    cust_double(double x = 0): val(x) {} 
    double val;
};

// CHECK:  static void constructor_pullback(double x, cust_double *_d_this, double *_d_x) {
// CHECK-NEXT:      {
// CHECK-NEXT:          *_d_x += _d_this->val;
// CHECK-NEXT:          _d_this->val = 0.;
// CHECK-NEXT:      }
// CHECK-NEXT:  }

// CHECK:  static inline constexpr void constructor_pullback(const cust_double &arg, cust_double *_d_this, cust_double *_d_arg) noexcept {
// CHECK-NEXT:      {
// CHECK-NEXT:          (*_d_arg).val += _d_this->val;
// CHECK-NEXT:          _d_this->val = 0.;
// CHECK-NEXT:      }
// CHECK-NEXT:  }

struct arrStructWrapper {
    cust_double arr[2];
};

// CHECK:  static inline constexpr void constructor_pullback(const arrStructWrapper &arg, arrStructWrapper *_d_this, arrStructWrapper *_d_arg) noexcept {
// CHECK-NEXT:      for (unsigned {{int|long}} i = 0; i < 2; ++i)
// CHECK-NEXT:          cust_double::constructor_pullback(arg.arr[i], &_d_this->arr[i], &(*_d_arg).arr[i]);
// CHECK-NEXT:  }

double fn12(double x, double y) {
    arrStructWrapper a = {x, y};
    arrStructWrapper b = a;
    return b.arr[0].val + b.arr[1].val;
}

// CHECK:  void fn12_grad(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:      arrStructWrapper _d_a = {{.*}};
// CHECK-NEXT:      arrStructWrapper a = {{.*x, y.*}};
// CHECK-NEXT:      arrStructWrapper _d_b = _d_a;
// CHECK-NEXT:      arrStructWrapper b = a;
// CHECK-NEXT:      {
// CHECK-NEXT:          _d_b.arr[0].val += 1;
// CHECK-NEXT:          _d_b.arr[1].val += 1;
// CHECK-NEXT:      }
// CHECK-NEXT:      arrStructWrapper::constructor_pullback(a, &_d_b, &_d_a);
// CHECK-NEXT:      {
// CHECK-NEXT:          double _r0 = 0.;
// CHECK-NEXT:          cust_double::constructor_pullback(x, &_d_a.arr[0], &_r0);
// CHECK-NEXT:          *_d_x += _r0;
// CHECK-NEXT:          double _r1 = 0.;
// CHECK-NEXT:          cust_double::constructor_pullback(y, &_d_a.arr[1], &_r1);
// CHECK-NEXT:          *_d_y += _r1;
// CHECK-NEXT:      }
// CHECK-NEXT:  }

class ptrConstr {
  double* ptr;
  double val;
  public:
  ptrConstr(double& x, const double& y) : ptr(&x), val(y * y) {}

  double getSum() const {
    return *ptr + val;
  }
};

// CHECK:  static clad::ValueAndAdjoint<ptrConstr, ptrConstr> constructor_reverse_forw(clad::Tag<ptrConstr>, double &x, const double &y, double &_d_x, const double &_d_y) {
// CHECK-NEXT:      ptrConstr *_this = (ptrConstr *)malloc(sizeof(ptrConstr));
// CHECK-NEXT:      ptrConstr *_d_this = (ptrConstr *)malloc(sizeof(ptrConstr));
// CHECK-NEXT:      memset(_d_this, 0, sizeof(ptrConstr));
// CHECK-NEXT:      _this->ptr = &x;
// CHECK-NEXT:      _d_this->ptr = &_d_x;
// CHECK-NEXT:      _this->val = y * y;
// CHECK-NEXT:      return {*_this, *_d_this};
// CHECK-NEXT:  }

// CHECK:  static void constructor_pullback(double &x, const double &y, ptrConstr *_d_this, double *_d_x, double *_d_y) {
// CHECK-NEXT:      ptrConstr *_this = (ptrConstr *)malloc(sizeof(ptrConstr));
// CHECK-NEXT:      _this->ptr = &x;
// CHECK-NEXT:      _this->val = y * y;
// CHECK-NEXT:      {
// CHECK-NEXT:          *_d_y += y * _d_this->val;
// CHECK-NEXT:          *_d_y += _d_this->val * y;
// CHECK-NEXT:          _d_this->val = 0.;
// CHECK-NEXT:      }
// CHECK-NEXT:      free(_this);
// CHECK-NEXT:  }

// CHECK:  void getSum_pullback(double _d_y, ptrConstr *_d_this) const {
// CHECK-NEXT:      {
// CHECK-NEXT:          *_d_this->ptr += _d_y;
// CHECK-NEXT:          _d_this->val += _d_y;
// CHECK-NEXT:      }
// CHECK-NEXT:  }

double fn13(double x, double y) {
  ptrConstr obj(x, y);
  return obj.getSum();
} // x + y ^ 2

// CHECK:  void fn13_grad(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:      double _t0 = x;
// CHECK-NEXT:      clad::ValueAndAdjoint<ptrConstr, ptrConstr> _t1 = ptrConstr::constructor_reverse_forw(clad::Tag<ptrConstr>(), x, y, *_d_x, *_d_y);
// CHECK-NEXT:      ptrConstr obj(_t1.value);
// CHECK-NEXT:      ptrConstr _d_obj(_t1.adjoint);
// CHECK-NEXT:      obj.getSum_pullback(1, &_d_obj);
// CHECK-NEXT:      {
// CHECK-NEXT:          x = _t0;
// CHECK-NEXT:          ptrConstr::constructor_pullback(x, y, &_d_obj, _d_x, _d_y);
// CHECK-NEXT:      }
// CHECK-NEXT:  }

int main() {
    double d_i, d_j;

    INIT_GRADIENT(fn1);
    TEST_GRADIENT(fn1, /*numOfDerivativeArgs=*/2, 3, 4, &d_i, &d_j);    // CHECK-EXEC: {7.00, 0.00}

    INIT_GRADIENT(fn2);

    INIT_GRADIENT(fn3);
    TEST_GRADIENT(fn3, /*numOfDerivativeArgs=*/2, 3, 2, &d_i, &d_j);    // CHECK-EXEC: {9.00, 0.00}
    
    INIT_GRADIENT(fn4);
    TEST_GRADIENT(fn4, /*numOfDerivativeArgs=*/2, 3, 4, &d_i, &d_j);    // CHECK-EXEC: {1.00, 0.00}
    
    INIT_GRADIENT(fn5);
    TEST_GRADIENT(fn5, /*numOfDerivativeArgs=*/2, 3, 4, &d_i, &d_j);    // CHECK-EXEC: {7.00, 0.00}
    
    INIT_GRADIENT(fn6);
    TEST_GRADIENT(fn6, /*numOfDerivativeArgs=*/2, 3, 4, &d_i, &d_j);    // CHECK-EXEC: {24.00, 9.00}
    
    INIT_GRADIENT(fn7);
    TEST_GRADIENT(fn7, /*numOfDerivativeArgs=*/2, 2, 9, &d_i, &d_j);    // CHECK-EXEC: {12.00, 0.00}

    INIT_GRADIENT(fn8);
    TEST_GRADIENT(fn8, /*numOfDerivativeArgs=*/2, 7, 2, &d_i, &d_j);    // CHECK-EXEC: {1.00, 1.00}

    S s{new double[3]{5, 6, 7}}, _d_s{new double[3]{0}};
    auto dfn9 = clad::gradient(fn9);
    dfn9.execute(s, &_d_s);
    printf("{%.2f, %.2f, %.2f}\n", _d_s.a[0], _d_s.a[1], _d_s.a[2]);   // CHECK-EXEC: {0.00, 0.00, 1.00}

    INIT_GRADIENT(fn10);
    TEST_GRADIENT(fn10, /*numOfDerivativeArgs=*/2, 7, 2, &d_i, &d_j);    // CHECK-EXEC: {1.00, 1.00}

    INIT_GRADIENT(fn11);
    TEST_GRADIENT(fn11, /*numOfDerivativeArgs=*/2, 9, -1, &d_i, &d_j);    // CHECK-EXEC: {1.00, 1.00}

    INIT_GRADIENT(fn12);
    TEST_GRADIENT(fn12, /*numOfDerivativeArgs=*/2, 3, 6, &d_i, &d_j);    // CHECK-EXEC: {1.00, 1.00}

    INIT_GRADIENT(fn13);
    TEST_GRADIENT(fn13, /*numOfDerivativeArgs=*/2, 7, 2, &d_i, &d_j);    // CHECK-EXEC: {1.00, 4.00}
}
