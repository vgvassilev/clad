// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr %s -I%S/../../include -oConstructors.out -Xclang -verify 2>&1 | %filecheck %s
// RUN: ./Constructors.out | %filecheck_exec %s
// RUN: %cladclang %s -I%S/../../include -oConstructors.out
// RUN: ./Constructors.out | %filecheck_exec %s

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
}

// CHECK:  static void constructor_pullback(double val, argByVal *_d_this, double *_d_val) {
// CHECK-NEXT:      argByVal *_this = (argByVal *)malloc(sizeof(argByVal));
// CHECK-NEXT:      double _t0 = _this->x;
// CHECK-NEXT:      _this->x = val;
// CHECK-NEXT:      double _t1 = val;
// CHECK-NEXT:      val *= val;
// CHECK-NEXT:      double _t2 = _this->y;
// CHECK-NEXT:      _this->y = val;
// CHECK-NEXT:      {
// CHECK-NEXT:          _this->y = _t2;
// CHECK-NEXT:          double _r_d2 = _d_this->y;
// CHECK-NEXT:          _d_this->y = 0.;
// CHECK-NEXT:          *_d_val += _r_d2;
// CHECK-NEXT:      }
// CHECK-NEXT:      {
// CHECK-NEXT:          val = _t1;
// CHECK-NEXT:          double _r_d1 = *_d_val;
// CHECK-NEXT:          *_d_val = 0.;
// CHECK-NEXT:          *_d_val += _r_d1 * val;
// CHECK-NEXT:          *_d_val += val * _r_d1;
// CHECK-NEXT:      }
// CHECK-NEXT:      {
// CHECK-NEXT:          _this->x = _t0;
// CHECK-NEXT:          double _r_d0 = _d_this->x;
// CHECK-NEXT:          _d_this->x = 0.;
// CHECK-NEXT:          *_d_val += _r_d0;
// CHECK-NEXT:      }
// CHECK-NEXT:      free(_this);
// CHECK-NEXT:  }

// CHECK:  void fn1_grad(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:      argByVal g(x);
// CHECK-NEXT:      argByVal _d_g(g);
// CHECK-NEXT:      clad::zero_init(_d_g);
// CHECK-NEXT:      double _t0 = y; 
// CHECK-NEXT:      y = x;
// CHECK-NEXT:      {
// CHECK-NEXT:          *_d_y += 1;
// CHECK-NEXT:          _d_g.y += 1;
// CHECK-NEXT:      }
// CHECK-NEXT:      {
// CHECK-NEXT:          y = _t0;
// CHECK-NEXT:          double _r_d0 = *_d_y;
// CHECK-NEXT:          *_d_y = 0.;
// CHECK-NEXT:          *_d_x += _r_d0;
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
  S1(double x) : p(x), d([](){return 12.;}()) {} // expected-warning {{Direct lambda calls are not supported, ignored.}}
};

struct S2{
  double p;
  double i;
  double d;
  S2(double x) : p(x), i(1.), d([&](){i *= 32; return 12.;}()) {} // expected-warning {{Direct lambda calls are not supported, ignored.}}
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
    try { // expected-warning {{Try statements are not supported, ignored.}}
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

// CHECK:  static void constructor_pullback(double x, S1 *_d_this, double *_d_x) {
// CHECK-NEXT:      S1 *_this = (S1 *)malloc(sizeof(S1));
// CHECK-NEXT:      _this->p = x;
// CHECK-NEXT:      _this->d = 0.;
// CHECK-NEXT:      {
// CHECK-NEXT:          _d_this->d = 0.;
// CHECK-NEXT:      }
// CHECK-NEXT:      {
// CHECK-NEXT:          *_d_x += _d_this->p;
// CHECK-NEXT:          _d_this->p = 0.;
// CHECK-NEXT:      }
// CHECK-NEXT:      free(_this);
// CHECK-NEXT:  }

// CHECK:  static void constructor_pullback(double x, S2 *_d_this, double *_d_x) {
// CHECK-NEXT:      S2 *_this = (S2 *)malloc(sizeof(S2));
// CHECK-NEXT:      _this->p = x;
// CHECK-NEXT:      _this->i = 1.;
// CHECK-NEXT:      _this->d = 0.;
// CHECK-NEXT:      {
// CHECK-NEXT:          _d_this->d = 0.;
// CHECK-NEXT:      }
// CHECK-NEXT:      {
// CHECK-NEXT:          _d_this->i = 0.;
// CHECK-NEXT:      }
// CHECK-NEXT:      {
// CHECK-NEXT:          *_d_x += _d_this->p;
// CHECK-NEXT:          _d_this->p = 0.;
// CHECK-NEXT:      }
// CHECK-NEXT:      free(_this);
// CHECK-NEXT:  }

// CHECK:  static void constructor_pullback(double x, S3 *_d_this, double *_d_x) {
// CHECK-NEXT:      S3 *_this = (S3 *)malloc(sizeof(S3));
// CHECK-NEXT:      double _t0 = _this->p;
// CHECK-NEXT:      _this->p = x * x;
// CHECK-NEXT:      {
// CHECK-NEXT:          _this->p = _t0;
// CHECK-NEXT:          double _r_d0 = _d_this->p;
// CHECK-NEXT:          _d_this->p = 0.;
// CHECK-NEXT:          *_d_x += _r_d0 * x;
// CHECK-NEXT:          *_d_x += x * _r_d0;
// CHECK-NEXT:      }
// CHECK-NEXT:      free(_this);
// CHECK-NEXT:  }

// CHECK:  static void constructor_pullback(double x, S5 *_d_this, double *_d_x) {
// CHECK-NEXT:      S5 *_this = (S5 *)malloc(sizeof(S5));
// CHECK-NEXT:      free(_this);
// CHECK-NEXT:  }

// CHECK:  void fn2_grad(double u, double v, double *_d_u, double *_d_v) {
// CHECK-NEXT:      S1 s1(u);
// CHECK-NEXT:      S1 _d_s1(s1);
// CHECK-NEXT:      clad::zero_init(_d_s1);
// CHECK-NEXT:      S2 s2(v);
// CHECK-NEXT:      S2 _d_s2(s2);
// CHECK-NEXT:      clad::zero_init(_d_s2);
// CHECK-NEXT:      S3 s3(v);
// CHECK-NEXT:      S3 _d_s3(s3);
// CHECK-NEXT:      clad::zero_init(_d_s3);
// CHECK-NEXT:      S5 s5(u);
// CHECK-NEXT:      S5 _d_s5(s5);
// CHECK-NEXT:      clad::zero_init(_d_s5);
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

// CHECK: static void constructor_pullback(double x, S4 *_d_this, double *_d_x) {
// CHECK-NEXT:    {
// CHECK-NEXT:        *_d_x += _d_this->p;
// CHECK-NEXT:        _d_this->p = 0.;
// CHECK-NEXT:    }
// CHECK-NEXT:    {
// CHECK-NEXT:        _d_this->i = 0.;
// CHECK-NEXT:    }
// CHECK-NEXT:}

// CHECK: void fn3_grad(double u, double v, double *_d_u, double *_d_v) {
// CHECK-NEXT:    S4 s(u);
// CHECK-NEXT:    S4 _d_s(s);
// CHECK-NEXT:    clad::zero_init(_d_s);
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
// CHECK-NEXT:      {
// CHECK-NEXT:          _d_this->y = 0.;
// CHECK-NEXT:      }
// CHECK-NEXT:      {
// CHECK-NEXT:          *_d_px += _d_this->x;
// CHECK-NEXT:          _d_this->x = 0.;
// CHECK-NEXT:      }
// CHECK-NEXT:  }

// CHECK:  void fn4_grad(double i, double j, double *_d_i, double *_d_j) {
// CHECK-NEXT:      {
// CHECK-NEXT:          double _r0 = 0.;
// CHECK-NEXT:          SimpleFunctions1 _r1 = {};
// CHECK-NEXT:          operator_plus_pullback(2, SimpleFunctions1(i), 1, &_r0, &_r1);
// CHECK-NEXT:          double _r2 = 0.;
// CHECK-NEXT:          SimpleFunctions1::constructor_pullback(i, &_r1, &_r2);
// CHECK-NEXT:          *_d_i += _r2;
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
}
