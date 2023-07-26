// RUN: %cladclang %s -I%S/../../include -oReverseLoops.out 2>&1 | FileCheck %s
// RUN: ./ReverseLoops.out | FileCheck -check-prefix=CHECK-EXEC %s
//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"
#include <cmath>

#include "../TestUtils.h"

double f1(double x) {
  double t = 1;
  for (int i = 0; i < 3; i++)
    t *= x;
  return t;
} // == x^3

//CHECK:   void f1_grad(double x, clad::array_ref<double> _d_x) {
//CHECK-NEXT:       double _d_t = 0;
//CHECK-NEXT:       unsigned long _t0;
//CHECK-NEXT:       int _d_i = 0;
//CHECK-NEXT:       clad::tape<double> _t1 = {};
//CHECK-NEXT:       clad::tape<double> _t2 = {};
//CHECK-NEXT:       double t = 1;
//CHECK-NEXT:       _t0 = 0;
//CHECK-NEXT:       for (int i = 0; i < 3; i++) {
//CHECK-NEXT:           _t0++;
//CHECK-NEXT:           clad::push(_t2, t);
//CHECK-NEXT:           t *= clad::push(_t1, x);
//CHECK-NEXT:       }
//CHECK-NEXT:       _d_t += 1;
//CHECK-NEXT:       for (; _t0; _t0--) {
//CHECK-NEXT:           double _r_d0 = _d_t;
//CHECK-NEXT:           _d_t += _r_d0 * clad::pop(_t1);
//CHECK-NEXT:           double _r0 = clad::pop(_t2) * _r_d0;
//CHECK-NEXT:           * _d_x += _r0;
//CHECK-NEXT:           _d_t -= _r_d0;
//CHECK-NEXT:       }
//CHECK-NEXT:   }

double f2(double x) {
  double t = 1;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      t *= x;
  return t;
} // == x^9

//CHECK:   void f2_grad(double x, clad::array_ref<double> _d_x) {
//CHECK-NEXT:       double _d_t = 0;
//CHECK-NEXT:       unsigned long _t0;
//CHECK-NEXT:       int _d_i = 0;
//CHECK-NEXT:       clad::tape<unsigned long> _t1 = {};
//CHECK-NEXT:       int _d_j = 0;
//CHECK-NEXT:       clad::tape<double> _t2 = {};
//CHECK-NEXT:       clad::tape<double> _t3 = {};
//CHECK-NEXT:       double t = 1;
//CHECK-NEXT:       _t0 = 0;
//CHECK-NEXT:       for (int i = 0; i < 3; i++) {
//CHECK-NEXT:           _t0++;
//CHECK-NEXT:           clad::push(_t1, 0UL);
//CHECK-NEXT:           for (int j = 0; j < 3; j++) {
//CHECK-NEXT:               clad::back(_t1)++;
//CHECK-NEXT:               clad::push(_t3, t);
//CHECK-NEXT:               t *= clad::push(_t2, x);
//CHECK-NEXT:           }
//CHECK-NEXT:       }
//CHECK-NEXT:       _d_t += 1;
//CHECK-NEXT:       for (; _t0; _t0--) {
//CHECK-NEXT:           for (; clad::back(_t1); clad::back(_t1)--) {
//CHECK-NEXT:               double _r_d0 = _d_t;
//CHECK-NEXT:               _d_t += _r_d0 * clad::pop(_t2);
//CHECK-NEXT:               double _r0 = clad::pop(_t3) * _r_d0;
//CHECK-NEXT:               * _d_x += _r0;
//CHECK-NEXT:               _d_t -= _r_d0;
//CHECK-NEXT:           }
//CHECK-NEXT:           _d_j = 0;
//CHECK-NEXT:           clad::pop(_t1);
//CHECK-NEXT:       }
//CHECK-NEXT:   }

double f3(double x) {
  double t = 1;
  for (int i = 0; i < 3; i++) {
    t *= x;
    if (i == 1)
      return t;
  }
  return t;
} // == x^2

//CHECK:   void f3_grad(double x, clad::array_ref<double> _d_x) {
//CHECK-NEXT:       double _d_t = 0;
//CHECK-NEXT:       unsigned long _t0;
//CHECK-NEXT:       int _d_i = 0;
//CHECK-NEXT:       clad::tape<double> _t1 = {};
//CHECK-NEXT:       clad::tape<double> _t2 = {};
//CHECK-NEXT:       clad::tape<bool> _t4 = {};
//CHECK-NEXT:       double t = 1;
//CHECK-NEXT:       _t0 = 0;
//CHECK-NEXT:       for (int i = 0; i < 3; i++) {
//CHECK-NEXT:           _t0++;
//CHECK-NEXT:           clad::push(_t2, t);
//CHECK-NEXT:           t *= clad::push(_t1, x);
//CHECK-NEXT:           bool _t3 = i == 1;
//CHECK-NEXT:           {
//CHECK-NEXT:               if (_t3)
//CHECK-NEXT:                   goto _label0;
//CHECK-NEXT:               clad::push(_t4, _t3);
//CHECK-NEXT:           }
//CHECK-NEXT:       }
//CHECK-NEXT:       _d_t += 1;
//CHECK-NEXT:       for (; _t0; _t0--) {
//CHECK-NEXT:           if (clad::pop(_t4))
//CHECK-NEXT:             _label0:
//CHECK-NEXT:               _d_t += 1;
//CHECK-NEXT:           {
//CHECK-NEXT:               double _r_d0 = _d_t;
//CHECK-NEXT:               _d_t += _r_d0 * clad::pop(_t1);
//CHECK-NEXT:               double _r0 = clad::pop(_t2) * _r_d0;
//CHECK-NEXT:               * _d_x += _r0;
//CHECK-NEXT:               _d_t -= _r_d0;
//CHECK-NEXT:           }
//CHECK-NEXT:       }
//CHECK-NEXT:   }

double f4(double x) {
  double t = 1;
  for (int i = 0; i < 3; t *= x)
    i++;
  return t;
} // == x^3

//CHECK:   void f4_grad(double x, clad::array_ref<double> _d_x) {
//CHECK-NEXT:       double _d_t = 0;
//CHECK-NEXT:       unsigned long _t0;
//CHECK-NEXT:       int _d_i = 0;
//CHECK-NEXT:       clad::tape<double> _t1 = {};
//CHECK-NEXT:       clad::tape<double> _t2 = {};
//CHECK-NEXT:       double t = 1;
//CHECK-NEXT:       _t0 = 0;
//CHECK-NEXT:       for (int i = 0; i < 3; clad::push(_t2, t) , (t *= clad::push(_t1, x))) {
//CHECK-NEXT:           _t0++;
//CHECK-NEXT:           i++;
//CHECK-NEXT:       }
//CHECK-NEXT:       _d_t += 1;
//CHECK-NEXT:       for (; _t0; _t0--) {
//CHECK-NEXT:           double _r_d0 = _d_t;
//CHECK-NEXT:           _d_t += _r_d0 * clad::pop(_t1);
//CHECK-NEXT:           double _r0 = clad::pop(_t2) * _r_d0;
//CHECK-NEXT:           * _d_x += _r0;
//CHECK-NEXT:           _d_t -= _r_d0;
//CHECK-NEXT:       }
//CHECK-NEXT:   }

double f5(double x){
  for (int i = 0; i < 10; i++)
    x++;
  return x;
} // == x + 10

//CHECK:   void f5_grad(double x, clad::array_ref<double> _d_x) {
//CHECK-NEXT:       unsigned long _t0;
//CHECK-NEXT:       int _d_i = 0;
//CHECK-NEXT:       _t0 = 0;
//CHECK-NEXT:       for (int i = 0; i < 10; i++) {
//CHECK-NEXT:           _t0++;
//CHECK-NEXT:           x++;
//CHECK-NEXT:       }
//CHECK-NEXT:       * _d_x += 1;
//CHECK-NEXT:       for (; _t0; _t0--)
//CHECK-NEXT:           ;
//CHECK-NEXT:   }

double f_sum(double *p, int n) {
  double s = 0;
  for (int i = 0; i < n; i++)
    s += p[i];
  return s;
}

//CHECK: void f_sum_grad_0(double *p, int n, clad::array_ref<double> _d_p) {
//CHECK-NEXT:     int _d_n = 0;
//CHECK-NEXT:     double _d_s = 0;
//CHECK-NEXT:     unsigned long _t0;
//CHECK-NEXT:     int _d_i = 0;
//CHECK-NEXT:     clad::tape<int> _t1 = {};
//CHECK-NEXT:     double s = 0;
//CHECK-NEXT:     _t0 = 0;
//CHECK-NEXT:     for (int i = 0; i < n; i++) {
//CHECK-NEXT:         _t0++;
//CHECK-NEXT:         s += p[clad::push(_t1, i)];
//CHECK-NEXT:     }
//CHECK-NEXT:     _d_s += 1;
//CHECK-NEXT:     for (; _t0; _t0--) {
//CHECK-NEXT:         double _r_d0 = _d_s;
//CHECK-NEXT:         _d_s += _r_d0;
//CHECK-NEXT:         int _t2 = clad::pop(_t1);
//CHECK-NEXT:         _d_p[_t2] += _r_d0;
//CHECK-NEXT:         _d_s -= _r_d0;
//CHECK-NEXT:     }
//CHECK-NEXT: }

double sq(double x) { return x * x; }
//CHECK:   void sq_pullback(double x, double _d_y, clad::array_ref<double> _d_x) {
//CHECK-NEXT:       double _t0;
//CHECK-NEXT:       double _t1;
//CHECK-NEXT:       _t1 = x;
//CHECK-NEXT:       _t0 = x;
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r0 = _d_y * _t0;
//CHECK-NEXT:           * _d_x += _r0;
//CHECK-NEXT:           double _r1 = _t1 * _d_y;
//CHECK-NEXT:           * _d_x += _r1;
//CHECK-NEXT:       }
//CHECK-NEXT:   }

double f_sum_squares(double *p, int n) {
  double s = 0;
  for (int i = 0; i < n; i++)
    s += sq(p[i]);
  return s;
}

//CHECK: void f_sum_squares_grad_0(double *p, int n, clad::array_ref<double> _d_p) {
//CHECK-NEXT:     int _d_n = 0;
//CHECK-NEXT:     double _d_s = 0;
//CHECK-NEXT:     unsigned long _t0;
//CHECK-NEXT:     int _d_i = 0;
//CHECK-NEXT:     clad::tape<int> _t1 = {};
//CHECK-NEXT:     clad::tape<double> _t3 = {};
//CHECK-NEXT:     double s = 0;
//CHECK-NEXT:     _t0 = 0;
//CHECK-NEXT:     for (int i = 0; i < n; i++) {
//CHECK-NEXT:         _t0++;
//CHECK-NEXT:         s += sq(clad::push(_t3, p[clad::push(_t1, i)]));
//CHECK-NEXT:     }
//CHECK-NEXT:     _d_s += 1;
//CHECK-NEXT:     for (; _t0; _t0--) {
//CHECK-NEXT:         double _r_d0 = _d_s;
//CHECK-NEXT:         _d_s += _r_d0;
//CHECK-NEXT:         double _grad0 = 0.;
//CHECK-NEXT:         sq_pullback(clad::pop(_t3), _r_d0, &_grad0);
//CHECK-NEXT:         double _r0 = _grad0;
//CHECK-NEXT:         int _t2 = clad::pop(_t1);
//CHECK-NEXT:         _d_p[_t2] += _r0;
//CHECK-NEXT:         _d_s -= _r_d0;
//CHECK-NEXT:     }
//CHECK-NEXT: }

// log-likelihood of n-dimensional gaussian distribution with covariance sigma^2*I
double f_log_gaus(double* x, double* p /*means*/, double n, double sigma) {
  double power = 0;
  for (int i = 0; i < n; i++)
    power += sq(x[i] - p[i]);
  power = -power/(2*sq(sigma));
  double gaus = 1./std::sqrt(std::pow(2*M_PI, n) * sigma) * std::exp(power);
  return std::log(gaus);
}

//CHECK: void f_log_gaus_grad_1(double *x, double *p, double n, double sigma, clad::array_ref<double> _d_p) {
//CHECK-NEXT:     double _d_n = 0;
//CHECK-NEXT:     double _d_sigma = 0;
//CHECK-NEXT:     double _d_power = 0;
//CHECK-NEXT:     unsigned long _t0;
//CHECK-NEXT:     int _d_i = 0;
//CHECK-NEXT:     clad::tape<int> _t1 = {};
//CHECK-NEXT:     clad::tape<int> _t3 = {};
//CHECK-NEXT:     clad::tape<double> _t5 = {};
//CHECK-NEXT:     double _t6;
//CHECK-NEXT:     double _t7;
//CHECK-NEXT:     double _t8;
//CHECK-NEXT:     double _t9;
//CHECK-NEXT:     double _t10;
//CHECK-NEXT:     double _t11;
//CHECK-NEXT:     double _t12;
//CHECK-NEXT:     double _t13;
//CHECK-NEXT:     double _t14;
//CHECK-NEXT:     double _t15;
//CHECK-NEXT:     double _t16;
//CHECK-NEXT:     double _t17;
//CHECK-NEXT:     double _t18;
//CHECK-NEXT:     double _d_gaus = 0;
//CHECK-NEXT:     double _t19;
//CHECK-NEXT:     double power = 0;
//CHECK-NEXT:     _t0 = 0;
//CHECK-NEXT:     for (int i = 0; i < n; i++) {
//CHECK-NEXT:         _t0++;
//CHECK-NEXT:         power += sq(clad::push(_t5, x[clad::push(_t1, i)] - p[clad::push(_t3, i)]));
//CHECK-NEXT:     }
//CHECK-NEXT:     _t7 = -power;
//CHECK-NEXT:     _t9 = sigma;
//CHECK-NEXT:     _t8 = sq(_t9);
//CHECK-NEXT:     _t6 = (2 * _t8);
//CHECK-NEXT:     power = _t7 / _t6;
//CHECK-NEXT:     _t13 = 2 * 3.1415926535897931;
//CHECK-NEXT:     _t14 = n;
//CHECK-NEXT:     _t15 = std::pow(_t13, _t14);
//CHECK-NEXT:     _t12 = sigma;
//CHECK-NEXT:     _t16 = _t15 * _t12;
//CHECK-NEXT:     _t11 = std::sqrt(_t16);
//CHECK-NEXT:     _t17 = 1. / _t11;
//CHECK-NEXT:     _t18 = power;
//CHECK-NEXT:     _t10 = std::exp(_t18);
//CHECK-NEXT:     double gaus = _t17 * _t10;
//CHECK-NEXT:     _t19 = gaus;
//CHECK-NEXT:     {
//CHECK-NEXT:          double _r17 = 1 * clad::custom_derivatives::log_pushforward(_t19, 1.).pushforward;
//CHECK-NEXT:         _d_gaus += _r17;
//CHECK-NEXT:     }
//CHECK-NEXT:     {
//CHECK-NEXT:         double _r6 = _d_gaus * _t10;
//CHECK-NEXT:         double _r7 = _r6 / _t11;
//CHECK-NEXT:         double _r8 = _r6 * -1. / (_t11 * _t11);
//CHECK-NEXT:         double _r9 = _r8 * clad::custom_derivatives::sqrt_pushforward(_t16, 1.).pushforward;
//CHECK-NEXT:         double _r10 = _r9 * _t12;
//CHECK-NEXT:         double _grad2 = 0.;
//CHECK-NEXT:         double _grad3 = 0.;
//CHECK-NEXT:         clad::custom_derivatives{{(::std)?}}::pow_pullback(_t13, _t14, _r10, &_grad2, &_grad3);
//CHECK-NEXT:         double _r11 = _grad2;
//CHECK-NEXT:         double _r12 = _r11 * 3.1415926535897931;
//CHECK-NEXT:         double _r13 = _grad3;
//CHECK-NEXT:         _d_n += _r13;
//CHECK-NEXT:         double _r14 = _t15 * _r9;
//CHECK-NEXT:         _d_sigma += _r14;
//CHECK-NEXT:         double _r15 = _t17 * _d_gaus;
//CHECK-NEXT:         double _r16 = _r15 * clad::custom_derivatives::exp_pushforward(_t18, 1.).pushforward;
//CHECK-NEXT:         _d_power += _r16;
//CHECK-NEXT:     }
//CHECK-NEXT:     {
//CHECK-NEXT:         double _r_d1 = _d_power;
//CHECK-NEXT:         double _r1 = _r_d1 / _t6;
//CHECK-NEXT:         _d_power += -_r1;
//CHECK-NEXT:         double _r2 = _r_d1 * -_t7 / (_t6 * _t6);
//CHECK-NEXT:         double _r3 = _r2 * _t8;
//CHECK-NEXT:         double _r4 = 2 * _r2;
//CHECK-NEXT:         double _grad1 = 0.;
//CHECK-NEXT:         sq_pullback(_t9, _r4, &_grad1);
//CHECK-NEXT:         double _r5 = _grad1;
//CHECK-NEXT:         _d_sigma += _r5;
//CHECK-NEXT:         _d_power -= _r_d1;
//CHECK-NEXT:     }
//CHECK-NEXT:     for (; _t0; _t0--) {
//CHECK-NEXT:         double _r_d0 = _d_power;
//CHECK-NEXT:         _d_power += _r_d0;
//CHECK-NEXT:         double _grad0 = 0.;
//CHECK-NEXT:         sq_pullback(clad::pop(_t5), _r_d0, &_grad0);
//CHECK-NEXT:         double _r0 = _grad0;
//CHECK-NEXT:         int _t2 = clad::pop(_t1);
//CHECK-NEXT:         int _t4 = clad::pop(_t3);
//CHECK-NEXT:         _d_p[_t4] += -_r0;
//CHECK-NEXT:         _d_power -= _r_d0;
//CHECK-NEXT:     }
//CHECK-NEXT: }

double f_const(const double a, const double b) {
  int r = 0;
  for (int i = 0; i < a; i++) {
    int sq = b * b;
    r += sq;
  }
  return r;
}

void f_const_grad(const double, const double, clad::array_ref<double>, clad::array_ref<double>);
//CHECK:   void f_const_grad(const double a, const double b, clad::array_ref<double> _d_a, clad::array_ref<double> _d_b) {
//CHECK-NEXT:       int _d_r = 0;
//CHECK-NEXT:       unsigned long _t0;
//CHECK-NEXT:       int _d_i = 0;
//CHECK-NEXT:       clad::tape<double> _t1 = {};
//CHECK-NEXT:       clad::tape<double> _t2 = {};
//CHECK-NEXT:       int _d_sq = 0;
//CHECK-NEXT:       int r = 0;
//CHECK-NEXT:       _t0 = 0;
//CHECK-NEXT:       for (int i = 0; i < a; i++) {
//CHECK-NEXT:           _t0++;
//CHECK-NEXT:           int sq0 = clad::push(_t2, b) * clad::push(_t1, b);
//CHECK-NEXT:           r += sq0;
//CHECK-NEXT:       }
//CHECK-NEXT:       _d_r += 1;
//CHECK-NEXT:       for (; _t0; _t0--) {
//CHECK-NEXT:           {
//CHECK-NEXT:               int _r_d0 = _d_r;
//CHECK-NEXT:               _d_r += _r_d0;
//CHECK-NEXT:               _d_sq += _r_d0;
//CHECK-NEXT:               _d_r -= _r_d0;
//CHECK-NEXT:           }
//CHECK-NEXT:           {
//CHECK-NEXT:               double _r0 = _d_sq * clad::pop(_t1);
//CHECK-NEXT:               * _d_b += _r0;
//CHECK-NEXT:               double _r1 = clad::pop(_t2) * _d_sq;
//CHECK-NEXT:               * _d_b += _r1;
// CHECK-NEXT:              _d_sq = 0;
//CHECK-NEXT:           }
//CHECK-NEXT:       }
//CHECK-NEXT:   }

double f6 (double i, double j) {
  double a = 0;
  for (int counter=0; counter<3; ++counter) {
    double b = i*i;
    double c = j*j;
    b += j;
    a += b + c + i;
  }
  return a;
}

// CHECK: void f6_grad(double i, double j, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j) {
// CHECK-NEXT:     double _d_a = 0;
// CHECK-NEXT:     unsigned long _t0;
// CHECK-NEXT:     int _d_counter = 0;
// CHECK-NEXT:     clad::tape<double> _t1 = {};
// CHECK-NEXT:     clad::tape<double> _t2 = {};
// CHECK-NEXT:     double _d_b = 0;
// CHECK-NEXT:     clad::tape<double> _t3 = {};
// CHECK-NEXT:     clad::tape<double> _t4 = {};
// CHECK-NEXT:     double _d_c = 0;
// CHECK-NEXT:     double a = 0;
// CHECK-NEXT:     _t0 = 0;
// CHECK-NEXT:     for (int counter = 0; counter < 3; ++counter) {
// CHECK-NEXT:         _t0++;
// CHECK-NEXT:         double b = clad::push(_t2, i) * clad::push(_t1, i);
// CHECK-NEXT:         double c = clad::push(_t4, j) * clad::push(_t3, j);
// CHECK-NEXT:         b += j;
// CHECK-NEXT:         a += b + c + i;
// CHECK-NEXT:     }
// CHECK-NEXT:     _d_a += 1;
// CHECK-NEXT:     for (; _t0; _t0--) {
// CHECK-NEXT:         {
// CHECK-NEXT:             double _r_d1 = _d_a;
// CHECK-NEXT:             _d_a += _r_d1;
// CHECK-NEXT:             _d_b += _r_d1;
// CHECK-NEXT:             _d_c += _r_d1;
// CHECK-NEXT:             * _d_i += _r_d1;
// CHECK-NEXT:             _d_a -= _r_d1;
// CHECK-NEXT:         }
// CHECK-NEXT:         {
// CHECK-NEXT:             double _r_d0 = _d_b;
// CHECK-NEXT:             _d_b += _r_d0;
// CHECK-NEXT:             * _d_j += _r_d0;
// CHECK-NEXT:             _d_b -= _r_d0;
// CHECK-NEXT:         }
// CHECK-NEXT:         {
// CHECK-NEXT:             double _r2 = _d_c * clad::pop(_t3);
// CHECK-NEXT:             * _d_j += _r2;
// CHECK-NEXT:             double _r3 = clad::pop(_t4) * _d_c;
// CHECK-NEXT:             * _d_j += _r3;
// CHECK-NEXT:             _d_c = 0;
// CHECK-NEXT:         }
// CHECK-NEXT:         {
// CHECK-NEXT:             double _r0 = _d_b * clad::pop(_t1);
// CHECK-NEXT:             * _d_i += _r0;
// CHECK-NEXT:             double _r1 = clad::pop(_t2) * _d_b;
// CHECK-NEXT:             * _d_i += _r1;
// CHECK-NEXT:             _d_b = 0;
// CHECK-NEXT:         }
// CHECK-NEXT:     }
// CHECK-NEXT: }

double fn7(double i, double j) {
  double a = 0;
  int counter = 3;
  while (counter--)
    a += i*i + j;
  return a;
}

// CHECK: void fn7_grad(double i, double j, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j) {
// CHECK-NEXT:     double _d_a = 0;
// CHECK-NEXT:     int _d_counter = 0;
// CHECK-NEXT:     unsigned long _t0;
// CHECK-NEXT:     clad::tape<double> _t1 = {};
// CHECK-NEXT:     clad::tape<double> _t2 = {};
// CHECK-NEXT:     double a = 0;
// CHECK-NEXT:     int counter = 3;
// CHECK-NEXT:     _t0 = 0;
// CHECK-NEXT:     while (counter--)
// CHECK-NEXT:         {
// CHECK-NEXT:             _t0++;
// CHECK-NEXT:             a += clad::push(_t2, i) * clad::push(_t1, i) + j;
// CHECK-NEXT:         }
// CHECK-NEXT:     _d_a += 1;
// CHECK-NEXT:     while (_t0)
// CHECK-NEXT:         {
// CHECK-NEXT:             {
// CHECK-NEXT:                 double _r_d0 = _d_a;
// CHECK-NEXT:                 _d_a += _r_d0;
// CHECK-NEXT:                 double _r0 = _r_d0 * clad::pop(_t1);
// CHECK-NEXT:                 * _d_i += _r0;
// CHECK-NEXT:                 double _r1 = clad::pop(_t2) * _r_d0;
// CHECK-NEXT:                 * _d_i += _r1;
// CHECK-NEXT:                 * _d_j += _r_d0;
// CHECK-NEXT:                 _d_a -= _r_d0;
// CHECK-NEXT:             }
// CHECK-NEXT:             _t0--;
// CHECK-NEXT:         }
// CHECK-NEXT: }

double fn8(double i, double j) {
  double a = 0;
  int counter = 3;
  while (counter > 0)
    do {
      a += i*i + j;
    } while (--counter);
  return a;
}

// CHECK: void fn8_grad(double i, double j, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j) {
// CHECK-NEXT:     double _d_a = 0;
// CHECK-NEXT:     int _d_counter = 0;
// CHECK-NEXT:     unsigned long _t0;
// CHECK-NEXT:     clad::tape<unsigned long> _t1 = {};
// CHECK-NEXT:     clad::tape<double> _t2 = {};
// CHECK-NEXT:     clad::tape<double> _t3 = {};
// CHECK-NEXT:     double a = 0;
// CHECK-NEXT:     int counter = 3;
// CHECK-NEXT:     _t0 = 0;
// CHECK-NEXT:     while (counter > 0)
// CHECK-NEXT:         {
// CHECK-NEXT:             _t0++;
// CHECK-NEXT:             clad::push(_t1, 0UL);
// CHECK-NEXT:             do {
// CHECK-NEXT:                 clad::back(_t1)++;
// CHECK-NEXT:                 a += clad::push(_t3, i) * clad::push(_t2, i) + j;
// CHECK-NEXT:             } while (--counter);
// CHECK-NEXT:         }
// CHECK-NEXT:     _d_a += 1;
// CHECK-NEXT:     while (_t0)
// CHECK-NEXT:         {
// CHECK-NEXT:             {
// CHECK-NEXT:                 do {
// CHECK-NEXT:                     {
// CHECK-NEXT:                         {
// CHECK-NEXT:                             double _r_d0 = _d_a;
// CHECK-NEXT:                             _d_a += _r_d0;
// CHECK-NEXT:                             double _r0 = _r_d0 * clad::pop(_t2);
// CHECK-NEXT:                             * _d_i += _r0;
// CHECK-NEXT:                             double _r1 = clad::pop(_t3) * _r_d0;
// CHECK-NEXT:                             * _d_i += _r1;
// CHECK-NEXT:                             * _d_j += _r_d0;
// CHECK-NEXT:                             _d_a -= _r_d0;
// CHECK-NEXT:                         }
// CHECK-NEXT:                     }
// CHECK-NEXT:                     clad::back(_t1)--;
// CHECK-NEXT:                 } while (clad::back(_t1));
// CHECK-NEXT:                 clad::pop(_t1);
// CHECK-NEXT:             }
// CHECK-NEXT:             _t0--;
// CHECK-NEXT:         }
// CHECK-NEXT: }

double fn9(double i, double j) {
  int counter, counter_again;
  counter = counter_again = 3;
  double a = 0;
  while (counter--) {
    counter_again = 3;
    while (counter_again--) {
      a += i*i + j;
    }
  }
  return a;
}

// CHECK: void fn9_grad(double i, double j, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j) {
// CHECK-NEXT:     int _d_counter = 0, _d_counter_again = 0;
// CHECK-NEXT:     double _d_a = 0;
// CHECK-NEXT:     unsigned long _t0;
// CHECK-NEXT:     clad::tape<unsigned long> _t1 = {};
// CHECK-NEXT:     clad::tape<double> _t2 = {};
// CHECK-NEXT:     clad::tape<double> _t3 = {};
// CHECK-NEXT:     int counter, counter_again;
// CHECK-NEXT:     counter = counter_again = 3;
// CHECK-NEXT:     double a = 0;
// CHECK-NEXT:     _t0 = 0;
// CHECK-NEXT:     while (counter--)
// CHECK-NEXT:         {
// CHECK-NEXT:             _t0++;
// CHECK-NEXT:             counter_again = 3;
// CHECK-NEXT:             clad::push(_t1, 0UL);
// CHECK-NEXT:             while (counter_again--)
// CHECK-NEXT:                 {
// CHECK-NEXT:                     clad::back(_t1)++;
// CHECK-NEXT:                     a += clad::push(_t3, i) * clad::push(_t2, i) + j;
// CHECK-NEXT:                 }
// CHECK-NEXT:         }
// CHECK-NEXT:     _d_a += 1;
// CHECK-NEXT:     while (_t0)
// CHECK-NEXT:         {
// CHECK-NEXT:             {
// CHECK-NEXT:                 {
// CHECK-NEXT:                     while (clad::back(_t1))
// CHECK-NEXT:                         {
// CHECK-NEXT:                             {
// CHECK-NEXT:                                 {
// CHECK-NEXT:                                     double _r_d3 = _d_a;
// CHECK-NEXT:                                     _d_a += _r_d3;
// CHECK-NEXT:                                     double _r0 = _r_d3 * clad::pop(_t2);
// CHECK-NEXT:                                     * _d_i += _r0;
// CHECK-NEXT:                                     double _r1 = clad::pop(_t3) * _r_d3;
// CHECK-NEXT:                                     * _d_i += _r1;
// CHECK-NEXT:                                     * _d_j += _r_d3;
// CHECK-NEXT:                                     _d_a -= _r_d3;
// CHECK-NEXT:                                 }
// CHECK-NEXT:                             }
// CHECK-NEXT:                             clad::back(_t1)--;
// CHECK-NEXT:                         }
// CHECK-NEXT:                     clad::pop(_t1);
// CHECK-NEXT:                 }
// CHECK-NEXT:                 {
// CHECK-NEXT:                     int _r_d2 = _d_counter_again;
// CHECK-NEXT:                     _d_counter_again -= _r_d2;
// CHECK-NEXT:                 }
// CHECK-NEXT:             }
// CHECK-NEXT:             _t0--;
// CHECK-NEXT:         }
// CHECK-NEXT:     {
// CHECK-NEXT:         int _r_d0 = _d_counter;
// CHECK-NEXT:         _d_counter_again += _r_d0;
// CHECK-NEXT:         int _r_d1 = _d_counter_again;
// CHECK-NEXT:         _d_counter_again -= _r_d1;
// CHECK-NEXT:         _d_counter -= _r_d0;
// CHECK-NEXT:     }
// CHECK-NEXT: }

double fn10(double i, double j) {
  double a = 0;
  int counter = 3;
  while (int b = counter) {
    b += i*i + j;
    a += b;
    counter -= 1;
  }
  return a;
}

// CHECK: void fn10_grad(double i, double j, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j) {
// CHECK-NEXT:     double _d_a = 0;
// CHECK-NEXT:     int _d_counter = 0;
// CHECK-NEXT:     unsigned long _t0;
// CHECK-NEXT:     int _d_b = 0;
// CHECK-NEXT:     clad::tape<double> _t1 = {};
// CHECK-NEXT:     clad::tape<double> _t2 = {};
// CHECK-NEXT:     double a = 0;
// CHECK-NEXT:     int counter = 3;
// CHECK-NEXT:     _t0 = 0;
// CHECK-NEXT:     while (int b = counter)
// CHECK-NEXT:         {
// CHECK-NEXT:             _t0++;
// CHECK-NEXT:             b += clad::push(_t2, i) * clad::push(_t1, i) + j;
// CHECK-NEXT:             a += b;
// CHECK-NEXT:             counter -= 1;
// CHECK-NEXT:         }
// CHECK-NEXT:     _d_a += 1;
// CHECK-NEXT:     while (_t0)
// CHECK-NEXT:         {
// CHECK-NEXT:             {
// CHECK-NEXT:                 {
// CHECK-NEXT:                     int _r_d2 = _d_counter;
// CHECK-NEXT:                     _d_counter += _r_d2;
// CHECK-NEXT:                     _d_counter -= _r_d2;
// CHECK-NEXT:                 }
// CHECK-NEXT:                 {
// CHECK-NEXT:                     double _r_d1 = _d_a;
// CHECK-NEXT:                     _d_a += _r_d1;
// CHECK-NEXT:                     _d_b += _r_d1;
// CHECK-NEXT:                     _d_a -= _r_d1;
// CHECK-NEXT:                 }
// CHECK-NEXT:                 {
// CHECK-NEXT:                     int _r_d0 = _d_b;
// CHECK-NEXT:                     _d_b += _r_d0;
// CHECK-NEXT:                     double _r0 = _r_d0 * clad::pop(_t1);
// CHECK-NEXT:                     * _d_i += _r0;
// CHECK-NEXT:                     double _r1 = clad::pop(_t2) * _r_d0;
// CHECK-NEXT:                     * _d_i += _r1;
// CHECK-NEXT:                     * _d_j += _r_d0;
// CHECK-NEXT:                     _d_b -= _r_d0;
// CHECK-NEXT:                 }
// CHECK-NEXT:             }
// CHECK-NEXT:             {
// CHECK-NEXT:                 _d_counter += _d_b;
// CHECK-NEXT:                 _d_b = 0;
// CHECK-NEXT:             }
// CHECK-NEXT:             _t0--;
// CHECK-NEXT:         }
// CHECK-NEXT: }

double fn11(double i, double j) {
  int counter = 3;
  double a = 0;
  do {
    a += i*i + j;
    counter -= 1;
  } while (counter);
  return a;
}

// CHECK: void fn11_grad(double i, double j, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j) {
// CHECK-NEXT:     int _d_counter = 0;
// CHECK-NEXT:     double _d_a = 0;
// CHECK-NEXT:     unsigned long _t0;
// CHECK-NEXT:     clad::tape<double> _t1 = {};
// CHECK-NEXT:     clad::tape<double> _t2 = {};
// CHECK-NEXT:     int counter = 3;
// CHECK-NEXT:     double a = 0;
// CHECK-NEXT:     _t0 = 0;
// CHECK-NEXT:     do {
// CHECK-NEXT:         _t0++;
// CHECK-NEXT:         a += clad::push(_t2, i) * clad::push(_t1, i) + j;
// CHECK-NEXT:         counter -= 1;
// CHECK-NEXT:     } while (counter);
// CHECK-NEXT:     _d_a += 1;
// CHECK-NEXT:     do {
// CHECK-NEXT:         {
// CHECK-NEXT:             {
// CHECK-NEXT:                 int _r_d1 = _d_counter;
// CHECK-NEXT:                 _d_counter += _r_d1;
// CHECK-NEXT:                 _d_counter -= _r_d1;
// CHECK-NEXT:             }
// CHECK-NEXT:             {
// CHECK-NEXT:                 double _r_d0 = _d_a;
// CHECK-NEXT:                 _d_a += _r_d0;
// CHECK-NEXT:                 double _r0 = _r_d0 * clad::pop(_t1);
// CHECK-NEXT:                 * _d_i += _r0;
// CHECK-NEXT:                 double _r1 = clad::pop(_t2) * _r_d0;
// CHECK-NEXT:                 * _d_i += _r1;
// CHECK-NEXT:                 * _d_j += _r_d0;
// CHECK-NEXT:                 _d_a -= _r_d0;
// CHECK-NEXT:             }
// CHECK-NEXT:         }
// CHECK-NEXT:         _t0--;
// CHECK-NEXT:     } while (_t0);
// CHECK-NEXT: }

double fn12(double i, double j) {
  int counter = 3;
  double a = 0;
  do {
    int counter_again = 3;
    do {
      a += i*i + j;
      counter_again -= 1;
      do
        a += j;
      while (0);
    } while (counter_again);
    counter -= 1;
  } while (counter);
  return a;
}

// CHECK: void fn12_grad(double i, double j, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j) {
// CHECK-NEXT:     int _d_counter = 0;
// CHECK-NEXT:     double _d_a = 0;
// CHECK-NEXT:     unsigned long _t0;
// CHECK-NEXT:     int _d_counter_again = 0;
// CHECK-NEXT:     clad::tape<unsigned long> _t1 = {};
// CHECK-NEXT:     clad::tape<double> _t2 = {};
// CHECK-NEXT:     clad::tape<double> _t3 = {};
// CHECK-NEXT:     clad::tape<unsigned long> _t4 = {};
// CHECK-NEXT:     int counter = 3;
// CHECK-NEXT:     double a = 0;
// CHECK-NEXT:     _t0 = 0;
// CHECK-NEXT:     do {
// CHECK-NEXT:         _t0++;
// CHECK-NEXT:         int counter_again = 3;
// CHECK-NEXT:         clad::push(_t1, 0UL);
// CHECK-NEXT:         do {
// CHECK-NEXT:             clad::back(_t1)++;
// CHECK-NEXT:             a += clad::push(_t3, i) * clad::push(_t2, i) + j;
// CHECK-NEXT:             counter_again -= 1;
// CHECK-NEXT:             clad::push(_t4, 0UL);
// CHECK-NEXT:             do {
// CHECK-NEXT:                 clad::back(_t4)++;
// CHECK-NEXT:                 a += j;
// CHECK-NEXT:             } while (0);
// CHECK-NEXT:         } while (counter_again);
// CHECK-NEXT:         counter -= 1;
// CHECK-NEXT:     } while (counter);
// CHECK-NEXT:     _d_a += 1;
// CHECK-NEXT:     do {
// CHECK-NEXT:         {
// CHECK-NEXT:             {
// CHECK-NEXT:                 int _r_d3 = _d_counter;
// CHECK-NEXT:                 _d_counter += _r_d3;
// CHECK-NEXT:                 _d_counter -= _r_d3;
// CHECK-NEXT:             }
// CHECK-NEXT:             {
// CHECK-NEXT:                 do {
// CHECK-NEXT:                     {
// CHECK-NEXT:                         {
// CHECK-NEXT:                             do {
// CHECK-NEXT:                                 {
// CHECK-NEXT:                                     double _r_d2 = _d_a;
// CHECK-NEXT:                                     _d_a += _r_d2;
// CHECK-NEXT:                                     * _d_j += _r_d2;
// CHECK-NEXT:                                     _d_a -= _r_d2;
// CHECK-NEXT:                                 }
// CHECK-NEXT:                                 clad::back(_t4)--;
// CHECK-NEXT:                             } while (clad::back(_t4));
// CHECK-NEXT:                             clad::pop(_t4);
// CHECK-NEXT:                         }
// CHECK-NEXT:                         {
// CHECK-NEXT:                             int _r_d1 = _d_counter_again;
// CHECK-NEXT:                             _d_counter_again += _r_d1;
// CHECK-NEXT:                             _d_counter_again -= _r_d1;
// CHECK-NEXT:                         }
// CHECK-NEXT:                         {
// CHECK-NEXT:                             double _r_d0 = _d_a;
// CHECK-NEXT:                             _d_a += _r_d0;
// CHECK-NEXT:                             double _r0 = _r_d0 * clad::pop(_t2);
// CHECK-NEXT:                             * _d_i += _r0;
// CHECK-NEXT:                             double _r1 = clad::pop(_t3) * _r_d0;
// CHECK-NEXT:                             * _d_i += _r1;
// CHECK-NEXT:                             * _d_j += _r_d0;
// CHECK-NEXT:                             _d_a -= _r_d0;
// CHECK-NEXT:                         }
// CHECK-NEXT:                     }
// CHECK-NEXT:                     clad::back(_t1)--;
// CHECK-NEXT:                 } while (clad::back(_t1));
// CHECK-NEXT:                 clad::pop(_t1);
// CHECK-NEXT:             }
// CHECK-NEXT:             _d_counter_again = 0;
// CHECK-NEXT:         }
// CHECK-NEXT:         _t0--;
// CHECK-NEXT:     } while (_t0);
// CHECK-NEXT: }

double fn13(double i, double j) {
  double res = 0;
  int counter = 3;
  for (; int k = counter; counter-=1) {
    k += i + 2*j;
    double temp = k;
    res += temp;
  }
  return res;
}

// CHECK: void fn13_grad(double i, double j, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j) {
// CHECK-NEXT:     double _d_res = 0;
// CHECK-NEXT:     int _d_counter = 0;
// CHECK-NEXT:     unsigned long _t0;
// CHECK-NEXT:     int _d_k = 0;
// CHECK-NEXT:     clad::tape<double> _t1 = {};
// CHECK-NEXT:     double _d_temp = 0;
// CHECK-NEXT:     double res = 0;
// CHECK-NEXT:     int counter = 3;
// CHECK-NEXT:     _t0 = 0;
// CHECK-NEXT:     for (; k; counter -= 1) {
// CHECK-NEXT:         _t0++;
// CHECK-NEXT:         k += i + 2 * clad::push(_t1, j);
// CHECK-NEXT:         double temp = k;
// CHECK-NEXT:         res += temp;
// CHECK-NEXT:     }
// CHECK-NEXT:     _d_res += 1;
// CHECK-NEXT:     for (; _t0; _t0--) {
// CHECK-NEXT:         {
// CHECK-NEXT:             int _r_d0 = _d_counter;
// CHECK-NEXT:             _d_counter += _r_d0;
// CHECK-NEXT:             _d_counter -= _r_d0;
// CHECK-NEXT:         }
// CHECK-NEXT:         {
// CHECK-NEXT:             {
// CHECK-NEXT:                 double _r_d2 = _d_res;
// CHECK-NEXT:                 _d_res += _r_d2;
// CHECK-NEXT:                 _d_temp += _r_d2;
// CHECK-NEXT:                 _d_res -= _r_d2;
// CHECK-NEXT:             }
// CHECK-NEXT:             {
// CHECK-NEXT:                 _d_k += _d_temp;
// CHECK-NEXT:                 _d_temp = 0;
// CHECK-NEXT:             }
// CHECK-NEXT:             {
// CHECK-NEXT:                 int _r_d1 = _d_k;
// CHECK-NEXT:                 _d_k += _r_d1;
// CHECK-NEXT:                 * _d_i += _r_d1;
// CHECK-NEXT:                 double _r0 = _r_d1 * clad::pop(_t1);
// CHECK-NEXT:                 int _r1 = 2 * _r_d1;
// CHECK-NEXT:                 * _d_j += _r1;
// CHECK-NEXT:                 _d_k -= _r_d1;
// CHECK-NEXT:             }
// CHECK-NEXT:         }
// CHECK-NEXT:         {
// CHECK-NEXT:             _d_counter += _d_k;
// CHECK-NEXT:             _d_k = 0;
// CHECK-NEXT:         }
// CHECK-NEXT:     }
// CHECK-NEXT: }

double fn14(double i, double j) {
  int choice = 5;
  double res = 0;
  while (choice--) {
    if (choice > 3) {
      res += i;
      continue;
    }
    if (choice > 1) {
      res += j;
      continue;
    }

    if (choice > 0) {
      res += i * j;
      continue;
    }
  }
  return res;
}

// CHECK: void fn14_grad(double i, double j, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j) {
// CHECK-NEXT:     int _d_choice = 0;
// CHECK-NEXT:     double _d_res = 0;
// CHECK-NEXT:     unsigned long _t0;
// CHECK-NEXT:     clad::tape<bool> _t2 = {};
// CHECK-NEXT:     clad::tape<unsigned long> _t3 = {};
// CHECK-NEXT:     clad::tape<bool> _t5 = {};
// CHECK-NEXT:     clad::tape<bool> _t7 = {};
// CHECK-NEXT:     clad::tape<double> _t8 = {};
// CHECK-NEXT:     clad::tape<double> _t9 = {};
// CHECK-NEXT:     int choice = 5;
// CHECK-NEXT:     double res = 0;
// CHECK-NEXT:     _t0 = 0;
// CHECK-NEXT:     while (choice--)
// CHECK-NEXT:         {
// CHECK-NEXT:             _t0++;
// CHECK-NEXT:             bool _t1 = choice > 3;
// CHECK-NEXT:             {
// CHECK-NEXT:                 if (_t1) {
// CHECK-NEXT:                     res += i;
// CHECK-NEXT:                     {
// CHECK-NEXT:                         clad::push(_t3, 1UL);
// CHECK-NEXT:                         continue;
// CHECK-NEXT:                     }
// CHECK-NEXT:                 }
// CHECK-NEXT:                 clad::push(_t2, _t1);
// CHECK-NEXT:             }
// CHECK-NEXT:             bool _t4 = choice > 1;
// CHECK-NEXT:             {
// CHECK-NEXT:                 if (_t4) {
// CHECK-NEXT:                     res += j;
// CHECK-NEXT:                     {
// CHECK-NEXT:                         clad::push(_t3, 2UL);
// CHECK-NEXT:                         continue;
// CHECK-NEXT:                     }
// CHECK-NEXT:                 }
// CHECK-NEXT:                 clad::push(_t5, _t4);
// CHECK-NEXT:             }
// CHECK-NEXT:             bool _t6 = choice > 0;
// CHECK-NEXT:             {
// CHECK-NEXT:                 if (_t6) {
// CHECK-NEXT:                     res += clad::push(_t9, i) * clad::push(_t8, j);
// CHECK-NEXT:                     {
// CHECK-NEXT:                         clad::push(_t3, 3UL);
// CHECK-NEXT:                         continue;
// CHECK-NEXT:                     }
// CHECK-NEXT:                 }
// CHECK-NEXT:                 clad::push(_t7, _t6);
// CHECK-NEXT:             }
// CHECK-NEXT:             clad::push(_t3, 4UL);
// CHECK-NEXT:         }
// CHECK-NEXT:     _d_res += 1;
// CHECK-NEXT:     while (_t0)
// CHECK-NEXT:         {
// CHECK-NEXT:             switch (clad::pop(_t3)) {
// CHECK-NEXT:               case 4UL:
// CHECK-NEXT:                 ;
// CHECK-NEXT:                 if (clad::pop(_t7)) {
// CHECK-NEXT:                   case 3UL:
// CHECK-NEXT:                     ;
// CHECK-NEXT:                     {
// CHECK-NEXT:                         double _r_d2 = _d_res;
// CHECK-NEXT:                         _d_res += _r_d2;
// CHECK-NEXT:                         double _r0 = _r_d2 * clad::pop(_t8);
// CHECK-NEXT:                         * _d_i += _r0;
// CHECK-NEXT:                         double _r1 = clad::pop(_t9) * _r_d2;
// CHECK-NEXT:                         * _d_j += _r1;
// CHECK-NEXT:                         _d_res -= _r_d2;
// CHECK-NEXT:                     }
// CHECK-NEXT:                 }
// CHECK-NEXT:                 if (clad::pop(_t5)) {
// CHECK-NEXT:                   case 2UL:
// CHECK-NEXT:                     ;
// CHECK-NEXT:                     {
// CHECK-NEXT:                         double _r_d1 = _d_res;
// CHECK-NEXT:                         _d_res += _r_d1;
// CHECK-NEXT:                         * _d_j += _r_d1;
// CHECK-NEXT:                         _d_res -= _r_d1;
// CHECK-NEXT:                     }
// CHECK-NEXT:                 }
// CHECK-NEXT:                 if (clad::pop(_t2)) {
// CHECK-NEXT:                   case 1UL:
// CHECK-NEXT:                     ;
// CHECK-NEXT:                     {
// CHECK-NEXT:                         double _r_d0 = _d_res;
// CHECK-NEXT:                         _d_res += _r_d0;
// CHECK-NEXT:                         * _d_i += _r_d0;
// CHECK-NEXT:                         _d_res -= _r_d0;
// CHECK-NEXT:                     }
// CHECK-NEXT:                 }
// CHECK-NEXT:             }
// CHECK-NEXT:             _t0--;
// CHECK-NEXT:         }
// CHECK-NEXT: }

double fn15(double i, double j) {
  int choice = 5;
  double res = 0;
  while (choice--) {
    if (choice > 2)
      continue;
    int another_choice = 3;
    while (another_choice--) {
      if (another_choice > 1) {
        res += i;
        continue;
      }
      if (another_choice > 0) {
        res += j;
      }
    }
  }
  return res;
}

// CHECK: void fn15_grad(double i, double j, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j) {
// CHECK-NEXT:     int _d_choice = 0;
// CHECK-NEXT:     double _d_res = 0;
// CHECK-NEXT:     unsigned long _t0;
// CHECK-NEXT:     clad::tape<bool> _t2 = {};
// CHECK-NEXT:     clad::tape<unsigned long> _t3 = {};
// CHECK-NEXT:     int _d_another_choice = 0;
// CHECK-NEXT:     clad::tape<unsigned long> _t4 = {};
// CHECK-NEXT:     clad::tape<bool> _t6 = {};
// CHECK-NEXT:     clad::tape<unsigned long> _t7 = {};
// CHECK-NEXT:     clad::tape<bool> _t9 = {};
// CHECK-NEXT:     int choice = 5;
// CHECK-NEXT:     double res = 0;
// CHECK-NEXT:     _t0 = 0;
// CHECK-NEXT:     while (choice--)
// CHECK-NEXT:         {
// CHECK-NEXT:             _t0++;
// CHECK-NEXT:             bool _t1 = choice > 2;
// CHECK-NEXT:             {
// CHECK-NEXT:                 if (_t1) {
// CHECK-NEXT:                     clad::push(_t3, 1UL);
// CHECK-NEXT:                     continue;
// CHECK-NEXT:                 }
// CHECK-NEXT:                 clad::push(_t2, _t1);
// CHECK-NEXT:             }
// CHECK-NEXT:             int another_choice = 3;
// CHECK-NEXT:             clad::push(_t4, 0UL);
// CHECK-NEXT:             while (another_choice--)
// CHECK-NEXT:                 {
// CHECK-NEXT:                     clad::back(_t4)++;
// CHECK-NEXT:                     bool _t5 = another_choice > 1;
// CHECK-NEXT:                     {
// CHECK-NEXT:                         if (_t5) {
// CHECK-NEXT:                             res += i;
// CHECK-NEXT:                             {
// CHECK-NEXT:                                 clad::push(_t7, 1UL);
// CHECK-NEXT:                                 continue;
// CHECK-NEXT:                             }
// CHECK-NEXT:                         }
// CHECK-NEXT:                         clad::push(_t6, _t5);
// CHECK-NEXT:                     }
// CHECK-NEXT:                     bool _t8 = another_choice > 0;
// CHECK-NEXT:                     {
// CHECK-NEXT:                         if (_t8) {
// CHECK-NEXT:                             res += j;
// CHECK-NEXT:                         }
// CHECK-NEXT:                         clad::push(_t9, _t8);
// CHECK-NEXT:                     }
// CHECK-NEXT:                     clad::push(_t7, 2UL);
// CHECK-NEXT:                 }
// CHECK-NEXT:             clad::push(_t3, 2UL);
// CHECK-NEXT:         }
// CHECK-NEXT:     _d_res += 1;
// CHECK-NEXT:     while (_t0)
// CHECK-NEXT:         {
// CHECK-NEXT:             switch (clad::pop(_t3)) {
// CHECK-NEXT:               case 2UL:
// CHECK-NEXT:                 ;
// CHECK-NEXT:                 {
// CHECK-NEXT:                     while (clad::back(_t4))
// CHECK-NEXT:                         {
// CHECK-NEXT:                             switch (clad::pop(_t7)) {
// CHECK-NEXT:                               case 2UL:
// CHECK-NEXT:                                 ;
// CHECK-NEXT:                                 if (clad::pop(_t9)) {
// CHECK-NEXT:                                     {
// CHECK-NEXT:                                         double _r_d1 = _d_res;
// CHECK-NEXT:                                         _d_res += _r_d1;
// CHECK-NEXT:                                         * _d_j += _r_d1;
// CHECK-NEXT:                                         _d_res -= _r_d1;
// CHECK-NEXT:                                     }
// CHECK-NEXT:                                 }
// CHECK-NEXT:                                 if (clad::pop(_t6)) {
// CHECK-NEXT:                                   case 1UL:
// CHECK-NEXT:                                     ;
// CHECK-NEXT:                                     {
// CHECK-NEXT:                                         double _r_d0 = _d_res;
// CHECK-NEXT:                                         _d_res += _r_d0;
// CHECK-NEXT:                                         * _d_i += _r_d0;
// CHECK-NEXT:                                         _d_res -= _r_d0;
// CHECK-NEXT:                                     }
// CHECK-NEXT:                                 }
// CHECK-NEXT:                             }
// CHECK-NEXT:                             clad::back(_t4)--;
// CHECK-NEXT:                         }
// CHECK-NEXT:                     clad::pop(_t4);
// CHECK-NEXT:                 }
// CHECK-NEXT:                 _d_another_choice = 0;
// CHECK-NEXT:                 if (clad::pop(_t2))
// CHECK-NEXT:                   case 1UL:
// CHECK-NEXT:                     ;
// CHECK-NEXT:             }
// CHECK-NEXT:             _t0--;
// CHECK-NEXT:         }
// CHECK-NEXT: }

double fn16(double i, double j) {
  int counter = 5;
  double res=0;
  for (int ii=0; ii<counter; ++ii) {
    if (ii == 4) {
      res += i*j;
      break;
    }
    if (ii > 2) {
      res += 2*i;
      continue;
    }
    res += i + j;
  }
  return res;
}

// CHECK: void fn16_grad(double i, double j, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j) {
// CHECK-NEXT:     int _d_counter = 0;
// CHECK-NEXT:     double _d_res = 0;
// CHECK-NEXT:     unsigned long _t0;
// CHECK-NEXT:     int _d_ii = 0;
// CHECK-NEXT:     clad::tape<bool> _t2 = {};
// CHECK-NEXT:     clad::tape<double> _t3 = {};
// CHECK-NEXT:     clad::tape<double> _t4 = {};
// CHECK-NEXT:     clad::tape<unsigned long> _t5 = {};
// CHECK-NEXT:     clad::tape<bool> _t7 = {};
// CHECK-NEXT:     clad::tape<double> _t8 = {};
// CHECK-NEXT:     int counter = 5;
// CHECK-NEXT:     double res = 0;
// CHECK-NEXT:     _t0 = 0;
// CHECK-NEXT:     for (int ii = 0; ii < counter; ++ii) {
// CHECK-NEXT:         _t0++;
// CHECK-NEXT:         bool _t1 = ii == 4;
// CHECK-NEXT:         {
// CHECK-NEXT:             if (_t1) {
// CHECK-NEXT:                 res += clad::push(_t4, i) * clad::push(_t3, j);
// CHECK-NEXT:                 {
// CHECK-NEXT:                     clad::push(_t5, 1UL);
// CHECK-NEXT:                     break;
// CHECK-NEXT:                 }
// CHECK-NEXT:             }
// CHECK-NEXT:             clad::push(_t2, _t1);
// CHECK-NEXT:         }
// CHECK-NEXT:         bool _t6 = ii > 2;
// CHECK-NEXT:         {
// CHECK-NEXT:             if (_t6) {
// CHECK-NEXT:                 res += 2 * clad::push(_t8, i);
// CHECK-NEXT:                 {
// CHECK-NEXT:                     clad::push(_t5, 2UL);
// CHECK-NEXT:                     continue;
// CHECK-NEXT:                 }
// CHECK-NEXT:             }
// CHECK-NEXT:             clad::push(_t7, _t6);
// CHECK-NEXT:         }
// CHECK-NEXT:         res += i + j;
// CHECK-NEXT:         clad::push(_t5, 3UL);
// CHECK-NEXT:     }
// CHECK-NEXT:     _d_res += 1;
// CHECK-NEXT:     for (; _t0; _t0--)
// CHECK-NEXT:         switch (clad::pop(_t5)) {
// CHECK-NEXT:           case 3UL:
// CHECK-NEXT:             ;
// CHECK-NEXT:             {
// CHECK-NEXT:                 double _r_d2 = _d_res;
// CHECK-NEXT:                 _d_res += _r_d2;
// CHECK-NEXT:                 * _d_i += _r_d2;
// CHECK-NEXT:                 * _d_j += _r_d2;
// CHECK-NEXT:                 _d_res -= _r_d2;
// CHECK-NEXT:             }
// CHECK-NEXT:             if (clad::pop(_t7)) {
// CHECK-NEXT:               case 2UL:
// CHECK-NEXT:                 ;
// CHECK-NEXT:                 {
// CHECK-NEXT:                     double _r_d1 = _d_res;
// CHECK-NEXT:                     _d_res += _r_d1;
// CHECK-NEXT:                     double _r2 = _r_d1 * clad::pop(_t8);
// CHECK-NEXT:                     double _r3 = 2 * _r_d1;
// CHECK-NEXT:                     * _d_i += _r3;
// CHECK-NEXT:                     _d_res -= _r_d1;
// CHECK-NEXT:                 }
// CHECK-NEXT:             }
// CHECK-NEXT:             if (clad::pop(_t2)) {
// CHECK-NEXT:               case 1UL:
// CHECK-NEXT:                 ;
// CHECK-NEXT:                 {
// CHECK-NEXT:                     double _r_d0 = _d_res;
// CHECK-NEXT:                     _d_res += _r_d0;
// CHECK-NEXT:                     double _r0 = _r_d0 * clad::pop(_t3);
// CHECK-NEXT:                     * _d_i += _r0;
// CHECK-NEXT:                     double _r1 = clad::pop(_t4) * _r_d0;
// CHECK-NEXT:                     * _d_j += _r1;
// CHECK-NEXT:                     _d_res -= _r_d0;
// CHECK-NEXT:                 }
// CHECK-NEXT:             }
// CHECK-NEXT:         }
// CHECK-NEXT: }

double fn17(double i, double j) {
  int counter = 5;
  double res = 0;
  for (int ii=0; ii<counter; ++ii) {
    int jj = ii;
    if (ii < 2)
      continue;
    while (jj--) {
      if (jj < 3) {
        res += i*j;
        break;
      } else {
        continue;
      }
      res += i*i*j*j;
    }
  }
  return res;
}

// CHECK: void fn17_grad(double i, double j, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j) {
// CHECK-NEXT:     int _d_counter = 0;
// CHECK-NEXT:     double _d_res = 0;
// CHECK-NEXT:     unsigned long _t0;
// CHECK-NEXT:     int _d_ii = 0;
// CHECK-NEXT:     int _d_jj = 0;
// CHECK-NEXT:     clad::tape<bool> _t2 = {};
// CHECK-NEXT:     clad::tape<unsigned long> _t3 = {};
// CHECK-NEXT:     clad::tape<unsigned long> _t4 = {};
// CHECK-NEXT:     clad::tape<bool> _t6 = {};
// CHECK-NEXT:     clad::tape<double> _t7 = {};
// CHECK-NEXT:     clad::tape<double> _t8 = {};
// CHECK-NEXT:     clad::tape<unsigned long> _t9 = {};
// CHECK-NEXT:     clad::tape<double> _t10 = {};
// CHECK-NEXT:     clad::tape<double> _t11 = {};
// CHECK-NEXT:     clad::tape<double> _t12 = {};
// CHECK-NEXT:     clad::tape<double> _t13 = {};
// CHECK-NEXT:     clad::tape<double> _t14 = {};
// CHECK-NEXT:     clad::tape<double> _t15 = {};
// CHECK-NEXT:     int counter = 5;
// CHECK-NEXT:     double res = 0;
// CHECK-NEXT:     _t0 = 0;
// CHECK-NEXT:     for (int ii = 0; ii < counter; ++ii) {
// CHECK-NEXT:         _t0++;
// CHECK-NEXT:         int jj = ii;
// CHECK-NEXT:         bool _t1 = ii < 2;
// CHECK-NEXT:         {
// CHECK-NEXT:             if (_t1) {
// CHECK-NEXT:                 clad::push(_t3, 1UL);
// CHECK-NEXT:                 continue;
// CHECK-NEXT:             }
// CHECK-NEXT:             clad::push(_t2, _t1);
// CHECK-NEXT:         }
// CHECK-NEXT:         clad::push(_t4, 0UL);
// CHECK-NEXT:         while (jj--)
// CHECK-NEXT:             {
// CHECK-NEXT:                 clad::back(_t4)++;
// CHECK-NEXT:                 bool _t5 = jj < 3;
// CHECK-NEXT:                 {
// CHECK-NEXT:                     if (_t5) {
// CHECK-NEXT:                         res += clad::push(_t8, i) * clad::push(_t7, j);
// CHECK-NEXT:                         {
// CHECK-NEXT:                             clad::push(_t9, 1UL);
// CHECK-NEXT:                             break;
// CHECK-NEXT:                         }
// CHECK-NEXT:                     } else {
// CHECK-NEXT:                         {
// CHECK-NEXT:                             clad::push(_t9, 2UL);
// CHECK-NEXT:                             continue;
// CHECK-NEXT:                         }
// CHECK-NEXT:                     }
// CHECK-NEXT:                     clad::push(_t6, _t5);
// CHECK-NEXT:                 }
// CHECK-NEXT:                 res += clad::push(_t15, clad::push(_t14, clad::push(_t13, i) * clad::push(_t12, i)) * clad::push(_t11, j)) * clad::push(_t10, j);
// CHECK-NEXT:                 clad::push(_t9, 3UL);
// CHECK-NEXT:             }
// CHECK-NEXT:         clad::push(_t3, 2UL);
// CHECK-NEXT:     }
// CHECK-NEXT:     _d_res += 1;
// CHECK-NEXT:     for (; _t0; _t0--)
// CHECK-NEXT:         switch (clad::pop(_t3)) {
// CHECK-NEXT:           case 2UL:
// CHECK-NEXT:             ;
// CHECK-NEXT:             {
// CHECK-NEXT:                 while (clad::back(_t4))
// CHECK-NEXT:                     {
// CHECK-NEXT:                         switch (clad::pop(_t9)) {
// CHECK-NEXT:                           case 3UL:
// CHECK-NEXT:                             ;
// CHECK-NEXT:                             {
// CHECK-NEXT:                                 double _r_d1 = _d_res;
// CHECK-NEXT:                                 _d_res += _r_d1;
// CHECK-NEXT:                                 double _r2 = _r_d1 * clad::pop(_t10);
// CHECK-NEXT:                                 double _r3 = _r2 * clad::pop(_t11);
// CHECK-NEXT:                                 double _r4 = _r3 * clad::pop(_t12);
// CHECK-NEXT:                                 * _d_i += _r4;
// CHECK-NEXT:                                 double _r5 = clad::pop(_t13) * _r3;
// CHECK-NEXT:                                 * _d_i += _r5;
// CHECK-NEXT:                                 double _r6 = clad::pop(_t14) * _r2;
// CHECK-NEXT:                                 * _d_j += _r6;
// CHECK-NEXT:                                 double _r7 = clad::pop(_t15) * _r_d1;
// CHECK-NEXT:                                 * _d_j += _r7;
// CHECK-NEXT:                                 _d_res -= _r_d1;
// CHECK-NEXT:                             }
// CHECK-NEXT:                             if (clad::pop(_t6)) {
// CHECK-NEXT:                               case 1UL:
// CHECK-NEXT:                                 ;
// CHECK-NEXT:                                 {
// CHECK-NEXT:                                     double _r_d0 = _d_res;
// CHECK-NEXT:                                     _d_res += _r_d0;
// CHECK-NEXT:                                     double _r0 = _r_d0 * clad::pop(_t7);
// CHECK-NEXT:                                     * _d_i += _r0;
// CHECK-NEXT:                                     double _r1 = clad::pop(_t8) * _r_d0;
// CHECK-NEXT:                                     * _d_j += _r1;
// CHECK-NEXT:                                     _d_res -= _r_d0;
// CHECK-NEXT:                                 }
// CHECK-NEXT:                             } else {
// CHECK-NEXT:                               case 2UL:
// CHECK-NEXT:                                 ;
// CHECK-NEXT:                             }
// CHECK-NEXT:                         }
// CHECK-NEXT:                         clad::back(_t4)--;
// CHECK-NEXT:                     }
// CHECK-NEXT:                 clad::pop(_t4);
// CHECK-NEXT:             }
// CHECK-NEXT:             if (clad::pop(_t2))
// CHECK-NEXT:               case 1UL:
// CHECK-NEXT:                 ;
// CHECK-NEXT:             {
// CHECK-NEXT:                 _d_ii += _d_jj;
// CHECK-NEXT:                 _d_jj = 0;
// CHECK-NEXT:             }
// CHECK-NEXT:         }
// CHECK-NEXT: }

double fn18(double i, double j) {
  int choice = 5;
  double res = 0;
  for (int counter=0; counter<choice; ++counter)
    if (counter < 2)
      res += i+j;
    else if (counter < 4)
      continue;
    else {
      res += 2*i + 2*j;
      break;
    }
  return res;
}

// CHECK: void fn18_grad(double i, double j, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j) {
// CHECK-NEXT:     int _d_choice = 0;
// CHECK-NEXT:     double _d_res = 0;
// CHECK-NEXT:     unsigned long _t0;
// CHECK-NEXT:     int _d_counter = 0;
// CHECK-NEXT:     clad::tape<bool> _t2 = {};
// CHECK-NEXT:     clad::tape<bool> _t4 = {};
// CHECK-NEXT:     clad::tape<unsigned long> _t5 = {};
// CHECK-NEXT:     clad::tape<double> _t6 = {};
// CHECK-NEXT:     clad::tape<double> _t7 = {};
// CHECK-NEXT:     int choice = 5;
// CHECK-NEXT:     double res = 0;
// CHECK-NEXT:     _t0 = 0;
// CHECK-NEXT:     for (int counter = 0; counter < choice; ++counter) {
// CHECK-NEXT:         _t0++;
// CHECK-NEXT:         bool _t1 = counter < 2;
// CHECK-NEXT:         {
// CHECK-NEXT:             if (_t1)
// CHECK-NEXT:                 res += i + j;
// CHECK-NEXT:             else {
// CHECK-NEXT:                 bool _t3 = counter < 4;
// CHECK-NEXT:                 {
// CHECK-NEXT:                     if (_t3) {
// CHECK-NEXT:                         clad::push(_t5, 1UL);
// CHECK-NEXT:                         continue;
// CHECK-NEXT:                     } else {
// CHECK-NEXT:                         res += 2 * clad::push(_t6, i) + 2 * clad::push(_t7, j);
// CHECK-NEXT:                         {
// CHECK-NEXT:                             clad::push(_t5, 2UL);
// CHECK-NEXT:                             break;
// CHECK-NEXT:                         }
// CHECK-NEXT:                     }
// CHECK-NEXT:                     clad::push(_t4, _t3);
// CHECK-NEXT:                 }
// CHECK-NEXT:             }
// CHECK-NEXT:             clad::push(_t2, _t1);
// CHECK-NEXT:         }
// CHECK-NEXT:         clad::push(_t5, 3UL);
// CHECK-NEXT:     }
// CHECK-NEXT:     _d_res += 1;
// CHECK-NEXT:     for (; _t0; _t0--)
// CHECK-NEXT:         switch (clad::pop(_t5)) {
// CHECK-NEXT:           case 3UL:
// CHECK-NEXT:             ;
// CHECK-NEXT:             if (clad::pop(_t2)) {
// CHECK-NEXT:                 double _r_d0 = _d_res;
// CHECK-NEXT:                 _d_res += _r_d0;
// CHECK-NEXT:                 * _d_i += _r_d0;
// CHECK-NEXT:                 * _d_j += _r_d0;
// CHECK-NEXT:                 _d_res -= _r_d0;
// CHECK-NEXT:             } else if (clad::pop(_t4))
// CHECK-NEXT:               case 1UL:
// CHECK-NEXT:                 ;
// CHECK-NEXT:             else {
// CHECK-NEXT:               case 2UL:
// CHECK-NEXT:                 ;
// CHECK-NEXT:                 {
// CHECK-NEXT:                     double _r_d1 = _d_res;
// CHECK-NEXT:                     _d_res += _r_d1;
// CHECK-NEXT:                     double _r0 = _r_d1 * clad::pop(_t6);
// CHECK-NEXT:                     double _r1 = 2 * _r_d1;
// CHECK-NEXT:                     * _d_i += _r1;
// CHECK-NEXT:                     double _r2 = _r_d1 * clad::pop(_t7);
// CHECK-NEXT:                     double _r3 = 2 * _r_d1;
// CHECK-NEXT:                     * _d_j += _r3;
// CHECK-NEXT:                     _d_res -= _r_d1;
// CHECK-NEXT:                 }
// CHECK-NEXT:             }
// CHECK-NEXT:         }
// CHECK-NEXT: }

double fn19(double* arr, int n) {
  double res = 0;
  for (int i=0; i<n; ++i) {
    double& ref = arr[i];
    res += ref;
  }
  return res;
}

// CHECK: void fn19_grad_0(double *arr, int n, clad::array_ref<double> _d_arr) {
// CHECK-NEXT:     int _d_n = 0;
// CHECK-NEXT:     double _d_res = 0;
// CHECK-NEXT:     unsigned long _t0;
// CHECK-NEXT:     int _d_i = 0;
// CHECK-NEXT:     clad::tape<int> _t1 = {};
// CHECK-NEXT:     clad::tape<double *> _t3 = {};
// CHECK-NEXT:     double *_d_ref = 0;
// CHECK-NEXT:     double res = 0;
// CHECK-NEXT:     _t0 = 0;
// CHECK-NEXT:     for (int i = 0; i < n; ++i) {
// CHECK-NEXT:         _t0++;
// CHECK-NEXT:         _d_ref = &_d_arr[i];
// CHECK-NEXT:         clad::push(_t3, _d_ref);
// CHECK-NEXT:         double &ref = arr[clad::push(_t1, i)];
// CHECK-NEXT:         res += ref;
// CHECK-NEXT:     }
// CHECK-NEXT:     _d_res += 1;
// CHECK-NEXT:     for (; _t0; _t0--) {
// CHECK-NEXT:         double *_t4 = clad::pop(_t3);
// CHECK-NEXT:         {
// CHECK-NEXT:             int _t2 = clad::pop(_t1);
// CHECK-NEXT:             double _r_d0 = _d_res;
// CHECK-NEXT:             _d_res += _r_d0;
// CHECK-NEXT:             *_t4 += _r_d0;
// CHECK-NEXT:             _d_res -= _r_d0;
// CHECK-NEXT:         }
// CHECK-NEXT:     }
// CHECK-NEXT: }

double f_loop_init_var(double lower, double upper) {
  double sum = 0;
  double num_points = 10000;
  double interval = (upper - lower) / num_points;
  for (double x = lower; x <= upper; x += interval) {
    sum += x * x * interval;
  }
  return sum;
}

// CHECK: void f_loop_init_var_grad(double lower, double upper, clad::array_ref<double> _d_lower, clad::array_ref<double> _d_upper) {
// CHECK-NEXT:     double _d_sum = 0;
// CHECK-NEXT:     double _d_num_points = 0;
// CHECK-NEXT:     double _t0;
// CHECK-NEXT:     double _t1;
// CHECK-NEXT:     double _d_interval = 0;
// CHECK-NEXT:     unsigned long _t2;
// CHECK-NEXT:     double _d_x = 0;
// CHECK-NEXT:     clad::tape<double> _t3 = {};
// CHECK-NEXT:     clad::tape<double> _t4 = {};
// CHECK-NEXT:     clad::tape<double> _t5 = {};
// CHECK-NEXT:     clad::tape<double> _t6 = {};
// CHECK-NEXT:     double sum = 0;
// CHECK-NEXT:     double num_points = 10000;
// CHECK-NEXT:     _t1 = (upper - lower);
// CHECK-NEXT:     _t0 = num_points;
// CHECK-NEXT:     double interval = _t1 / _t0;
// CHECK-NEXT:     _t2 = 0;
// CHECK-NEXT:     for (double x = lower; x <= upper; x += interval) {
// CHECK-NEXT:         _t2++;
// CHECK-NEXT:         sum += clad::push(_t6, clad::push(_t5, x) * clad::push(_t4, x)) * clad::push(_t3, interval);
// CHECK-NEXT:     }
// CHECK-NEXT:     _d_sum += 1;
// CHECK-NEXT:     {
// CHECK-NEXT:         for (; _t2; _t2--) {
// CHECK-NEXT:             {
// CHECK-NEXT:                 double _r_d0 = _d_x;
// CHECK-NEXT:                 _d_x += _r_d0;
// CHECK-NEXT:                 _d_interval += _r_d0;
// CHECK-NEXT:                 _d_x -= _r_d0;
// CHECK-NEXT:             }
// CHECK-NEXT:             {
// CHECK-NEXT:                 {
// CHECK-NEXT:                     double _r_d1 = _d_sum;
// CHECK-NEXT:                     _d_sum += _r_d1;
// CHECK-NEXT:                     double _r2 = _r_d1 * clad::pop(_t3);
// CHECK-NEXT:                     double _r3 = _r2 * clad::pop(_t4);
// CHECK-NEXT:                     _d_x += _r3;
// CHECK-NEXT:                     double _r4 = clad::pop(_t5) * _r2;
// CHECK-NEXT:                     _d_x += _r4;
// CHECK-NEXT:                     double _r5 = clad::pop(_t6) * _r_d1;
// CHECK-NEXT:                     _d_interval += _r5;
// CHECK-NEXT:                     _d_sum -= _r_d1;
// CHECK-NEXT:                 }
// CHECK-NEXT:             }
// CHECK-NEXT:         }
// CHECK-NEXT:         * _d_lower += _d_x;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r0 = _d_interval / _t0;
// CHECK-NEXT:         * _d_upper += _r0;
// CHECK-NEXT:         * _d_lower += -_r0;
// CHECK-NEXT:         double _r1 = _d_interval * -_t1 / (_t0 * _t0);
// CHECK-NEXT:         _d_num_points += _r1;
// CHECK-NEXT:     }
// CHECK-NEXT: }

#define TEST(F, x) { \
  result[0] = 0; \
  auto F##grad = clad::gradient(F);\
  F##grad.execute(x, result);\
  printf("{%.2f}\n", result[0]); \
}

#define TEST_2(F, x, y)                                                        \
  {                                                                            \
    result[0] = result[1] = 0;                                                 \
    auto d_##F = clad::gradient(F);                                            \
    d_##F.execute(x, y, result, result + 1);                                   \
    printf("{%.2f, %.2f}\n", result[0], result[1]);                            \
  }

int main() {
  double result[5] = {};
  clad::array_ref<double> result_ref(result, 5);
  TEST(f1, 3); // CHECK-EXEC: {27.00}
  TEST(f2, 3); // CHECK-EXEC: {59049.00}
  TEST(f3, 3); // CHECK-EXEC: {6.00}
  TEST(f4, 3); // CHECK-EXEC: {27.00}
  TEST(f5, 3); // CHECK-EXEC: {1.00}

  double p[] = { 1, 2, 3, 4, 5 };

  for (int i = 0; i < 5; i++) result[i] = 0;
  auto f_sum_grad = clad::gradient(f_sum, "p");
  f_sum_grad.execute(p, 5, result_ref);
  printf("{%.2f, %.2f, %.2f, %.2f, %.2f}\n", result[0], result[1], result[2], result[3], result[4]); // CHECK-EXEC: {1.00, 1.00, 1.00, 1.00, 1.00}

  for (int i = 0; i < 5; i++) result[i] = 0;
  auto f_sum_squares_grad = clad::gradient(f_sum_squares, "p");
  f_sum_squares_grad.execute(p, 5, result_ref);
  printf("{%.2f, %.2f, %.2f, %.2f, %.2f}\n", result[0], result[1], result[2], result[3], result[4]); // CHECK-EXEC: {2.00, 4.00, 6.00, 8.00, 10.00}

  for (int i = 0; i < 5; i++) result[i] = 0;
  auto f_log_gaus_d_means = clad::gradient(f_log_gaus, "p"); // == { (x[i] - p[i])/sigma^2 }
  double x[] = { 1, 1, 1, 1, 1 };
  f_log_gaus_d_means.execute(x, p, 5, 2.0, result_ref);
  printf("{%.2f, %.2f, %.2f, %.2f, %.2f}\n", result[0], result[1], result[2], result[3], result[4]); // CHECK-EXEC: {0.00, -0.25, -0.50, -0.75, -1.00}

  TEST_2(f_const, 2, 3);  // CHECK-EXEC: {0.00, 12.00}
  TEST_2(f6, 3, 5);       // CHECK-EXEC: {21.00, 33.00}
  TEST_2(fn7, 3, 5);      // CHECK-EXEC: {18.00, 3.00}
  TEST_2(fn8, 3, 5);      // CHECK-EXEC: {18.00, 3.00}
  TEST_2(fn9, 3, 5);      // CHECK-EXEC: {54.00, 9.00}
  TEST_2(fn10, 3, 5);     // CHECK-EXEC: {18.00, 3.00}
  TEST_2(fn11, 3, 5);     // CHECK-EXEC: {18.00, 3.00}
  TEST_2(fn12, 3, 5);     // CHECK-EXEC: {54.00, 18.00}
  TEST_2(fn13, 3, 5);     // CHECK-EXEC: {3.00, 6.00}
  TEST_2(fn14, 3, 5);     // CHECK-EXEC: {6.00, 5.00}
  TEST_2(fn15, 3, 5);     // CHECK-EXEC: {3.00, 3.00}
  TEST_2(fn16, 3, 5);     // CHECK-EXEC: {10.00, 6.00}
  TEST_2(fn17, 3, 5);     // CHECK-EXEC: {15.00, 9.00}
  TEST_2(fn18, 3, 5);     // CHECK-EXEC: {4.00, 4.00}

  INIT_GRADIENT(fn19, "arr");

  double arr[5] = {};
  double d_arr[5] = {};
  clad::array_ref<double> ref(d_arr, 5);

  TEST_GRADIENT(fn19, 1, arr, 5, d_arr);
  TEST_2(f_loop_init_var, 1, 2); // CHECK-EXEC: {-1.00, 4.00}
}
