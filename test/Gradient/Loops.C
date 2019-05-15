// RUN: %cladclang %s -I%S/../../include -oReverseLoops.out 2>&1 -lstdc++ -lm | FileCheck %s
// RUN: ./ReverseLoops.out | FileCheck -check-prefix=CHECK-EXEC %s
//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"
#include <cmath>

double f1(double x) {
  double t = 1;
  for (int i = 0; i < 3; i++)
    t *= x;
  return t;
} // == x^3

//CHECK:   void f1_grad(double x, double *_result) {
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
//CHECK-NEXT:       double f1_return = t;
//CHECK-NEXT:       goto _label0;
//CHECK-NEXT:     _label0:
//CHECK-NEXT:       _d_t += 1;
//CHECK-NEXT:       for (; _t0; _t0--) {
//CHECK-NEXT:           double _r_d0 = _d_t;
//CHECK-NEXT:           _d_t += _r_d0 * clad::pop(_t1);
//CHECK-NEXT:           double _r0 = clad::pop(_t2) * _r_d0;
//CHECK-NEXT:           _result[0UL] += _r0;
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

//CHECK:   void f2_grad(double x, double *_result) {
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
//CHECK-NEXT:       double f2_return = t;
//CHECK-NEXT:       goto _label0;
//CHECK-NEXT:     _label0:
//CHECK-NEXT:       _d_t += 1;
//CHECK-NEXT:       for (; _t0; _t0--) {
//CHECK-NEXT:           for (; clad::back(_t1); clad::back(_t1)--) {
//CHECK-NEXT:               double _r_d0 = _d_t;
//CHECK-NEXT:               _d_t += _r_d0 * clad::pop(_t2);
//CHECK-NEXT:               double _r0 = clad::pop(_t3) * _r_d0;
//CHECK-NEXT:               _result[0UL] += _r0;
//CHECK-NEXT:               _d_t -= _r_d0;
//CHECK-NEXT:           }
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

//CHECK:   void f3_grad(double x, double *_result) {
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
//CHECK-NEXT:               if (_t3) {
//CHECK-NEXT:                   double f3_return = t;
//CHECK-NEXT:                   goto _label0;
//CHECK-NEXT:               }
//CHECK-NEXT:               clad::push(_t4, _t3);
//CHECK-NEXT:           }
//CHECK-NEXT:       }
//CHECK-NEXT:       double f3_return = t;
//CHECK-NEXT:       goto _label1;
//CHECK-NEXT:     _label1:
//CHECK-NEXT:       _d_t += 1;
//CHECK-NEXT:       for (; _t0; _t0--) {
//CHECK-NEXT:           if (clad::pop(_t4))
//CHECK-NEXT:             _label0:
//CHECK-NEXT:               _d_t += 1;
//CHECK-NEXT:           {
//CHECK-NEXT:               double _r_d0 = _d_t;
//CHECK-NEXT:               _d_t += _r_d0 * clad::pop(_t1);
//CHECK-NEXT:               double _r0 = clad::pop(_t2) * _r_d0;
//CHECK-NEXT:               _result[0UL] += _r0;
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

//CHECK:   void f4_grad(double x, double *_result) {
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
//CHECK-NEXT:       double f4_return = t;
//CHECK-NEXT:       goto _label0;
//CHECK-NEXT:     _label0:
//CHECK-NEXT:       _d_t += 1;
//CHECK-NEXT:       for (; _t0; _t0--) {
//CHECK-NEXT:           double _r_d0 = _d_t;
//CHECK-NEXT:           _d_t += _r_d0 * clad::pop(_t1);
//CHECK-NEXT:           double _r0 = clad::pop(_t2) * _r_d0;
//CHECK-NEXT:           _result[0UL] += _r0;
//CHECK-NEXT:           _d_t -= _r_d0;
//CHECK-NEXT:       }
//CHECK-NEXT:   }

double f_sum(double *p, int n) {
  double s = 0;
  for (int i = 0; i < n; i++)
    s += p[i];
  return s;
}

//CHECK:   void f_sum_grad_0(double *p, int n, double *_result) {
//CHECK-NEXT:       double _d_s = 0;
//CHECK-NEXT:       unsigned long _t0;
//CHECK-NEXT:       int _d_i = 0;
//CHECK-NEXT:       clad::tape<int> _t1 = {};
//CHECK-NEXT:       double s = 0;
//CHECK-NEXT:       _t0 = 0;
//CHECK-NEXT:       for (int i = 0; i < n; i++) {
//CHECK-NEXT:           _t0++;
//CHECK-NEXT:           s += p[clad::push(_t1, i)];
//CHECK-NEXT:       }
//CHECK-NEXT:       double f_sum_return = s;
//CHECK-NEXT:       goto _label0;
//CHECK-NEXT:     _label0:
//CHECK-NEXT:       _d_s += 1;
//CHECK-NEXT:       for (; _t0; _t0--) {
//CHECK-NEXT:           double _r_d0 = _d_s;
//CHECK-NEXT:           _d_s += _r_d0;
//CHECK-NEXT:           _result[clad::pop(_t1)] += _r_d0;
//CHECK-NEXT:           _d_s -= _r_d0;
//CHECK-NEXT:       }
//CHECK-NEXT:   }

double sq(double x) { return x * x; }
//CHECK:   void sq_grad(double x, double *_result) {
//CHECK-NEXT:       double _t0;
//CHECK-NEXT:       double _t1;
//CHECK-NEXT:       _t1 = x;
//CHECK-NEXT:       _t0 = x;
//CHECK-NEXT:       double sq_return = _t1 * _t0;
//CHECK-NEXT:       goto _label0;
//CHECK-NEXT:     _label0:
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r0 = 1 * _t0;
//CHECK-NEXT:           _result[0UL] += _r0;
//CHECK-NEXT:           double _r1 = _t1 * 1;
//CHECK-NEXT:           _result[0UL] += _r1;
//CHECK-NEXT:       }
//CHECK-NEXT:   }

double f_sum_squares(double *p, int n) {
  double s = 0;
  for (int i = 0; i < n; i++)
    s += sq(p[i]);
  return s;
}

//CHECK:   void f_sum_squares_grad_0(double *p, int n, double *_result) {
//CHECK-NEXT:       double _d_s = 0;
//CHECK-NEXT:       unsigned long _t0;
//CHECK-NEXT:       int _d_i = 0;
//CHECK-NEXT:       clad::tape<int> _t1 = {};
//CHECK-NEXT:       clad::tape<double> _t2 = {};
//CHECK-NEXT:       double s = 0;
//CHECK-NEXT:       _t0 = 0;
//CHECK-NEXT:       for (int i = 0; i < n; i++) {
//CHECK-NEXT:           _t0++;
//CHECK-NEXT:           s += sq(clad::push(_t2, p[clad::push(_t1, i)]));
//CHECK-NEXT:       }
//CHECK-NEXT:       double f_sum_squares_return = s;
//CHECK-NEXT:       goto _label0;
//CHECK-NEXT:     _label0:
//CHECK-NEXT:       _d_s += 1;
//CHECK-NEXT:       for (; _t0; _t0--) {
//CHECK-NEXT:           double _r_d0 = _d_s;
//CHECK-NEXT:           _d_s += _r_d0;
//CHECK-NEXT:           double _grad0[1] = {};
//CHECK-NEXT:           sq_grad(clad::pop(_t2), _grad0);
//CHECK-NEXT:           double _r0 = _r_d0 * _grad0[0UL];
//CHECK-NEXT:           _result[clad::pop(_t1)] += _r0;
//CHECK-NEXT:           _d_s -= _r_d0;
//CHECK-NEXT:       }
//CHECK-NEXT:   }

// log-likelihood of n-dimensional gaussian distribution with covariance sigma^2*I
double f_log_gaus(double* x, double* p /*means*/, double n, double sigma) {
  double power = 0;
  for (int i = 0; i < n; i++)
    power += sq(x[i] - p[i]);
  power = -power/(2*sq(sigma));
  double gaus = 1./std::sqrt(std::pow(2*M_PI, n) * sigma) * std::exp(power);
  return std::log(gaus);
}

//CHECK:   void f_log_gaus_grad_1(double *x, double *p, double n, double sigma, double *_result) {
//CHECK-NEXT:       double _d_power = 0;
//CHECK-NEXT:       unsigned long _t0;
//CHECK-NEXT:       int _d_i = 0;
//CHECK-NEXT:       clad::tape<int> _t1 = {};
//CHECK-NEXT:       clad::tape<int> _t2 = {};
//CHECK-NEXT:       clad::tape<double> _t3 = {};
//CHECK-NEXT:       double _t4;
//CHECK-NEXT:       double _t5;
//CHECK-NEXT:       double _t6;
//CHECK-NEXT:       double _t7;
//CHECK-NEXT:       double _t8;
//CHECK-NEXT:       double _t9;
//CHECK-NEXT:       double _t10;
//CHECK-NEXT:       double _t11;
//CHECK-NEXT:       double _t12;
//CHECK-NEXT:       double _t13;
//CHECK-NEXT:       double _t14;
//CHECK-NEXT:       double _t15;
//CHECK-NEXT:       double _t16;
//CHECK-NEXT:       double _d_gaus = 0;
//CHECK-NEXT:       double _t17;
//CHECK-NEXT:       double power = 0;
//CHECK-NEXT:       _t0 = 0;
//CHECK-NEXT:       for (int i = 0; i < n; i++) {
//CHECK-NEXT:           _t0++;
//CHECK-NEXT:           power += sq(clad::push(_t3, x[clad::push(_t1, i)] - p[clad::push(_t2, i)]));
//CHECK-NEXT:       }
//CHECK-NEXT:       _t5 = -power;
//CHECK-NEXT:       _t7 = sigma;
//CHECK-NEXT:       _t6 = sq(_t7);
//CHECK-NEXT:       _t4 = (2 * _t6);
//CHECK-NEXT:       power = _t5 / _t4;
//CHECK-NEXT:       _t11 = 2 * 3.1415926535897931;
//CHECK-NEXT:       _t12 = n;
//CHECK-NEXT:       _t13 = std::pow(_t11, _t12);
//CHECK-NEXT:       _t10 = sigma;
//CHECK-NEXT:       _t14 = _t13 * _t10;
//CHECK-NEXT:       _t9 = std::sqrt(_t14);
//CHECK-NEXT:       _t15 = 1. / _t9;
//CHECK-NEXT:       _t16 = power;
//CHECK-NEXT:       _t8 = std::exp(_t16);
//CHECK-NEXT:       double gaus = _t15 * _t8;
//CHECK-NEXT:       _t17 = gaus;
//CHECK-NEXT:       double f_log_gaus_return = std::log(_t17);
//CHECK-NEXT:       goto _label0;
//CHECK-NEXT:     _label0:
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r17 = 1 * custom_derivatives::log_darg0(_t17);
//CHECK-NEXT:           _d_gaus += _r17;
//CHECK-NEXT:       }
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r6 = _d_gaus * _t8;
//CHECK-NEXT:           double _r7 = _r6 / _t9;
//CHECK-NEXT:           double _r8 = _r6 * -1. / (_t9 * _t9);
//CHECK-NEXT:           double _r9 = _r8 * custom_derivatives::sqrt_darg0(_t14);
//CHECK-NEXT:           double _r10 = _r9 * _t10;
//CHECK-NEXT:           double _grad2[2] = {};
//CHECK-NEXT:           custom_derivatives::pow_grad(_t11, _t12, _grad2);
//CHECK-NEXT:           double _r11 = _r10 * _grad2[0UL];
//CHECK-NEXT:           double _r12 = _r11 * 3.1415926535897931;
//CHECK-NEXT:           double _r13 = _r10 * _grad2[1UL];
//CHECK-NEXT:           double _r14 = _t13 * _r9;
//CHECK-NEXT:           double _r15 = _t15 * _d_gaus;
//CHECK-NEXT:           double _r16 = _r15 * custom_derivatives::exp_darg0(_t16);
//CHECK-NEXT:           _d_power += _r16;
//CHECK-NEXT:       }
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r_d1 = _d_power;
//CHECK-NEXT:           double _r1 = _r_d1 / _t4;
//CHECK-NEXT:           _d_power += -_r1;
//CHECK-NEXT:           double _r2 = _r_d1 * -_t5 / (_t4 * _t4);
//CHECK-NEXT:           double _r3 = _r2 * _t6;
//CHECK-NEXT:           double _r4 = 2 * _r2;
//CHECK-NEXT:           double _grad1[1] = {};
//CHECK-NEXT:           sq_grad(_t7, _grad1);
//CHECK-NEXT:           double _r5 = _r4 * _grad1[0UL];
//CHECK-NEXT:           _d_power -= _r_d1;
//CHECK-NEXT:       }
//CHECK-NEXT:       for (; _t0; _t0--) {
//CHECK-NEXT:           double _r_d0 = _d_power;
//CHECK-NEXT:           _d_power += _r_d0;
//CHECK-NEXT:           double _grad0[1] = {};
//CHECK-NEXT:           sq_grad(clad::pop(_t3), _grad0);
//CHECK-NEXT:           double _r0 = _r_d0 * _grad0[0UL];
//CHECK-NEXT:           _result[clad::pop(_t2)] += -_r0;
//CHECK-NEXT:           _d_power -= _r_d0;
//CHECK-NEXT:       }
//CHECK-NEXT:   }

#define TEST(F, x) { \
  result[0] = 0; \
  auto F##grad = clad::gradient(F);\
  F##grad.execute(x, result);\
  printf("{%.2f}\n", result[0]); \
}

int main() {
  double result[5] = {};
  TEST(f1, 3); // CHECK-EXEC: {27.00}
  TEST(f2, 3); // CHECK-EXEC: {59049.00} 
  TEST(f3, 3); // CHECK-EXEC: {6.00} 
  TEST(f4, 3); // CHECK-EXEC: {27.00}

  double p[] = { 1, 2, 3, 4, 5 };

  for (int i = 0; i < 5; i++) result[i] = 0;
  auto f_sum_grad = clad::gradient(f_sum, "p");
  f_sum_grad.execute(p, 5, result);
  printf("{%.2f, %.2f, %.2f, %.2f, %.2f}\n", result[0], result[1], result[2], result[3], result[4]); // CHECK-EXEC: {1.00, 1.00, 1.00, 1.00, 1.00}

  for (int i = 0; i < 5; i++) result[i] = 0;
  auto f_sum_squares_grad = clad::gradient(f_sum_squares, "p");
  f_sum_squares_grad.execute(p, 5, result);
  printf("{%.2f, %.2f, %.2f, %.2f, %.2f}\n", result[0], result[1], result[2], result[3], result[4]); // CHECK-EXEC: {2.00, 4.00, 6.00, 8.00, 10.00}

  for (int i = 0; i < 5; i++) result[i] = 0;
  auto f_log_gaus_d_means = clad::gradient(f_log_gaus, "p"); // == { (x[i] - p[i])/sigma^2 }
  double x[] = { 1, 1, 1, 1, 1 };
  f_log_gaus_d_means.execute(x, p, 5, 2.0, result);
  printf("{%.2f, %.2f, %.2f, %.2f, %.2f}\n", result[0], result[1], result[2], result[3], result[4]); // CHECK-EXEC: {0.00, -0.25, -0.50, -0.75, -1.00}
}
