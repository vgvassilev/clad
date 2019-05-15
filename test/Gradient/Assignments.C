// RUN: %cladclang %s -I%S/../../include -oReverseAssignments.out 2>&1 | FileCheck %s
// RUN: ./ReverseAssignments.out | FileCheck -check-prefix=CHECK-EXEC %s
//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"
#include <cmath>

// = y
double f1(double x, double y) {
  x = y;
  return y;
}

//CHECK:   void f1_grad(double x, double y, double *_result) {
//CHECK-NEXT:       x = y;
//CHECK-NEXT:       double f1_return = y;
//CHECK-NEXT:       goto _label0;
//CHECK-NEXT:     _label0:
//CHECK-NEXT:       _result[1UL] += 1;
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r_d0 = _result[0UL];
//CHECK-NEXT:          _result[1UL] += _r_d0;
//CHECK-NEXT:          _result[0UL] -= _r_d0;
//CHECK-NEXT:       }
//CHECK-NEXT:   }

// = max(x, y)
double f2(double x, double y) {
  if (x < y)
    x = y;
  return x;
}

//CHECK:   void f2_grad(double x, double y, double *_result) {
//CHECK-NEXT:       bool _cond0;
//CHECK-NEXT:       _cond0 = x < y;
//CHECK-NEXT:       if (_cond0)
//CHECK-NEXT:           x = y;
//CHECK-NEXT:       double f2_return = x;
//CHECK-NEXT:       goto _label0;
//CHECK-NEXT:     _label0:
//CHECK-NEXT:       _result[0UL] += 1;
//CHECK-NEXT:       if (_cond0) {
//CHECK-NEXT:          double _r_d0 = _result[0UL];
//CHECK-NEXT:         _result[1UL] += _r_d0;
//CHECK-NEXT:         _result[0UL] -= _r_d0;
//CHECK-NEXT:       }
//CHECK-NEXT:   }

// = x^4
double f3(double x, double y) {
  x = x;
  x = x * x;
  y = x * x;
  x = y;
  return y;
}

//CHECK:   void f3_grad(double x, double y, double *_result) {
//CHECK-NEXT:       double _t0;
//CHECK-NEXT:       double _t1;
//CHECK-NEXT:       double _t2;
//CHECK-NEXT:       double _t3;
//CHECK-NEXT:       x = x;
//CHECK-NEXT:       _t1 = x;
//CHECK-NEXT:       _t0 = x;
//CHECK-NEXT:       x = _t1 * _t0;
//CHECK-NEXT:       _t3 = x;
//CHECK-NEXT:       _t2 = x;
//CHECK-NEXT:       y = _t3 * _t2;
//CHECK-NEXT:       x = y;
//CHECK-NEXT:       double f3_return = y;
//CHECK-NEXT:       goto _label0;
//CHECK-NEXT:     _label0:
//CHECK-NEXT:       _result[1UL] += 1;
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r_d3 = _result[0UL];
//CHECK-NEXT:           _result[1UL] += _r_d3;
//CHECK-NEXT:           _result[0UL] -= _r_d3;
//CHECK-NEXT:       }
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r_d2 = _result[1UL];
//CHECK-NEXT:           double _r2 = _r_d2 * _t2;
//CHECK-NEXT:           _result[0UL] += _r2;
//CHECK-NEXT:           double _r3 = _t3 * _r_d2;
//CHECK-NEXT:           _result[0UL] += _r3;
//CHECK-NEXT:           _result[1UL] -= _r_d2;
//CHECK-NEXT:       }
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r_d1 = _result[0UL];
//CHECK-NEXT:           double _r0 = _r_d1 * _t0;
//CHECK-NEXT:           _result[0UL] += _r0;
//CHECK-NEXT:           double _r1 = _t1 * _r_d1;
//CHECK-NEXT:           _result[0UL] += _r1;
//CHECK-NEXT:           _result[0UL] -= _r_d1;
//CHECK-NEXT:       }
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r_d0 = _result[0UL];
//CHECK-NEXT:           _result[0UL] += _r_d0;
//CHECK-NEXT:           _result[0UL] -= _r_d0;
//CHECK-NEXT:       }
//CHECK-NEXT:   }

// = x
double f4(double x, double y) {
   y = x;
   x = 0;
   return y;
}

//CHECK:   void f4_grad(double x, double y, double *_result) {
//CHECK-NEXT:       y = x;
//CHECK-NEXT:       x = 0;
//CHECK-NEXT:       double f4_return = y;
//CHECK-NEXT:       goto _label0;
//CHECK-NEXT:     _label0:
//CHECK-NEXT:       _result[1UL] += 1;
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r_d1 = _result[0UL];
//CHECK-NEXT:           _result[0UL] -= _r_d1;
//CHECK-NEXT:       }
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r_d0 = _result[1UL];
//CHECK-NEXT:           _result[0UL] += _r_d0;
//CHECK-NEXT:           _result[1UL] -= _r_d0;
//CHECK-NEXT:       }
//CHECK-NEXT:   }

// = (x < 0 ? (-x * x) : ((y < 0) ? (-x * x) : (x * x)))
double f5(double x, double y) {
  double t = x * x;
  if (x < 0) {
    t = -t;
    return t;
  }
  if (y < 0) {
    double z = t;
    t = -t;
  }
  return t;
}

//CHECK:   void f5_grad(double x, double y, double *_result) {
//CHECK-NEXT:       double _t0;
//CHECK-NEXT:       double _t1;
//CHECK-NEXT:       double _d_t = 0;
//CHECK-NEXT:       bool _cond0;
//CHECK-NEXT:       bool _cond1;
//CHECK-NEXT:       double _d_z = 0;
//CHECK-NEXT:       _t1 = x;
//CHECK-NEXT:       _t0 = x;
//CHECK-NEXT:       double t = _t1 * _t0;
//CHECK-NEXT:       _cond0 = x < 0;
//CHECK-NEXT:       if (_cond0) {
//CHECK-NEXT:           t = -t;
//CHECK-NEXT:           double f5_return = t;
//CHECK-NEXT:           goto _label0;
//CHECK-NEXT:       }
//CHECK-NEXT:       _cond1 = y < 0;
//CHECK-NEXT:       if (_cond1) {
//CHECK-NEXT:           double z = t;
//CHECK-NEXT:           t = -t;
//CHECK-NEXT:       }
//CHECK-NEXT:       double f5_return = t;
//CHECK-NEXT:       goto _label1;
//CHECK-NEXT:     _label1:
//CHECK-NEXT:       _d_t += 1;
//CHECK-NEXT:       if (_cond1) {
//CHECK-NEXT:           {
//CHECK-NEXT:               double _r_d1 = _d_t;
//CHECK-NEXT:               _d_t += -_r_d1;
//CHECK-NEXT:               _d_t -= _r_d1;
//CHECK-NEXT:           }
//CHECK-NEXT:           _d_t += _d_z;
//CHECK-NEXT:       }
//CHECK-NEXT:       if (_cond0) {
//CHECK-NEXT:         _label0:
//CHECK-NEXT:           _d_t += 1;
//CHECK-NEXT:           {
//CHECK-NEXT:               double _r_d0 = _d_t;
//CHECK-NEXT:               _d_t += -_r_d0;
//CHECK-NEXT:               _d_t -= _r_d0;
//CHECK-NEXT:           }
//CHECK-NEXT:       }
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r0 = _d_t * _t0;
//CHECK-NEXT:           _result[0UL] += _r0;
//CHECK-NEXT:           double _r1 = _t1 * _d_t;
//CHECK-NEXT:           _result[0UL] += _r1;
//CHECK-NEXT:       }
//CHECK-NEXT:   }

// = sign(x) * sign(y) * x * x
double f6(double x, double y) {
  double t = x * x;
  if (x < 0) {
    t = -t;
    return t;
  }
  if (y < 0) {
    double z = t;
    t = -t;
  }
  return t;
}

//CHECK:   void f6_grad(double x, double y, double *_result) {
//CHECK-NEXT:       double _t0;
//CHECK-NEXT:       double _t1;
//CHECK-NEXT:       double _d_t = 0;
//CHECK-NEXT:       bool _cond0;
//CHECK-NEXT:       bool _cond1;
//CHECK-NEXT:       double _d_z = 0;
//CHECK-NEXT:       _t1 = x;
//CHECK-NEXT:       _t0 = x;
//CHECK-NEXT:       double t = _t1 * _t0;
//CHECK-NEXT:       _cond0 = x < 0;
//CHECK-NEXT:       if (_cond0) {
//CHECK-NEXT:           t = -t;
//CHECK-NEXT:           double f6_return = t;
//CHECK-NEXT:           goto _label0;
//CHECK-NEXT:       }
//CHECK-NEXT:       _cond1 = y < 0;
//CHECK-NEXT:       if (_cond1) {
//CHECK-NEXT:           double z = t;
//CHECK-NEXT:           t = -t;
//CHECK-NEXT:       }
//CHECK-NEXT:       double f6_return = t;
//CHECK-NEXT:       goto _label1;
//CHECK-NEXT:     _label1:
//CHECK-NEXT:       _d_t += 1;
//CHECK-NEXT:       if (_cond1) {
//CHECK-NEXT:           {
//CHECK-NEXT:               double _r_d1 = _d_t;
//CHECK-NEXT:               _d_t += -_r_d1;
//CHECK-NEXT:               _d_t -= _r_d1;
//CHECK-NEXT:           }
//CHECK-NEXT:           _d_t += _d_z;
//CHECK-NEXT:       }
//CHECK-NEXT:       if (_cond0) {
//CHECK-NEXT:         _label0:
//CHECK-NEXT:           _d_t += 1;
//CHECK-NEXT:           {
//CHECK-NEXT:               double _r_d0 = _d_t;
//CHECK-NEXT:               _d_t += -_r_d0;
//CHECK-NEXT:               _d_t -= _r_d0;
//CHECK-NEXT:           }
//CHECK-NEXT:       }
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r0 = _d_t * _t0;
//CHECK-NEXT:           _result[0UL] += _r0;
//CHECK-NEXT:           double _r1 = _t1 * _d_t;
//CHECK-NEXT:           _result[0UL] += _r1;
//CHECK-NEXT:       }
//CHECK-NEXT:   }

double f7(double x, double y) {
  double t[] = {1, x, x * x};
  t[0]++;
  t[0]--;
  ++t[0];
  --t[0];
  t[0] = x;
  x = y;
  t[0] += t[1];
  t[0] *= t[1];
  t[0] /= t[1];
  t[0] -= t[1];
  x = ++t[0];
  return t[0]; // == x
}

//CHECK:   void f7_grad(double x, double y, double *_result) {
//CHECK-NEXT:       double _t0;
//CHECK-NEXT:       double _t1;
//CHECK-NEXT:       double _d_t[3] = {};
//CHECK-NEXT:       double _t2;
//CHECK-NEXT:       double _t3;
//CHECK-NEXT:       double _t4;
//CHECK-NEXT:       double _t5;
//CHECK-NEXT:       _t1 = x;
//CHECK-NEXT:       _t0 = x;
//CHECK-NEXT:       double t[3] = {1, x, _t1 * _t0};
//CHECK-NEXT:       t[0]++;
//CHECK-NEXT:       t[0]--;
//CHECK-NEXT:       ++t[0];
//CHECK-NEXT:       --t[0];
//CHECK-NEXT:       t[0] = x;
//CHECK-NEXT:       x = y;
//CHECK-NEXT:       t[0] += t[1];
//CHECK-NEXT:       _t3 = t[0];
//CHECK-NEXT:       _t2 = t[1];
//CHECK-NEXT:       t[0] *= _t2;
//CHECK-NEXT:       _t5 = t[0];
//CHECK-NEXT:       _t4 = t[1];
//CHECK-NEXT:       t[0] /= _t4;
//CHECK-NEXT:       t[0] -= t[1];
//CHECK-NEXT:       x = ++t[0];
//CHECK-NEXT:       double f7_return = t[0];
//CHECK-NEXT:       goto _label0;
//CHECK-NEXT:     _label0:
//CHECK-NEXT:       _d_t[0] += 1;
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r_d6 = _result[0UL];
//CHECK-NEXT:           _d_t[0] += _r_d6;
//CHECK-NEXT:           _result[0UL] -= _r_d6;
//CHECK-NEXT:       }
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r_d5 = _d_t[0];
//CHECK-NEXT:           _d_t[0] += _r_d5;
//CHECK-NEXT:           _d_t[1] += -_r_d5;
//CHECK-NEXT:           _d_t[0] -= _r_d5;
//CHECK-NEXT:       }
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r_d4 = _d_t[0];
//CHECK-NEXT:           _d_t[0] += _r_d4 / _t4;
//CHECK-NEXT:           double _r3 = _r_d4 * -_t5 / (_t4 * _t4);
//CHECK-NEXT:           _d_t[1] += _r3;
//CHECK-NEXT:           _d_t[0] -= _r_d4;
//CHECK-NEXT:       }
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r_d3 = _d_t[0];
//CHECK-NEXT:           _d_t[0] += _r_d3 * _t2;
//CHECK-NEXT:           double _r2 = _t3 * _r_d3;
//CHECK-NEXT:           _d_t[1] += _r2;
//CHECK-NEXT:           _d_t[0] -= _r_d3;
//CHECK-NEXT:       }
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r_d2 = _d_t[0];
//CHECK-NEXT:           _d_t[0] += _r_d2;
//CHECK-NEXT:           _d_t[1] += _r_d2;
//CHECK-NEXT:           _d_t[0] -= _r_d2;
//CHECK-NEXT:       }
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r_d1 = _result[0UL];
//CHECK-NEXT:           _result[1UL] += _r_d1;
//CHECK-NEXT:           _result[0UL] -= _r_d1;
//CHECK-NEXT:       }
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r_d0 = _d_t[0];
//CHECK-NEXT:           _result[0UL] += _r_d0;
//CHECK-NEXT:           _d_t[0] -= _r_d0;
//CHECK-NEXT:       }
//CHECK-NEXT:       {
//CHECK-NEXT:           _result[0UL] += _d_t[1];
//CHECK-NEXT:           double _r0 = _d_t[2] * _t0;
//CHECK-NEXT:           _result[0UL] += _r0;
//CHECK-NEXT:           double _r1 = _t1 * _d_t[2];
//CHECK-NEXT:           _result[0UL] += _r1;
//CHECK-NEXT:       }
//CHECK-NEXT:   }

double f8(double x, double y) {
  double t[] = {1, x, y, 1};
  t[3] = (y *= (t[0] = t[1] = t[2]));
  return t[3]; // == y * y
}

//CHECK:   void f8_grad(double x, double y, double *_result) {
//CHECK-NEXT:       double _d_t[4] = {};
//CHECK-NEXT:       double _t0;
//CHECK-NEXT:       double _t1;
//CHECK-NEXT:       double t[4] = {1, x, y, 1};
//CHECK-NEXT:       _t1 = y;
//CHECK-NEXT:       _t0 = (t[0] = t[1] = t[2]);
//CHECK-NEXT:       t[3] = (y *= _t0);
//CHECK-NEXT:       double f8_return = t[3];
//CHECK-NEXT:       goto _label0;
//CHECK-NEXT:     _label0:
//CHECK-NEXT:       _d_t[3] += 1;
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r_d0 = _d_t[3];
//CHECK-NEXT:           _result[1UL] += _r_d0;
//CHECK-NEXT:           double _r_d1 = _result[1UL];
//CHECK-NEXT:           _result[1UL] += _r_d1 * _t0;
//CHECK-NEXT:           double _r0 = _t1 * _r_d1;
//CHECK-NEXT:           _d_t[0] += _r0;
//CHECK-NEXT:           double _r_d2 = _d_t[0];
//CHECK-NEXT:           _d_t[1] += _r_d2;
//CHECK-NEXT:           double _r_d3 = _d_t[1];
//CHECK-NEXT:           _d_t[2] += _r_d3;
//CHECK-NEXT:           _d_t[1] -= _r_d3;
//CHECK-NEXT:           _d_t[0] -= _r_d2;
//CHECK-NEXT:           _result[1UL] -= _r_d1;
//CHECK-NEXT:           _d_t[3] -= _r_d0;
//CHECK-NEXT:       }
//CHECK-NEXT:       {
//CHECK-NEXT:           _result[0UL] += _d_t[1];
//CHECK-NEXT:           _result[1UL] += _d_t[2];
//CHECK-NEXT:       }
//CHECK-NEXT:   }

double f9(double x, double y) {
  double t = x;
  (t *= x) *= y;
  return t; // x * x * y
}

//CHECK:   void f9_grad(double x, double y, double *_result) {
//CHECK-NEXT:       double _d_t = 0;
//CHECK-NEXT:       double _t0;
//CHECK-NEXT:       double _t1;
//CHECK-NEXT:       double _t2;
//CHECK-NEXT:       double _t3;
//CHECK-NEXT:       double t = x;
//CHECK-NEXT:       _t1 = t;
//CHECK-NEXT:       _t0 = x;
//CHECK-NEXT:       double &_ref0 = (t *= _t0);
//CHECK-NEXT:       _t3 = _ref0;
//CHECK-NEXT:       _t2 = y;
//CHECK-NEXT:       _ref0 *= _t2;
//CHECK-NEXT:       double f9_return = t;
//CHECK-NEXT:       goto _label0;
//CHECK-NEXT:     _label0:
//CHECK-NEXT:       _d_t += 1;
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r_d1 = _d_t;
//CHECK-NEXT:           _d_t += _r_d1 * _t2;
//CHECK-NEXT:           double _r1 = _t3 * _r_d1;
//CHECK-NEXT:           _result[1UL] += _r1;
//CHECK-NEXT:           _d_t -= _r_d1;
//CHECK-NEXT:           double _r_d0 = _d_t;
//CHECK-NEXT:           _d_t += _r_d0 * _t0;
//CHECK-NEXT:           double _r0 = _t1 * _r_d0;
//CHECK-NEXT:           _result[0UL] += _r0;
//CHECK-NEXT:           _d_t -= _r_d0;
//CHECK-NEXT:       }
//CHECK-NEXT:       _result[0UL] += _d_t;
//CHECK-NEXT:   }

double f10(double x, double y) {
  double t = x;
  t = x = y;
  return t; // = y
}

//CHECK:   void f10_grad(double x, double y, double *_result) {
//CHECK-NEXT:       double _d_t = 0;
//CHECK-NEXT:       double t = x;
//CHECK-NEXT:       t = x = y;
//CHECK-NEXT:       double f10_return = t;
//CHECK-NEXT:       goto _label0;
//CHECK-NEXT:     _label0:
//CHECK-NEXT:       _d_t += 1;
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r_d0 = _d_t;
//CHECK-NEXT:           _result[0UL] += _r_d0;
//CHECK-NEXT:           double _r_d1 = _result[0UL];
//CHECK-NEXT:           _result[1UL] += _r_d1;
//CHECK-NEXT:           _result[0UL] -= _r_d1;
//CHECK-NEXT:           _d_t -= _r_d0;
//CHECK-NEXT:       }
//CHECK-NEXT:       _result[0UL] += _d_t;
//CHECK-NEXT:   }

double f11(double x, double y) {
  double t = x;
  (t = x) = y;
  return t; // = y
}

//CHECK:   void f11_grad(double x, double y, double *_result) {
//CHECK-NEXT:       double _d_t = 0;
//CHECK-NEXT:       double t = x;
//CHECK-NEXT:       (t = x) = y;
//CHECK-NEXT:       double f11_return = t;
//CHECK-NEXT:       goto _label0;
//CHECK-NEXT:     _label0:
//CHECK-NEXT:       _d_t += 1;
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r_d1 = _d_t;
//CHECK-NEXT:           _result[1UL] += _r_d1;
//CHECK-NEXT:           _d_t -= _r_d1;
//CHECK-NEXT:           double _r_d0 = _d_t;
//CHECK-NEXT:           _result[0UL] += _r_d0;
//CHECK-NEXT:           _d_t -= _r_d0;
//CHECK-NEXT:       }
//CHECK-NEXT:       _result[0UL] += _d_t;
//CHECK-NEXT:   }

double f12(double x, double y) {
  double t;
  (x > y ? (t = x) : (t = y)) *= y;
  return t; // == max(x, y) * y;
}

//CHECK:   void f12_grad(double x, double y, double *_result) {
//CHECK-NEXT:       double _d_t = 0;
//CHECK-NEXT:       bool _cond0;
//CHECK-NEXT:       double _t0;
//CHECK-NEXT:       double _t1;
//CHECK-NEXT:       double t;
//CHECK-NEXT:       _cond0 = x > y;
//CHECK-NEXT:       double &_ref0 = (_cond0 ? (t = x) : (t = y));
//CHECK-NEXT:       _t1 = _ref0;
//CHECK-NEXT:       _t0 = y;
//CHECK-NEXT:       _ref0 *= _t0;
//CHECK-NEXT:       double f12_return = t;
//CHECK-NEXT:       goto _label0;
//CHECK-NEXT:     _label0:
//CHECK-NEXT:       _d_t += 1;
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r_d2 = (_cond0 ? _d_t : _d_t);
//CHECK-NEXT:           (_cond0 ? _d_t : _d_t) += _r_d2 * _t0;
//CHECK-NEXT:           double _r0 = _t1 * _r_d2;
//CHECK-NEXT:           _result[1UL] += _r0;
//CHECK-NEXT:           (_cond0 ? _d_t : _d_t) -= _r_d2;
//CHECK-NEXT:           if (_cond0) {
//CHECK-NEXT:               double _r_d0 = _d_t;
//CHECK-NEXT:               _result[0UL] += _r_d0;
//CHECK-NEXT:               _d_t -= _r_d0;
//CHECK-NEXT:           } else {
//CHECK-NEXT:               double _r_d1 = _d_t;
//CHECK-NEXT:               _result[1UL] += _r_d1;
//CHECK-NEXT:               _d_t -= _r_d1;
//CHECK-NEXT:           }
//CHECK-NEXT:       }
//CHECK-NEXT:   }

double f13(double x, double y) {
  double t = x * (y = x);
  return t * y; // == x * x * x
}

//CHECK:   void f13_grad(double x, double y, double *_result) {
//CHECK-NEXT:       double _t0;
//CHECK-NEXT:       double _t1;
//CHECK-NEXT:       double _d_t = 0;
//CHECK-NEXT:       double _t2;
//CHECK-NEXT:       double _t3;
//CHECK-NEXT:       _t1 = x;
//CHECK-NEXT:       _t0 = (y = x);
//CHECK-NEXT:       double t = _t1 * _t0;
//CHECK-NEXT:       _t3 = t;
//CHECK-NEXT:       _t2 = y;
//CHECK-NEXT:       double f13_return = _t3 * _t2;
//CHECK-NEXT:       goto _label0;
//CHECK-NEXT:     _label0:
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r2 = 1 * _t2;
//CHECK-NEXT:           _d_t += _r2;
//CHECK-NEXT:           double _r3 = _t3 * 1;
//CHECK-NEXT:           _result[1UL] += _r3;
//CHECK-NEXT:       }
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r0 = _d_t * _t0;
//CHECK-NEXT:           _result[0UL] += _r0;
//CHECK-NEXT:           double _r1 = _t1 * _d_t;
//CHECK-NEXT:           _result[1UL] += _r1;
//CHECK-NEXT:           double _r_d0 = _result[1UL];
//CHECK-NEXT:           _result[0UL] += _r_d0;
//CHECK-NEXT:           _result[1UL] -= _r_d0;
//CHECK-NEXT:       }
//CHECK-NEXT:   }

#define TEST(F, x, y) { \
  result[0] = 0; result[1] = 0;\
  auto F##grad = clad::gradient(F);\
  F##grad.execute(x, y, result);\
  printf("{%.2f, %.2f}\n", result[0], result[1]); \
}

int main() {
  double result[2] = {};
  TEST(f1, 3, 4); // CHECK-EXEC: {0.00, 1.00}
  TEST(f2, 3, 4); // CHECK-EXEC: {0.00, 1.00} 
  TEST(f3, 2, 100); // CHECK-EXEC: {32.00, 0.00} 
  TEST(f4, 3, 4); // CHECK-EXEC: {1.00, 0.00} 
  TEST(f5, -3, 4); // CHECK-EXEC: {6.00, 0.00} 
  TEST(f6, 3, -4); // CHECK-EXEC: {-6.00, 0.00} 
  TEST(f7, 3, 4); // CHECK-EXEC: {1.00, 0.00}
  TEST(f8, 3, 4); // CHECK-EXEC: {0.00, 8.00}
  TEST(f9, 3, 4); // CHECK-EXEC: {24.00, 9.00}
  TEST(f10, 3, 4); // CHECK-EXEC: {0.00, 1.00}
  TEST(f11, 3, 4); // CHECK-EXEC: {0.00, 1.00}
  TEST(f12, 3, 4); // CHECK-EXEC: {0.00, 8.00}
  TEST(f13, 3, 4); // CHECK-EXEC: {27.00, 0.00}
}
