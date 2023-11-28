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

//CHECK:   void f1_grad(double x, double y, clad::array_ref<double> _d_x, clad::array_ref<double> _d_y) {
//CHECK-NEXT:       x = y;
//CHECK-NEXT:       * _d_y += 1;
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r_d0 = * _d_x;
//CHECK-NEXT:           * _d_y += _r_d0;
//CHECK-NEXT:           * _d_x -= _r_d0;
//CHECK-NEXT:           * _d_x;
//CHECK-NEXT:       }
//CHECK-NEXT:   }

// = max(x, y)
double f2(double x, double y) {
  if (x < y)
    x = y;
  return x;
}

//CHECK:   void f2_grad(double x, double y, clad::array_ref<double> _d_x, clad::array_ref<double> _d_y) {
//CHECK-NEXT:       bool _cond0;
//CHECK-NEXT:       _cond0 = x < y;
//CHECK-NEXT:       if (_cond0)
//CHECK-NEXT:           x = y;
//CHECK-NEXT:       * _d_x += 1;
//CHECK-NEXT:       if (_cond0) {
//CHECK-NEXT:           double _r_d0 = * _d_x;
//CHECK-NEXT:           * _d_y += _r_d0;
//CHECK-NEXT:           * _d_x -= _r_d0;
//CHECK-NEXT:           * _d_x;
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

//CHECK:   void f3_grad(double x, double y, clad::array_ref<double> _d_x, clad::array_ref<double> _d_y) {
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
//CHECK-NEXT:       * _d_y += 1;
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r_d3 = * _d_x;
//CHECK-NEXT:           * _d_y += _r_d3;
//CHECK-NEXT:           * _d_x -= _r_d3;
//CHECK-NEXT:           * _d_x;
//CHECK-NEXT:       }
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r_d2 = * _d_y;
//CHECK-NEXT:           double _r2 = _r_d2 * _t2;
//CHECK-NEXT:           * _d_x += _r2;
//CHECK-NEXT:           double _r3 = _t3 * _r_d2;
//CHECK-NEXT:           * _d_x += _r3;
//CHECK-NEXT:           * _d_y -= _r_d2;
//CHECK-NEXT:           * _d_y;
//CHECK-NEXT:       }
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r_d1 = * _d_x;
//CHECK-NEXT:           double _r0 = _r_d1 * _t0;
//CHECK-NEXT:           * _d_x += _r0;
//CHECK-NEXT:           double _r1 = _t1 * _r_d1;
//CHECK-NEXT:           * _d_x += _r1;
//CHECK-NEXT:           * _d_x -= _r_d1;
//CHECK-NEXT:           * _d_x;
//CHECK-NEXT:       }
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r_d0 = * _d_x;
//CHECK-NEXT:           * _d_x += _r_d0;
//CHECK-NEXT:           * _d_x -= _r_d0;
//CHECK-NEXT:           * _d_x;
//CHECK-NEXT:       }
//CHECK-NEXT:   }

// = x
double f4(double x, double y) {
   y = x;
   x = 0;
   return y;
}

//CHECK:   void f4_grad(double x, double y, clad::array_ref<double> _d_x, clad::array_ref<double> _d_y) {
//CHECK-NEXT:       y = x;
//CHECK-NEXT:       x = 0;
//CHECK-NEXT:       * _d_y += 1;
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r_d1 = * _d_x;
//CHECK-NEXT:           * _d_x -= _r_d1;
//CHECK-NEXT:           * _d_x;
//CHECK-NEXT:       }
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r_d0 = * _d_y;
//CHECK-NEXT:           * _d_x += _r_d0;
//CHECK-NEXT:           * _d_y -= _r_d0;
//CHECK-NEXT:           * _d_y;
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

//CHECK:   void f5_grad(double x, double y, clad::array_ref<double> _d_x, clad::array_ref<double> _d_y) {
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
//CHECK-NEXT:           goto _label0;
//CHECK-NEXT:       }
//CHECK-NEXT:       _cond1 = y < 0;
//CHECK-NEXT:       if (_cond1) {
//CHECK-NEXT:           double z = t;
//CHECK-NEXT:           t = -t;
//CHECK-NEXT:       }
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
//CHECK-NEXT:           * _d_x += _r0;
//CHECK-NEXT:           double _r1 = _t1 * _d_t;
//CHECK-NEXT:           * _d_x += _r1;
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

//CHECK:   void f6_grad(double x, double y, clad::array_ref<double> _d_x, clad::array_ref<double> _d_y) {
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
//CHECK-NEXT:           goto _label0;
//CHECK-NEXT:       }
//CHECK-NEXT:       _cond1 = y < 0;
//CHECK-NEXT:       if (_cond1) {
//CHECK-NEXT:           double z = t;
//CHECK-NEXT:           t = -t;
//CHECK-NEXT:       }
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
//CHECK-NEXT:           * _d_x += _r0;
//CHECK-NEXT:           double _r1 = _t1 * _d_t;
//CHECK-NEXT:           * _d_x += _r1;
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

//CHECK:   void f7_grad(double x, double y, clad::array_ref<double> _d_x, clad::array_ref<double> _d_y) {
//CHECK-NEXT:       double _t0;
//CHECK-NEXT:       double _t1;
//CHECK-NEXT:       clad::array<double> _d_t(3UL);
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
//CHECK-NEXT:       _d_t[0] += 1;
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r_d6 = * _d_x;
//CHECK-NEXT:           _d_t[0] += _r_d6;
//CHECK-NEXT:           * _d_x -= _r_d6;
//CHECK-NEXT:           * _d_x;
//CHECK-NEXT:       }
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r_d5 = _d_t[0];
//CHECK-NEXT:           _d_t[0] += _r_d5;
//CHECK-NEXT:           _d_t[1] += -_r_d5;
//CHECK-NEXT:           _d_t[0] -= _r_d5;
//CHECK-NEXT:           _d_t[0];
//CHECK-NEXT:       }
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r_d4 = _d_t[0];
//CHECK-NEXT:           _d_t[0] += _r_d4 / _t4;
//CHECK-NEXT:           double _r3 = _r_d4 * -_t5 / (_t4 * _t4);
//CHECK-NEXT:           _d_t[1] += _r3;
//CHECK-NEXT:           _d_t[0] -= _r_d4;
//CHECK-NEXT:           _d_t[0];
//CHECK-NEXT:       }
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r_d3 = _d_t[0];
//CHECK-NEXT:           _d_t[0] += _r_d3 * _t2;
//CHECK-NEXT:           double _r2 = _t3 * _r_d3;
//CHECK-NEXT:           _d_t[1] += _r2;
//CHECK-NEXT:           _d_t[0] -= _r_d3;
//CHECK-NEXT:           _d_t[0];
//CHECK-NEXT:       }
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r_d2 = _d_t[0];
//CHECK-NEXT:           _d_t[0] += _r_d2;
//CHECK-NEXT:           _d_t[1] += _r_d2;
//CHECK-NEXT:           _d_t[0] -= _r_d2;
//CHECK-NEXT:           _d_t[0];
//CHECK-NEXT:       }
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r_d1 = * _d_x;
//CHECK-NEXT:           * _d_y += _r_d1;
//CHECK-NEXT:           * _d_x -= _r_d1;
//CHECK-NEXT:           * _d_x;
//CHECK-NEXT:       }
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r_d0 = _d_t[0];
//CHECK-NEXT:           * _d_x += _r_d0;
//CHECK-NEXT:           _d_t[0] -= _r_d0;
//CHECK-NEXT:           _d_t[0];
//CHECK-NEXT:       }
//CHECK-NEXT:       _d_t[0];
//CHECK-NEXT:       _d_t[0];
//CHECK-NEXT:       {
//CHECK-NEXT:           * _d_x += _d_t[1];
//CHECK-NEXT:           double _r0 = _d_t[2] * _t0;
//CHECK-NEXT:           * _d_x += _r0;
//CHECK-NEXT:           double _r1 = _t1 * _d_t[2];
//CHECK-NEXT:           * _d_x += _r1;
//CHECK-NEXT:       }
//CHECK-NEXT:   }

double f8(double x, double y) {
  double t[] = {1, x, y, 1};
  t[3] = (y *= (t[0] = t[1] = t[2]));
  return t[3]; // == y * y
}

//CHECK:   void f8_grad(double x, double y, clad::array_ref<double> _d_x, clad::array_ref<double> _d_y) {
//CHECK-NEXT:       clad::array<double> _d_t(4UL);
//CHECK-NEXT:       double _t0;
//CHECK-NEXT:       double _t1;
//CHECK-NEXT:       double t[4] = {1, x, y, 1};
//CHECK-NEXT:       _t1 = y;
//CHECK-NEXT:       _t0 = (t[0] = t[1] = t[2]);
//CHECK-NEXT:       t[3] = (y *= _t0);
//CHECK-NEXT:       _d_t[3] += 1;
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r_d0 = _d_t[3];
//CHECK-NEXT:           * _d_y += _r_d0;
//CHECK-NEXT:           double _r_d1 = * _d_y;
//CHECK-NEXT:           * _d_y += _r_d1 * _t0;
//CHECK-NEXT:           double _r0 = _t1 * _r_d1;
//CHECK-NEXT:           _d_t[0] += _r0;
//CHECK-NEXT:           double _r_d2 = _d_t[0];
//CHECK-NEXT:           _d_t[1] += _r_d2;
//CHECK-NEXT:           double _r_d3 = _d_t[1];
//CHECK-NEXT:           _d_t[2] += _r_d3;
//CHECK-NEXT:           _d_t[1] -= _r_d3;
//CHECK-NEXT:           _d_t[0] -= _r_d2;
//CHECK-NEXT:           * _d_y -= _r_d1;
//CHECK-NEXT:           _d_t[3] -= _r_d0;
//CHECK-NEXT:           _d_t[3];
//CHECK-NEXT:       }
//CHECK-NEXT:       {
//CHECK-NEXT:           * _d_x += _d_t[1];
//CHECK-NEXT:           * _d_y += _d_t[2];
//CHECK-NEXT:       }
//CHECK-NEXT:   }

double f9(double x, double y) {
  double t = x;
  (t *= x) *= y;
  return t; // x * x * y
}

//CHECK:   void f9_grad(double x, double y, clad::array_ref<double> _d_x, clad::array_ref<double> _d_y) {
//CHECK-NEXT:       double _d_t = 0;
//CHECK-NEXT:       double _t0;
//CHECK-NEXT:       double _t1;
//CHECK-NEXT:       double _t2;
//CHECK-NEXT:       double _t3;
//CHECK-NEXT:       double t = x;
//CHECK-NEXT:       _t1 = t;
//CHECK-NEXT:       _t0 = x;
//CHECK-NEXT:       double _ref0 = (t *= _t0);
//CHECK-NEXT:       _t3 = _ref0;
//CHECK-NEXT:       _t2 = y;
//CHECK-NEXT:       _ref0 *= _t2;
//CHECK-NEXT:       _d_t += 1;
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r_d1 = _d_t;
//CHECK-NEXT:           _d_t += _r_d1 * _t2;
//CHECK-NEXT:           double _r1 = _t3 * _r_d1;
//CHECK-NEXT:           * _d_y += _r1;
//CHECK-NEXT:           _d_t -= _r_d1;
//CHECK-NEXT:           double _r_d0 = _d_t;
//CHECK-NEXT:           _d_t += _r_d0 * _t0;
//CHECK-NEXT:           double _r0 = _t1 * _r_d0;
//CHECK-NEXT:           * _d_x += _r0;
//CHECK-NEXT:           _d_t -= _r_d0;
//CHECK-NEXT:       }
//CHECK-NEXT:       * _d_x += _d_t;
//CHECK-NEXT:   }

double f10(double x, double y) {
  double t = x;
  t = x = y;
  return t; // = y
}

//CHECK:   void f10_grad(double x, double y, clad::array_ref<double> _d_x, clad::array_ref<double> _d_y) {
//CHECK-NEXT:       double _d_t = 0;
//CHECK-NEXT:       double t = x;
//CHECK-NEXT:       t = x = y;
//CHECK-NEXT:       _d_t += 1;
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r_d0 = _d_t;
//CHECK-NEXT:           * _d_x += _r_d0;
//CHECK-NEXT:           double _r_d1 = * _d_x;
//CHECK-NEXT:           * _d_y += _r_d1;
//CHECK-NEXT:           * _d_x -= _r_d1;
//CHECK-NEXT:           _d_t -= _r_d0;
//CHECK-NEXT:       }
//CHECK-NEXT:       * _d_x += _d_t;
//CHECK-NEXT:   }

double f11(double x, double y) {
  double t = x;
  (t = x) = y;
  return t; // = y
}

//CHECK:   void f11_grad(double x, double y, clad::array_ref<double> _d_x, clad::array_ref<double> _d_y) {
//CHECK-NEXT:       double _d_t = 0;
//CHECK-NEXT:       double t = x;
//CHECK-NEXT:       (t = x) = y;
//CHECK-NEXT:       _d_t += 1;
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r_d1 = _d_t;
//CHECK-NEXT:           * _d_y += _r_d1;
//CHECK-NEXT:           _d_t -= _r_d1;
//CHECK-NEXT:           double _r_d0 = _d_t;
//CHECK-NEXT:           * _d_x += _r_d0;
//CHECK-NEXT:           _d_t -= _r_d0;
//CHECK-NEXT:       }
//CHECK-NEXT:       * _d_x += _d_t;
//CHECK-NEXT:   }

double f12(double x, double y) {
  double t;
  (x > y ? (t = x) : (t = y)) *= y;
  return t; // == max(x, y) * y;
}

//CHECK:   void f12_grad(double x, double y, clad::array_ref<double> _d_x, clad::array_ref<double> _d_y) {
//CHECK-NEXT:       double _d_t = 0;
//CHECK-NEXT:       bool _cond0;
//CHECK-NEXT:       double _t0;
//CHECK-NEXT:       double _t1;
//CHECK-NEXT:       double t;
//CHECK-NEXT:       _cond0 = x > y;
//CHECK-NEXT:       double _ref0 = (_cond0 ? (t = x) : (t = y));
//CHECK-NEXT:       _t1 = _ref0;
//CHECK-NEXT:       _t0 = y;
//CHECK-NEXT:       _ref0 *= _t0;
//CHECK-NEXT:       _d_t += 1;
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r_d2 = (_cond0 ? _d_t : _d_t);
//CHECK-NEXT:           (_cond0 ? _d_t : _d_t) += _r_d2 * _t0;
//CHECK-NEXT:           double _r0 = _t1 * _r_d2;
//CHECK-NEXT:           * _d_y += _r0;
//CHECK-NEXT:           (_cond0 ? _d_t : _d_t) -= _r_d2;
//CHECK-NEXT:           if (_cond0) {
//CHECK-NEXT:               double _r_d0 = _d_t;
//CHECK-NEXT:               * _d_x += _r_d0;
//CHECK-NEXT:               _d_t -= _r_d0;
//CHECK-NEXT:           } else {
//CHECK-NEXT:               double _r_d1 = _d_t;
//CHECK-NEXT:               * _d_y += _r_d1;
//CHECK-NEXT:               _d_t -= _r_d1;
//CHECK-NEXT:           }
//CHECK-NEXT:       }
//CHECK-NEXT:   }

double f13(double x, double y) {
  double t = x * (y = x);
  return t * y; // == x * x * x
}

//CHECK:   void f13_grad(double x, double y, clad::array_ref<double> _d_x, clad::array_ref<double> _d_y) {
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
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r2 = 1 * _t2;
//CHECK-NEXT:           _d_t += _r2;
//CHECK-NEXT:           double _r3 = _t3 * 1;
//CHECK-NEXT:           * _d_y += _r3;
//CHECK-NEXT:       }
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r0 = _d_t * _t0;
//CHECK-NEXT:           * _d_x += _r0;
//CHECK-NEXT:           double _r1 = _t1 * _d_t;
//CHECK-NEXT:           * _d_y += _r1;
//CHECK-NEXT:           double _r_d0 = * _d_y;
//CHECK-NEXT:           * _d_x += _r_d0;
//CHECK-NEXT:           * _d_y -= _r_d0;
//CHECK-NEXT:       }
//CHECK-NEXT:   }

double f14(double i, double j) {
  double& a = i;
  a = 2*i;
  a += i;
  a *= i;
  return i;
}

// CHECK: void f14_grad(double i, double j, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j) {
// CHECK-NEXT:     double *_d_a = 0;
// CHECK-NEXT:     double _t0;
// CHECK-NEXT:     double _t1;
// CHECK-NEXT:     double _t2;
// CHECK-NEXT:     _d_a = &* _d_i;
// CHECK-NEXT:     double &a = i;
// CHECK-NEXT:     _t0 = i;
// CHECK-NEXT:     a = 2 * _t0;
// CHECK-NEXT:     a += i;
// CHECK-NEXT:     _t2 = a;
// CHECK-NEXT:     _t1 = i;
// CHECK-NEXT:     a *= _t1;
// CHECK-NEXT:     * _d_i += 1;
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r_d2 = *_d_a;
// CHECK-NEXT:         *_d_a += _r_d2 * _t1;
// CHECK-NEXT:         double _r2 = _t2 * _r_d2;
// CHECK-NEXT:         * _d_i += _r2;
// CHECK-NEXT:         *_d_a -= _r_d2;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r_d1 = *_d_a;
// CHECK-NEXT:         *_d_a += _r_d1;
// CHECK-NEXT:         * _d_i += _r_d1;
// CHECK-NEXT:         *_d_a -= _r_d1;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r_d0 = *_d_a;
// CHECK-NEXT:         double _r0 = _r_d0 * _t0;
// CHECK-NEXT:         double _r1 = 2 * _r_d0;
// CHECK-NEXT:         * _d_i += _r1;
// CHECK-NEXT:         *_d_a -= _r_d0;
// CHECK-NEXT:     }
// CHECK-NEXT: }

double f15(double i, double j) {
  double b = i*j;
  double& a = b;
  double& c = i;
  double& d = j;
  a *= i;
  b += 2*i;
  c += 3*i;
  d *= 3*j;
  return a+c+d;
}

// CHECK: void f15_grad(double i, double j, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j) {
// CHECK-NEXT:     double _t0;
// CHECK-NEXT:     double _t1;
// CHECK-NEXT:     double _d_b = 0;
// CHECK-NEXT:     double *_d_a = 0;
// CHECK-NEXT:     double *_d_c = 0;
// CHECK-NEXT:     double *_d_d = 0;
// CHECK-NEXT:     double _t2;
// CHECK-NEXT:     double _t3;
// CHECK-NEXT:     double _t4;
// CHECK-NEXT:     double _t5;
// CHECK-NEXT:     double _t6;
// CHECK-NEXT:     double _t7;
// CHECK-NEXT:     double _t8;
// CHECK-NEXT:     _t1 = i;
// CHECK-NEXT:     _t0 = j;
// CHECK-NEXT:     double b = _t1 * _t0;
// CHECK-NEXT:     _d_a = &_d_b;
// CHECK-NEXT:     double &a = b;
// CHECK-NEXT:     _d_c = &* _d_i;
// CHECK-NEXT:     double &c = i;
// CHECK-NEXT:     _d_d = &* _d_j;
// CHECK-NEXT:     double &d = j;
// CHECK-NEXT:     _t3 = a;
// CHECK-NEXT:     _t2 = i;
// CHECK-NEXT:     a *= _t2;
// CHECK-NEXT:     _t4 = i;
// CHECK-NEXT:     b += 2 * _t4;
// CHECK-NEXT:     _t5 = i;
// CHECK-NEXT:     c += 3 * _t5;
// CHECK-NEXT:     _t7 = d;
// CHECK-NEXT:     _t8 = j;
// CHECK-NEXT:     _t6 = 3 * _t8;
// CHECK-NEXT:     d *= _t6;
// CHECK-NEXT:     {
// CHECK-NEXT:         *_d_a += 1;
// CHECK-NEXT:         *_d_c += 1;
// CHECK-NEXT:         *_d_d += 1;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r_d3 = *_d_d;
// CHECK-NEXT:         *_d_d += _r_d3 * _t6;
// CHECK-NEXT:         double _r7 = _t7 * _r_d3;
// CHECK-NEXT:         double _r8 = _r7 * _t8;
// CHECK-NEXT:         double _r9 = 3 * _r7;
// CHECK-NEXT:         * _d_j += _r9;
// CHECK-NEXT:         *_d_d -= _r_d3;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r_d2 = *_d_c;
// CHECK-NEXT:         *_d_c += _r_d2;
// CHECK-NEXT:         double _r5 = _r_d2 * _t5;
// CHECK-NEXT:         double _r6 = 3 * _r_d2;
// CHECK-NEXT:         * _d_i += _r6;
// CHECK-NEXT:         *_d_c -= _r_d2;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r_d1 = _d_b;
// CHECK-NEXT:         _d_b += _r_d1;
// CHECK-NEXT:         double _r3 = _r_d1 * _t4;
// CHECK-NEXT:         double _r4 = 2 * _r_d1;
// CHECK-NEXT:         * _d_i += _r4;
// CHECK-NEXT:         _d_b -= _r_d1;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r_d0 = *_d_a;
// CHECK-NEXT:         *_d_a += _r_d0 * _t2;
// CHECK-NEXT:         double _r2 = _t3 * _r_d0;
// CHECK-NEXT:         * _d_i += _r2;
// CHECK-NEXT:         *_d_a -= _r_d0;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r0 = _d_b * _t0;
// CHECK-NEXT:         * _d_i += _r0;
// CHECK-NEXT:         double _r1 = _t1 * _d_b;
// CHECK-NEXT:         * _d_j += _r1;
// CHECK-NEXT:     }
// CHECK-NEXT: }

double f16(double i, double j) {
  double& a = i;
  double& b = a;
  double& c = b;
  c *= 4*j;
  return i;
}

// CHECK: void f16_grad(double i, double j, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j) {
// CHECK-NEXT:     double *_d_a = 0;
// CHECK-NEXT:     double *_d_b = 0;
// CHECK-NEXT:     double *_d_c = 0;
// CHECK-NEXT:     double _t0;
// CHECK-NEXT:     double _t1;
// CHECK-NEXT:     double _t2;
// CHECK-NEXT:     _d_a = &* _d_i;
// CHECK-NEXT:     double &a = i;
// CHECK-NEXT:     _d_b = &*_d_a;
// CHECK-NEXT:     double &b = a;
// CHECK-NEXT:     _d_c = &*_d_b;
// CHECK-NEXT:     double &c = b;
// CHECK-NEXT:     _t1 = c;
// CHECK-NEXT:     _t2 = j;
// CHECK-NEXT:     _t0 = 4 * _t2;
// CHECK-NEXT:     c *= _t0;
// CHECK-NEXT:     * _d_i += 1;
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r_d0 = *_d_c;
// CHECK-NEXT:         *_d_c += _r_d0 * _t0;
// CHECK-NEXT:         double _r0 = _t1 * _r_d0;
// CHECK-NEXT:         double _r1 = _r0 * _t2;
// CHECK-NEXT:         double _r2 = 4 * _r0;
// CHECK-NEXT:         * _d_j += _r2;
// CHECK-NEXT:         *_d_c -= _r_d0;
// CHECK-NEXT:     }
// CHECK-NEXT: }

double f17(double i, double j, double k) {
  j = 2*i;
  return j;
}

// CHECK: void f17_grad_0(double i, double j, double k, clad::array_ref<double> _d_i) {
// CHECK-NEXT:     double _d_j = 0;
// CHECK-NEXT:     double _d_k = 0;
// CHECK-NEXT:     double _t0;
// CHECK-NEXT:     _t0 = i;
// CHECK-NEXT:     j = 2 * _t0;
// CHECK-NEXT:     _d_j += 1;
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r_d0 = _d_j;
// CHECK-NEXT:         double _r0 = _r_d0 * _t0;
// CHECK-NEXT:         double _r1 = 2 * _r_d0;
// CHECK-NEXT:         * _d_i += _r1;
// CHECK-NEXT:         _d_j -= _r_d0;
// CHECK-NEXT:     }
// CHECK-NEXT: }

double f18(double i, double j, double k) {
  k = 2*i + 2*j;
  k += i;
  return k;
}

// CHECK: void f18_grad_0_1(double i, double j, double k, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j) {
// CHECK-NEXT:     double _d_k = 0;
// CHECK-NEXT:     double _t0;
// CHECK-NEXT:     double _t1;
// CHECK-NEXT:     _t0 = i;
// CHECK-NEXT:     _t1 = j;
// CHECK-NEXT:     k = 2 * _t0 + 2 * _t1;
// CHECK-NEXT:     k += i;
// CHECK-NEXT:     _d_k += 1;
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r_d1 = _d_k;
// CHECK-NEXT:         _d_k += _r_d1;
// CHECK-NEXT:         * _d_i += _r_d1;
// CHECK-NEXT:         _d_k -= _r_d1;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r_d0 = _d_k;
// CHECK-NEXT:         double _r0 = _r_d0 * _t0;
// CHECK-NEXT:         double _r1 = 2 * _r_d0;
// CHECK-NEXT:         * _d_i += _r1;
// CHECK-NEXT:         double _r2 = _r_d0 * _t1;
// CHECK-NEXT:         double _r3 = 2 * _r_d0;
// CHECK-NEXT:         * _d_j += _r3;
// CHECK-NEXT:         _d_k -= _r_d0;
// CHECK-NEXT:     }
// CHECK-NEXT: }

double f19(double a, double b) {
  return std::fma(a, b, b);
}

//CHECK: void f19_grad(double a, double b, clad::array_ref<double> _d_a, clad::array_ref<double> _d_b) {
//CHECK-NEXT:     double _t0;
//CHECK-NEXT:     double _t1;
//CHECK-NEXT:     double _t2;
//CHECK-NEXT:     _t0 = a;
//CHECK-NEXT:     _t1 = b;
//CHECK-NEXT:     _t2 = b;
//CHECK-NEXT:     {
//CHECK-NEXT:         double _grad0 = 0.;
//CHECK-NEXT:         double _grad1 = 0.;
//CHECK-NEXT:         double _grad2 = 0.;
//CHECK-NEXT:         clad::custom_derivatives::fma_pullback(_t0, _t1, _t2, 1, &_grad0, &_grad1, &_grad2);
//CHECK-NEXT:         double _r0 = _grad0;
//CHECK-NEXT:         * _d_a += _r0;
//CHECK-NEXT:         double _r1 = _grad1;
//CHECK-NEXT:         * _d_b += _r1;
//CHECK-NEXT:         double _r2 = _grad2;
//CHECK-NEXT:         * _d_b += _r2;
//CHECK-NEXT:     }
//CHECK-NEXT: }


#define TEST(F, x, y)                                                          \
  {                                                                            \
    result[0] = 0;                                                             \
    result[1] = 0;                                                             \
    auto F##grad = clad::gradient(F);                                          \
    F##grad.execute(x, y, &result[0], &result[1]);                             \
    printf("{%.2f, %.2f}\n", result[0], result[1]);                            \
  }

template<typename FirstArg>
void set_to_zero(FirstArg firstArg) {
  *firstArg = 0;
}

template<typename FirstArg, typename... Args>
void set_to_zero(FirstArg firstArg, Args... args) {
  *firstArg = 0;
  set_to_zero(args...);
}

template<typename FirstArg>
void display(FirstArg firstArg) {
  printf("%.2f", *firstArg);
}

template<typename FirstArg, typename... Args>
void display(FirstArg firstArg, Args... args) {
  printf("%.2f, ", *firstArg);
  display(args...);
}

#define VAR_TEST(F, args, x, y, z, ...) {\
  set_to_zero(__VA_ARGS__);\
  auto d_##F = clad::gradient(F, args);\
  d_##F.execute(x, y, z, __VA_ARGS__);\
  printf("{");\
  display(__VA_ARGS__);\
  printf("}\n");\
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
  TEST(f14, 3, 4);  // CHECK-EXEC: {96.00, 0.00}
  TEST(f15, 3, 4);  // CHECK-EXEC: {30.00, 33.00}
  TEST(f16, 3, 4);  // CHECK-EXEC: {16.00, 12.00}
  VAR_TEST(f17, "i", 3, 4, 5, &result[0]);  // CHECK-EXEC: {2.00}
  VAR_TEST(f18, "i, j", 3, 4, 5, &result[0], &result[1]); // CHECK-EXEC: {3.00, 2.00}
  TEST(f19, 1, 2); // CHECK-EXEC: {2.00, 2.00}
}
