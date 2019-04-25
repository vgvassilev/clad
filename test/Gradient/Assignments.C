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
//CHECK-NEXT:       goto _label0;
//CHECK-NEXT:     _label0:
//CHECK-NEXT:       _result[1UL] += 1;
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r_d_x0 = 0;
//CHECK-NEXT:           _result[1UL] += _result[0UL];
//CHECK-NEXT:           _result[0UL] = _r_d_x0;
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
//CHECK-NEXT:       goto _label0;
//CHECK-NEXT:     _label0:
//CHECK-NEXT:       _result[0UL] += 1;
//CHECK-NEXT:       if (_cond0) {
//CHECK-NEXT:           double _r_d_x0 = 0;
//CHECK-NEXT:           _result[1UL] += _result[0UL];
//CHECK-NEXT:           _result[0UL] = _r_d_x0;
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
//CHECK-NEXT:       x = _t1 * x;
//CHECK-NEXT:       _t3 = x;
//CHECK-NEXT:       _t2 = x;
//CHECK-NEXT:       y = _t3 * x;
//CHECK-NEXT:       x = y;
//CHECK-NEXT:       goto _label0;
//CHECK-NEXT:     _label0:
//CHECK-NEXT:       _result[1UL] += 1;
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r_d_x2 = 0;
//CHECK-NEXT:           _result[1UL] += _result[0UL];
//CHECK-NEXT:           _result[0UL] = _r_d_x2;
//CHECK-NEXT:       }
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r_d_y0 = 0;
//CHECK-NEXT:           double _r2 = _result[1UL] * _t2;
//CHECK-NEXT:           _result[0UL] += _r2;
//CHECK-NEXT:           double _r3 = _t3 * _result[1UL];
//CHECK-NEXT:           _result[0UL] += _r3;
//CHECK-NEXT:           _result[1UL] = _r_d_y0;
//CHECK-NEXT:       }
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r_d_x1 = 0;
//CHECK-NEXT:           double _r0 = _result[0UL] * _t0;
//CHECK-NEXT:           _r_d_x1 += _r0;
//CHECK-NEXT:           double _r1 = _t1 * _result[0UL];
//CHECK-NEXT:           _r_d_x1 += _r1;
//CHECK-NEXT:           _result[0UL] = _r_d_x1;
//CHECK-NEXT:       }
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r_d_x0 = 0;
//CHECK-NEXT:           _r_d_x0 += _result[0UL];
//CHECK-NEXT:           _result[0UL] = _r_d_x0;
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
//CHECK-NEXT:       goto _label0;
//CHECK-NEXT:     _label0:
//CHECK-NEXT:       _result[1UL] += 1;
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r_d_x0 = 0;
//CHECK-NEXT:           _result[0UL] = _r_d_x0;
//CHECK-NEXT:       }
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r_d_y0 = 0;
//CHECK-NEXT:           _result[0UL] += _result[1UL];
//CHECK-NEXT:           _result[1UL] = _r_d_y0;
//CHECK-NEXT:       }
//CHECK-NEXT:   }

// = sign(x) * sign(y) * x * x
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
//CHECK-NEXT:       double t = _t1 * x;
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
//CHECK-NEXT:       goto _label1;
//CHECK-NEXT:     _label1:
//CHECK-NEXT:       _d_t += 1;
//CHECK-NEXT:       if (_cond1) {
//CHECK-NEXT:           {
//CHECK-NEXT:               double _r_d_t1 = 0;
//CHECK-NEXT:               _r_d_t1 += -_d_t;
//CHECK-NEXT:               _d_t = _r_d_t1;
//CHECK-NEXT:           }
//CHECK-NEXT:           _d_t += _d_z;
//CHECK-NEXT:       }
//CHECK-NEXT:       if (_cond0) {
//CHECK-NEXT:         _label0:
//CHECK-NEXT:           _d_t += 1;
//CHECK-NEXT:           {
//CHECK-NEXT:               double _r_d_t0 = 0;
//CHECK-NEXT:               _r_d_t0 += -_d_t;
//CHECK-NEXT:               _d_t = _r_d_t0;
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
//CHECK-NEXT:       double t = _t1 * x;
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
//CHECK-NEXT:       goto _label1;
//CHECK-NEXT:     _label1:
//CHECK-NEXT:       _d_t += 1;
//CHECK-NEXT:       if (_cond1) {
//CHECK-NEXT:           {
//CHECK-NEXT:               double _r_d_t1 = 0;
//CHECK-NEXT:               _r_d_t1 += -_d_t;
//CHECK-NEXT:               _d_t = _r_d_t1;
//CHECK-NEXT:           }
//CHECK-NEXT:           _d_t += _d_z;
//CHECK-NEXT:       }
//CHECK-NEXT:       if (_cond0) {
//CHECK-NEXT:         _label0:
//CHECK-NEXT:           _d_t += 1;
//CHECK-NEXT:           {
//CHECK-NEXT:               double _r_d_t0 = 0;
//CHECK-NEXT:               _r_d_t0 += -_d_t;
//CHECK-NEXT:               _d_t = _r_d_t0;
//CHECK-NEXT:           }
//CHECK-NEXT:       }
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r0 = _d_t * _t0;
//CHECK-NEXT:           _result[0UL] += _r0;
//CHECK-NEXT:           double _r1 = _t1 * _d_t;
//CHECK-NEXT:           _result[0UL] += _r1;
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
  TEST(f5, -3, 4); // NOT-CHECK-EXEC: {-6.00, 0.00} 
  TEST(f6, 3, -4); // CHECK-EXEC: {-6.00, 0.00} 
}
