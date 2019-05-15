// RUN: %cladclang %s -lm -I%S/../../include -oGradientDiffInterface.out 2>&1 | FileCheck %s
// RUN: ./GradientDiffInterface.out | FileCheck -check-prefix=CHECK-EXEC %s

#include "clad/Differentiator/Differentiator.h"

//CHECK-NOT: {{.*error|warning|note:.*}}

extern "C" int printf(const char* fmt, ...);

double f_1(double x, double y, double z) {
  return 0 * x + 1 * y + 2 * z;
}

// all
//CHECK:   void f_1_grad(double x, double y, double z, double *_result) {
//CHECK-NEXT:       double _t0;
//CHECK-NEXT:       double _t1;
//CHECK-NEXT:       double _t2;
//CHECK-NEXT:       _t0 = x;
//CHECK-NEXT:       _t1 = y;
//CHECK-NEXT:       _t2 = z;
//CHECK-NEXT:       double f_1_return = 0 * _t0 + 1 * _t1 + 2 * _t2;
//CHECK-NEXT:       goto _label0;
//CHECK-NEXT:     _label0:
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r0 = 1 * _t0;
//CHECK-NEXT:           double _r1 = 0 * 1;
//CHECK-NEXT:           _result[0UL] += _r1;
//CHECK-NEXT:           double _r2 = 1 * _t1;
//CHECK-NEXT:           double _r3 = 1 * 1;
//CHECK-NEXT:           _result[1UL] += _r3;
//CHECK-NEXT:           double _r4 = 1 * _t2;
//CHECK-NEXT:           double _r5 = 2 * 1;
//CHECK-NEXT:           _result[2UL] += _r5;
//CHECK-NEXT:       }
//CHECK-NEXT:   }

// x
//CHECK:   void f_1_grad_0(double x, double y, double z, double *_result) {
//CHECK-NEXT:       double _t0;
//CHECK-NEXT:       double _t1;
//CHECK-NEXT:       double _t2;
//CHECK-NEXT:       _t0 = x;
//CHECK-NEXT:       _t1 = y;
//CHECK-NEXT:       _t2 = z;
//CHECK-NEXT:       double f_1_return = 0 * _t0 + 1 * _t1 + 2 * _t2;
//CHECK-NEXT:       goto _label0;
//CHECK-NEXT:     _label0:
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r0 = 1 * _t0;
//CHECK-NEXT:           double _r1 = 0 * 1;
//CHECK-NEXT:           _result[0UL] += _r1;
//CHECK-NEXT:           double _r2 = 1 * _t1;
//CHECK-NEXT:           double _r3 = 1 * 1;
//CHECK-NEXT:           double _r4 = 1 * _t2;
//CHECK-NEXT:           double _r5 = 2 * 1;
//CHECK-NEXT:       }
//CHECK-NEXT:   }

// y
//CHECK:   void f_1_grad_1(double x, double y, double z, double *_result) {
//CHECK-NEXT:       double _t0;
//CHECK-NEXT:       double _t1;
//CHECK-NEXT:       double _t2;
//CHECK-NEXT:       _t0 = x;
//CHECK-NEXT:       _t1 = y;
//CHECK-NEXT:       _t2 = z;
//CHECK-NEXT:       double f_1_return = 0 * _t0 + 1 * _t1 + 2 * _t2;
//CHECK-NEXT:       goto _label0;
//CHECK-NEXT:     _label0:
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r0 = 1 * _t0;
//CHECK-NEXT:           double _r1 = 0 * 1;
//CHECK-NEXT:           double _r2 = 1 * _t1;
//CHECK-NEXT:           double _r3 = 1 * 1;
//CHECK-NEXT:           _result[0UL] += _r3;
//CHECK-NEXT:           double _r4 = 1 * _t2;
//CHECK-NEXT:           double _r5 = 2 * 1;
//CHECK-NEXT:       }
//CHECK-NEXT:   }

// z
//CHECK:   void f_1_grad_2(double x, double y, double z, double *_result) {
//CHECK-NEXT:       double _t0;
//CHECK-NEXT:       double _t1;
//CHECK-NEXT:       double _t2;
//CHECK-NEXT:       _t0 = x;
//CHECK-NEXT:       _t1 = y;
//CHECK-NEXT:       _t2 = z;
//CHECK-NEXT:       double f_1_return = 0 * _t0 + 1 * _t1 + 2 * _t2;
//CHECK-NEXT:       goto _label0;
//CHECK-NEXT:     _label0:
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r0 = 1 * _t0;
//CHECK-NEXT:           double _r1 = 0 * 1;
//CHECK-NEXT:           double _r2 = 1 * _t1;
//CHECK-NEXT:           double _r3 = 1 * 1;
//CHECK-NEXT:           double _r4 = 1 * _t2;
//CHECK-NEXT:           double _r5 = 2 * 1;
//CHECK-NEXT:           _result[0UL] += _r5;
//CHECK-NEXT:       }
//CHECK-NEXT:   }

// x, y
//CHECK:   void f_1_grad_0_1(double x, double y, double z, double *_result) {
//CHECK-NEXT:       double _t0;
//CHECK-NEXT:       double _t1;
//CHECK-NEXT:       double _t2;
//CHECK-NEXT:       _t0 = x;
//CHECK-NEXT:       _t1 = y;
//CHECK-NEXT:       _t2 = z;
//CHECK-NEXT:       double f_1_return = 0 * _t0 + 1 * _t1 + 2 * _t2;
//CHECK-NEXT:       goto _label0;
//CHECK-NEXT:     _label0:
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r0 = 1 * _t0;
//CHECK-NEXT:           double _r1 = 0 * 1;
//CHECK-NEXT:           _result[0UL] += _r1;
//CHECK-NEXT:           double _r2 = 1 * _t1;
//CHECK-NEXT:           double _r3 = 1 * 1;
//CHECK-NEXT:           _result[1UL] += _r3;
//CHECK-NEXT:           double _r4 = 1 * _t2;
//CHECK-NEXT:           double _r5 = 2 * 1;
//CHECK-NEXT:       }
//CHECK-NEXT:   }

// y, x
//CHECK:   void f_1_grad_1_0(double x, double y, double z, double *_result) {
//CHECK-NEXT:       double _t0;
//CHECK-NEXT:       double _t1;
//CHECK-NEXT:       double _t2;
//CHECK-NEXT:       _t0 = x;
//CHECK-NEXT:       _t1 = y;
//CHECK-NEXT:       _t2 = z;
//CHECK-NEXT:       double f_1_return = 0 * _t0 + 1 * _t1 + 2 * _t2;
//CHECK-NEXT:       goto _label0;
//CHECK-NEXT:     _label0:
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r0 = 1 * _t0;
//CHECK-NEXT:           double _r1 = 0 * 1;
//CHECK-NEXT:           _result[1UL] += _r1;
//CHECK-NEXT:           double _r2 = 1 * _t1;
//CHECK-NEXT:           double _r3 = 1 * 1;
//CHECK-NEXT:           _result[0UL] += _r3;
//CHECK-NEXT:           double _r4 = 1 * _t2;
//CHECK-NEXT:           double _r5 = 2 * 1;
//CHECK-NEXT:       }
//CHECK-NEXT:   }

// x, y, z
//CHECK:   void f_1_grad(double x, double y, double z, double *_result) {
//CHECK-NEXT:       double _t0;
//CHECK-NEXT:       double _t1;
//CHECK-NEXT:       double _t2;
//CHECK-NEXT:       _t0 = x;
//CHECK-NEXT:       _t1 = y;
//CHECK-NEXT:       _t2 = z;
//CHECK-NEXT:       double f_1_return = 0 * _t0 + 1 * _t1 + 2 * _t2;
//CHECK-NEXT:       goto _label0;
//CHECK-NEXT:     _label0:
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r0 = 1 * _t0;
//CHECK-NEXT:           double _r1 = 0 * 1;
//CHECK-NEXT:           _result[0UL] += _r1;
//CHECK-NEXT:           double _r2 = 1 * _t1;
//CHECK-NEXT:           double _r3 = 1 * 1;
//CHECK-NEXT:           _result[1UL] += _r3;
//CHECK-NEXT:           double _r4 = 1 * _t2;
//CHECK-NEXT:           double _r5 = 2 * 1;
//CHECK-NEXT:           _result[2UL] += _r5;
//CHECK-NEXT:       }
//CHECK-NEXT:   }

// z, y, z
//CHECK:   void f_1_grad_2_1_0(double x, double y, double z, double *_result) {
//CHECK-NEXT:       double _t0;
//CHECK-NEXT:       double _t1;
//CHECK-NEXT:       double _t2;
//CHECK-NEXT:       _t0 = x;
//CHECK-NEXT:       _t1 = y;
//CHECK-NEXT:       _t2 = z;
//CHECK-NEXT:       double f_1_return = 0 * _t0 + 1 * _t1 + 2 * _t2;
//CHECK-NEXT:       goto _label0;
//CHECK-NEXT:     _label0:
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r0 = 1 * _t0;
//CHECK-NEXT:           double _r1 = 0 * 1;
//CHECK-NEXT:           _result[2UL] += _r1;
//CHECK-NEXT:           double _r2 = 1 * _t1;
//CHECK-NEXT:           double _r3 = 1 * 1;
//CHECK-NEXT:           _result[1UL] += _r3;
//CHECK-NEXT:           double _r4 = 1 * _t2;
//CHECK-NEXT:           double _r5 = 2 * 1;
//CHECK-NEXT:           _result[0UL] += _r5;
//CHECK-NEXT:       }
//CHECK-NEXT:   }

#define TEST(F) { \
  result[0] = 0; result[1] = 0; result[2] = 0;\
  F.execute(0, 0, 0, result);\
  printf("{%.2f, %.2f, %.2f}\n", result[0], result[1], result[2]); \
}

int main () {
  double result[3];

  auto f1_grad_all = clad::gradient(f_1);
  TEST(f1_grad_all); // CHECK-EXEC: {0.00, 1.00, 2.00}
  
  auto f1_grad_x = clad::gradient(f_1, "x");
  TEST(f1_grad_x); // CHECK-EXEC: {0.00, 0.00, 0.00}

  auto f1_grad_y = clad::gradient(f_1, "y");
  TEST(f1_grad_y); // CHECK-EXEC: {1.00, 0.00, 0.00}
  auto f1_grad_z = clad::gradient(f_1, "z");
  TEST(f1_grad_z); // CHECK-EXEC: {2.00, 0.00, 0.00}

  auto f1_grad_xy = clad::gradient(f_1, "x, y");
  TEST(f1_grad_xy); // CHECK-EXEC: {0.00, 1.00, 0.00}

  auto f1_grad_yz = clad::gradient(f_1, "y, x");
  TEST(f1_grad_yz); // CHECK-EXEC: {1.00, 0.00, 0.00}

  auto f1_grad_xyz = clad::gradient(f_1, "x, y, z");
  TEST(f1_grad_xyz); // CHECK-EXEC: {0.00, 1.00, 2.00}

  auto f1_grad_zyx = clad::gradient(f_1, "z,y,x");
  TEST(f1_grad_zyx); // CHECK-EXEC: {2.00, 1.00, 0.00}

  return 0;
}
