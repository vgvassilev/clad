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
//CHECK-NEXT:       double _t0 = 1 * x;
//CHECK-NEXT:       double _t1 = 0 * 1;
//CHECK-NEXT:       _result[0UL] += _t1;
//CHECK-NEXT:       double _t2 = 1 * y;
//CHECK-NEXT:       double _t3 = 1 * 1;
//CHECK-NEXT:       _result[1UL] += _t3;
//CHECK-NEXT:       double _t4 = 1 * z;
//CHECK-NEXT:       double _t5 = 2 * 1;
//CHECK-NEXT:       _result[2UL] += _t5;
//CHECK-NEXT:       return;
//CHECK-NEXT:   }

// x
//CHECK:   void f_1_grad_0(double x, double y, double z, double *_result) {
//CHECK-NEXT:       double _t0 = 1 * x;
//CHECK-NEXT:       double _t1 = 0 * 1;
//CHECK-NEXT:       _result[0UL] += _t1;
//CHECK-NEXT:       double _t2 = 1 * y;
//CHECK-NEXT:       double _t3 = 1 * 1;
//CHECK-NEXT:       double _t4 = 1 * z;
//CHECK-NEXT:       double _t5 = 2 * 1;
//CHECK-NEXT:       return;
//CHECK-NEXT:   }

// y
//CHECK:   void f_1_grad_1(double x, double y, double z, double *_result) {
//CHECK-NEXT:       double _t0 = 1 * x;
//CHECK-NEXT:       double _t1 = 0 * 1;
//CHECK-NEXT:       double _t2 = 1 * y;
//CHECK-NEXT:       double _t3 = 1 * 1;
//CHECK-NEXT:       _result[0UL] += _t3;
//CHECK-NEXT:       double _t4 = 1 * z;
//CHECK-NEXT:       double _t5 = 2 * 1;
//CHECK-NEXT:       return;
//CHECK-NEXT:   }

// z
//CHECK:   void f_1_grad_2(double x, double y, double z, double *_result) {
//CHECK-NEXT:       double _t0 = 1 * x;
//CHECK-NEXT:       double _t1 = 0 * 1;
//CHECK-NEXT:       double _t2 = 1 * y;
//CHECK-NEXT:       double _t3 = 1 * 1;
//CHECK-NEXT:       double _t4 = 1 * z;
//CHECK-NEXT:       double _t5 = 2 * 1;
//CHECK-NEXT:       _result[0UL] += _t5;
//CHECK-NEXT:       return;
//CHECK-NEXT:   }

// x, y
//CHECK:   void f_1_grad_0_1(double x, double y, double z, double *_result) {
//CHECK-NEXT:       double _t0 = 1 * x;
//CHECK-NEXT:       double _t1 = 0 * 1;
//CHECK-NEXT:       _result[0UL] += _t1;
//CHECK-NEXT:       double _t2 = 1 * y;
//CHECK-NEXT:       double _t3 = 1 * 1;
//CHECK-NEXT:       _result[1UL] += _t3;
//CHECK-NEXT:       double _t4 = 1 * z;
//CHECK-NEXT:       double _t5 = 2 * 1;
//CHECK-NEXT:       return;
//CHECK-NEXT:   }

// y, x
//CHECK:   void f_1_grad_1_0(double x, double y, double z, double *_result) {
//CHECK-NEXT:       double _t0 = 1 * x;
//CHECK-NEXT:       double _t1 = 0 * 1;
//CHECK-NEXT:       _result[1UL] += _t1;
//CHECK-NEXT:       double _t2 = 1 * y;
//CHECK-NEXT:       double _t3 = 1 * 1;
//CHECK-NEXT:       _result[0UL] += _t3;
//CHECK-NEXT:       double _t4 = 1 * z;
//CHECK-NEXT:       double _t5 = 2 * 1;
//CHECK-NEXT:       return;
//CHECK-NEXT:   }

// x, y, z
//CHECK:   void f_1_grad(double x, double y, double z, double *_result) {
//CHECK-NEXT:       double _t0 = 1 * x;
//CHECK-NEXT:       double _t1 = 0 * 1;
//CHECK-NEXT:       _result[0UL] += _t1;
//CHECK-NEXT:       double _t2 = 1 * y;
//CHECK-NEXT:       double _t3 = 1 * 1;
//CHECK-NEXT:       _result[1UL] += _t3;
//CHECK-NEXT:       double _t4 = 1 * z;
//CHECK-NEXT:       double _t5 = 2 * 1;
//CHECK-NEXT:       _result[2UL] += _t5;
//CHECK-NEXT:       return;
//CHECK-NEXT:   }

// z, y, z
//CHECK:   void f_1_grad_2_1_0(double x, double y, double z, double *_result) {
//CHECK-NEXT:       double _t0 = 1 * x;
//CHECK-NEXT:       double _t1 = 0 * 1;
//CHECK-NEXT:       _result[2UL] += _t1;
//CHECK-NEXT:       double _t2 = 1 * y;
//CHECK-NEXT:       double _t3 = 1 * 1;
//CHECK-NEXT:       _result[1UL] += _t3;
//CHECK-NEXT:       double _t4 = 1 * z;
//CHECK-NEXT:       double _t5 = 2 * 1;
//CHECK-NEXT:       _result[0UL] += _t5;
//CHECK-NEXT:       return;
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
