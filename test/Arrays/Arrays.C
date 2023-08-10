// RUN: %cladclang %s -I%S/../../include -oArrays.out 2>&1 | FileCheck %s
// RUN: ./Arrays.out | FileCheck -check-prefix=CHECK-EXEC %s

//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"
#include <cmath>

extern "C" int printf(const char* fmt, ...);

double sum(double x, double y, double z) {
  double vars[] = {x, y, z};
  double s = 0;
  for (int i = 0; i < 3; i++)
    s = s + vars[i];
  return s;
}

//CHECK:   double sum_darg0(double x, double y, double z) {
//CHECK-NEXT:       double _d_x = 1;
//CHECK-NEXT:       double _d_y = 0;
//CHECK-NEXT:       double _d_z = 0;
//CHECK-NEXT:       double _d_vars[3] = {_d_x, _d_y, _d_z};
//CHECK-NEXT:       double vars[3] = {x, y, z};
//CHECK-NEXT:       double _d_s = 0;
//CHECK-NEXT:       double s = 0;
//CHECK-NEXT:       {
//CHECK-NEXT:           int _d_i = 0;
//CHECK-NEXT:           for (int i = 0; i < 3; i++) {
//CHECK-NEXT:               _d_s = _d_s + _d_vars[i];
//CHECK-NEXT:               s = s + vars[i];
//CHECK-NEXT:           }
//CHECK-NEXT:       }
//CHECK-NEXT:       return _d_s;
//CHECK-NEXT:   }

double sum_squares(double x, double y, double z) {
  double vars[3] = {x, y, z};
  double squares[3];
  for (int i = 0; i < 3; i++)
    squares[i] = vars[i] * vars[i];
  double s = 0;
  for (int i = 0; i < 3; i++)
    s = s + squares[i];
  return s;
}

//CHECK:   double sum_squares_darg0(double x, double y, double z) {
//CHECK-NEXT:       double _d_x = 1;
//CHECK-NEXT:       double _d_y = 0;
//CHECK-NEXT:       double _d_z = 0;
//CHECK-NEXT:       double _d_vars[3] = {_d_x, _d_y, _d_z};
//CHECK-NEXT:       double vars[3] = {x, y, z};
//CHECK-NEXT:       double _d_squares[3];
//CHECK-NEXT:       double squares[3];
//CHECK-NEXT:       {
//CHECK-NEXT:           int _d_i = 0;
//CHECK-NEXT:           for (int i = 0; i < 3; i++) {
//CHECK-NEXT:               _d_squares[i] = _d_vars[i] * vars[i] + vars[i] * _d_vars[i];
//CHECK-NEXT:               squares[i] = vars[i] * vars[i];
//CHECK-NEXT:           }
//CHECK-NEXT:       }
//CHECK-NEXT:       double _d_s = 0;
//CHECK-NEXT:       double s = 0;
//CHECK-NEXT:       {
//CHECK-NEXT:           int _d_i = 0;
//CHECK-NEXT:           for (int i = 0; i < 3; i++) {
//CHECK-NEXT:               _d_s = _d_s + _d_squares[i];
//CHECK-NEXT:               s = s + squares[i];
//CHECK-NEXT:           }
//CHECK-NEXT:       }
//CHECK-NEXT:       return _d_s;
//CHECK-NEXT:   }

double const_dot_product(double x, double y, double z) {
  double vars[] = { x, y, z };
  double consts[] = { 1, 2, 3 };
  return vars[0] * consts[0] + vars[1] * consts[1] + vars[2] * consts[2];
}

//CHECK:   double const_dot_product_darg0(double x, double y, double z) {
//CHECK-NEXT:       double _d_x = 1;
//CHECK-NEXT:       double _d_y = 0;
//CHECK-NEXT:       double _d_z = 0;
//CHECK-NEXT:       double _d_vars[3] = {_d_x, _d_y, _d_z};
//CHECK-NEXT:       double vars[3] = {x, y, z};
//CHECK-NEXT:       double _d_consts[3] = {0, 0, 0};
//CHECK-NEXT:       double consts[3] = {1, 2, 3};
//CHECK-NEXT:       return _d_vars[0] * consts[0] + vars[0] * _d_consts[0] + _d_vars[1] * consts[1] + vars[1] * _d_consts[1] + _d_vars[2] * consts[2] + vars[2] * _d_consts[2];
//CHECK-NEXT:   }

//CHECK:   void const_dot_product_grad(double x, double y, double z, clad::array_ref<double> _d_x, clad::array_ref<double> _d_y, clad::array_ref<double> _d_z) {
//CHECK-NEXT:       clad::array<double> _d_vars(3UL);
//CHECK-NEXT:       clad::array<double> _d_consts(3UL);
//CHECK-NEXT:       double _t0;
//CHECK-NEXT:       double _t1;
//CHECK-NEXT:       double _t2;
//CHECK-NEXT:       double _t3;
//CHECK-NEXT:       double _t4;
//CHECK-NEXT:       double _t5;
//CHECK-NEXT:       double vars[3] = {x, y, z};
//CHECK-NEXT:       double consts[3] = {1, 2, 3};
//CHECK-NEXT:       _t1 = vars[0];
//CHECK-NEXT:       _t0 = consts[0];
//CHECK-NEXT:       _t3 = vars[1];
//CHECK-NEXT:       _t2 = consts[1];
//CHECK-NEXT:       _t5 = vars[2];
//CHECK-NEXT:       _t4 = consts[2];
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r0 = 1 * _t0;
//CHECK-NEXT:           _d_vars[0] += _r0;
//CHECK-NEXT:           double _r1 = _t1 * 1;
//CHECK-NEXT:           _d_consts[0] += _r1;
//CHECK-NEXT:           double _r2 = 1 * _t2;
//CHECK-NEXT:           _d_vars[1] += _r2;
//CHECK-NEXT:           double _r3 = _t3 * 1;
//CHECK-NEXT:           _d_consts[1] += _r3;
//CHECK-NEXT:           double _r4 = 1 * _t4;
//CHECK-NEXT:           _d_vars[2] += _r4;
//CHECK-NEXT:           double _r5 = _t5 * 1;
//CHECK-NEXT:           _d_consts[2] += _r5;
//CHECK-NEXT:       }
//CHECK-NEXT:       {
//CHECK-NEXT:           * _d_x += _d_vars[0];
//CHECK-NEXT:           * _d_y += _d_vars[1];
//CHECK-NEXT:           * _d_z += _d_vars[2];
//CHECK-NEXT:       }
//CHECK-NEXT:   }

double const_matmul_sum(double a, double b, double c, double d) {
  double A[2][2] = {{a, b}, {c, d}};
  double B[2][2] = {{1, 2}, {3, 4}};
  double C[2][2] = {{A[0][0] * B[0][0] + A[0][1] * B[1][0],
                     A[0][0] * B[0][1] + A[0][1] * B[1][1]},
                    {A[1][0] * B[0][0] + A[1][1] * B[1][0],
                     A[1][0] * B[0][1] + A[1][1] * B[1][1]}};
  return C[0][0] + C[0][1] + C[1][0] + C[1][1];
}

// FIXME: Add multi index support to clad::array and clad::array_ref
//:   double const_matmul_sum_darg0(double a, double b, double c, double d) {
//:       double _d_a = 1;
//:       double _d_b = 0;
//:       double _d_c = 0;
//:       double _d_d = 0;
//:       double _d_A[2][2] = {{[{][{]}}_d_a, _d_b}, {_d_c, _d_d}};
//:       double A[2][2] = {{[{][{]}}a, b}, {c, d}};
//:       double _d_B[2][2] = {{[{][{]}}0, 0}, {0, 0}};
//:       double B[2][2] = {{[{][{]}}1, 2}, {3, 4}};
//:       double _d_C[2][2] = {{[{][{]}}_d_A[0][0] * B[0][0] + A[0][0] * _d_B[0][0] + _d_A[0][1] * B[1][0] + A[0][1] * _d_B[1][0], _d_A[0][0] * B[0][1] + A[0][0] * _d_B[0][1] + _d_A[0][1] * B[1][1] + A[0][1] * _d_B[1][1]}, {_d_A[1][0] * B[0][0] + A[1][0] * _d_B[0][0] + _d_A[1][1] * B[1][0] + A[1][1] * _d_B[1][0], _d_A[1][0] * B[0][1] + A[1][0] * _d_B[0][1] + _d_A[1][1] * B[1][1] + A[1][1] * _d_B[1][1]}};
//:       double C[2][2] = {{[{][{]}}A[0][0] * B[0][0] + A[0][1] * B[1][0], A[0][0] * B[0][1] + A[0][1] * B[1][1]}, {A[1][0] * B[0][0] + A[1][1] * B[1][0], A[1][0] * B[0][1] + A[1][1] * B[1][1]}};
//:       return _d_C[0][0] + _d_C[0][1] + _d_C[1][0] + _d_C[1][1];
//:   }

//:   void const_matmul_sum_grad(double a, double b, double c, double d, clad::array_ref<double> _d_a, clad::array_ref<double> _d_b, clad::array_ref<double> _d_c, clad::array_ref<double> _d_d) {
//:       double _d_A[2][2] = {};
//:       double _d_B[2][2] = {};
//:       double _t0;
//:       double _t1;
//:       double _t2;
//:       double _t3;
//:       double _t4;
//:       double _t5;
//:       double _t6;
//:       double _t7;
//:       double _t8;
//:       double _t9;
//:       double _t10;
//:       double _t11;
//:       double _t12;
//:       double _t13;
//:       double _t14;
//:       double _t15;
//:       double _d_C[2][2] = {};
//:       double A[2][2] = {{[{][{]}}a, b}, {c, d}};
//:       double B[2][2] = {{[{][{]}}1, 2}, {3, 4}};
//:       _t1 = A[0][0];
//:       _t0 = B[0][0];
//:       _t3 = A[0][1];
//:       _t2 = B[1][0];
//:       _t5 = A[0][0];
//:       _t4 = B[0][1];
//:       _t7 = A[0][1];
//:       _t6 = B[1][1];
//:       _t9 = A[1][0];
//:       _t8 = B[0][0];
//:       _t11 = A[1][1];
//:       _t10 = B[1][0];
//:       _t13 = A[1][0];
//:       _t12 = B[0][1];
//:       _t15 = A[1][1];
//:       _t14 = B[1][1];
//:       double C[2][2] = {{[{][{]}}_t1 * _t0 + _t3 * _t2, _t5 * _t4 + _t7 * _t6}, {_t9 * _t8 + _t11 * _t10, _t13 * _t12 + _t15 * _t14}};
//:       {
//:           _d_C[0][0] += 1;
//:           _d_C[0][1] += 1;
//:           _d_C[1][0] += 1;
//:           _d_C[1][1] += 1;
//:       }
//:       {
//:           double _r0 = _d_C[0][0] * _t0;
//:           _d_A[0][0] += _r0;
//:           double _r1 = _t1 * _d_C[0][0];
//:           _d_B[0][0] += _r1;
//:           double _r2 = _d_C[0][0] * _t2;
//:           _d_A[0][1] += _r2;
//:           double _r3 = _t3 * _d_C[0][0];
//:           _d_B[1][0] += _r3;
//:           double _r4 = _d_C[0][1] * _t4;
//:           _d_A[0][0] += _r4;
//:           double _r5 = _t5 * _d_C[0][1];
//:           _d_B[0][1] += _r5;
//:           double _r6 = _d_C[0][1] * _t6;
//:           _d_A[0][1] += _r6;
//:           double _r7 = _t7 * _d_C[0][1];
//:           _d_B[1][1] += _r7;
//:           double _r8 = _d_C[1][0] * _t8;
//:           _d_A[1][0] += _r8;
//:           double _r9 = _t9 * _d_C[1][0];
//:           _d_B[0][0] += _r9;
//:           double _r10 = _d_C[1][0] * _t10;
//:           _d_A[1][1] += _r10;
//:           double _r11 = _t11 * _d_C[1][0];
//:           _d_B[1][0] += _r11;
//:           double _r12 = _d_C[1][1] * _t12;
//:           _d_A[1][0] += _r12;
//:           double _r13 = _t13 * _d_C[1][1];
//:           _d_B[0][1] += _r13;
//:           double _r14 = _d_C[1][1] * _t14;
//:           _d_A[1][1] += _r14;
//:           double _r15 = _t15 * _d_C[1][1];
//:           _d_B[1][1] += _r15;
//:       }
//:       {
//:           * _d_a += _d_A[0][0];
//:           * _d_b += _d_A[0][1];
//:           * _d_c += _d_A[1][0];
//:           * _d_d += _d_A[1][1];
//:       }
//:   }

int main () { // expected-no-diagnostics
  auto dsum = clad::differentiate(sum, 0);
  printf("%.2f\n", dsum.execute(11, 12, 13)); // CHECK-EXEC: 1.00

  auto dssum = clad::differentiate(sum_squares, 0);
  printf("%.2f\n", dssum.execute(11, 12, 13)); // CHECK-EXEC: 22.00
  auto dcdp = clad::differentiate(const_dot_product, 0);
  printf("%.2f\n", dcdp.execute(11, 12, 13)); // CHECK-EXEC: 1.00

  auto gradcdp = clad::gradient(const_dot_product);
  double result[3] = {};
  gradcdp.execute(11, 12, 13, &result[0], &result[1], &result[2]);
  printf("{%.2f, %.2f, %.2f}\n", result[0], result[1], result[2]); // CHECK-EXEC: {1.00, 2.00, 3.00}

//  FIXME: Add multi index support to clad::array and clad::array_ref
//  auto dcms = clad::differentiate(const_matmul_sum, 0);
//  printf("%.2f\n", dcms.execute(11, 12, 13, 14)); // : 3.00

//  auto grad = clad::gradient(const_matmul_sum);
//  double result2[4] = {};
//  grad.execute(
//      11, 12, 13, 14, &result2[0], &result2[1], &result2[2], &result2[3]);
//  printf("{%.2f, %.2f, %.2f, %.2f}\n", result2[0], result2[1], result2[2], result2[3]); // : {3.00, 7.00, 3.00, 7.00}
  return 0;
}
