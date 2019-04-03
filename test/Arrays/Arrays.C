// RUN: %cladclang %s -lm -I%S/../../include -oArrays.out 2>&1 | FileCheck %s
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
//CHECK-NEXT:               double _t0 = vars[i];
//CHECK-NEXT:               double _t1 = vars[i];
//CHECK-NEXT:               _d_squares[i] = _d_vars[i] * _t1 + _t0 * _d_vars[i];
//CHECK-NEXT:               squares[i] = _t0 * _t1;
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
//CHECK-NEXT:       double _t0 = vars[0];
//CHECK-NEXT:       double _t1 = consts[0];
//CHECK-NEXT:       double _t2 = vars[1];
//CHECK-NEXT:       double _t3 = consts[1];
//CHECK-NEXT:       double _t4 = vars[2];
//CHECK-NEXT:       double _t5 = consts[2];
//CHECK-NEXT:       return _d_vars[0] * _t1 + _t0 * _d_consts[0] + _d_vars[1] * _t3 + _t2 * _d_consts[1] + _d_vars[2] * _t5 + _t4 * _d_consts[2];
//CHECK-NEXT:   }

//CHECK:   void const_dot_product_grad(double x, double y, double z, double *_result) {
//CHECK-NEXT:       double _d_vars[3] = {};
//CHECK-NEXT:       double vars[3] = {x, y, z};
//CHECK-NEXT:       double _d_consts[3] = {};
//CHECK-NEXT:       double consts[3] = {1, 2, 3};
//CHECK-NEXT:       double _t0 = 1 * consts[0];
//CHECK-NEXT:       _d_vars[0] += _t0;
//CHECK-NEXT:       double _t1 = vars[0] * 1;
//CHECK-NEXT:       _d_consts[0] += _t1;
//CHECK-NEXT:       double _t2 = 1 * consts[1];
//CHECK-NEXT:       _d_vars[1] += _t2;
//CHECK-NEXT:       double _t3 = vars[1] * 1;
//CHECK-NEXT:       _d_consts[1] += _t3;
//CHECK-NEXT:       double _t4 = 1 * consts[2];
//CHECK-NEXT:       _d_vars[2] += _t4;
//CHECK-NEXT:       double _t5 = vars[2] * 1;
//CHECK-NEXT:       _d_consts[2] += _t5;
//CHECK-NEXT:       _result[0UL] += _d_vars[0];
//CHECK-NEXT:       _result[1UL] += _d_vars[1];
//CHECK-NEXT:       _result[2UL] += _d_vars[2];
//CHECK-NEXT:       return;
//CHECK-NEXT:   }

int main () { // expected-no-diagnostics
  auto dsum = clad::differentiate(sum, 0);
  printf("%.2f\n", dsum.execute(11, 12, 13)); // CHECK-EXEC: 1.00

  auto dssum = clad::differentiate(sum_squares, 0);
  printf("%.2f\n", dssum.execute(11, 12, 13)); // CHECK-EXEC: 22.00
  auto dcdp = clad::differentiate(const_dot_product, 0);
  printf("%.2f\n", dcdp.execute(11, 12, 13)); // CHECK-EXEC: 1.00

  auto gradcdp = clad::gradient(const_dot_product);
  double result[3] = {};
  gradcdp.execute(11, 12, 13, result);
  printf("{%.2f, %.2f, %.2f}\n", result[0], result[1], result[2]); // CHECK-EXEC: {1.00, 2.00, 3.00}
  return 0;
}
