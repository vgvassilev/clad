// RUN: %cladclang %s -I%S/../../include -oArrays.out 2>&1 | %filecheck %s
// RUN: ./Arrays.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -enable-tbr %s -I%S/../../include -oArrays.out
// RUN: ./Arrays.out | %filecheck_exec %s

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

//CHECK:   void const_dot_product_grad(double x, double y, double z, double *_d_x, double *_d_y, double *_d_z) {
//CHECK-NEXT:       double _d_vars[3] = {0};
//CHECK-NEXT:       double vars[3] = {x, y, z};
//CHECK-NEXT:       double _d_consts[3] = {0};
//CHECK-NEXT:       double consts[3] = {1, 2, 3};
//CHECK-NEXT:       {
//CHECK-NEXT:           _d_vars[0] += 1 * consts[0];
//CHECK-NEXT:           _d_consts[0] += vars[0] * 1;
//CHECK-NEXT:           _d_vars[1] += 1 * consts[1];
//CHECK-NEXT:           _d_consts[1] += vars[1] * 1;
//CHECK-NEXT:           _d_vars[2] += 1 * consts[2];
//CHECK-NEXT:           _d_consts[2] += vars[2] * 1;
//CHECK-NEXT:       }
//CHECK-NEXT:       {
//CHECK-NEXT:           *_d_x += _d_vars[0];
//CHECK-NEXT:           *_d_y += _d_vars[1];
//CHECK-NEXT:           *_d_z += _d_vars[2];
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

// CHECK: double const_matmul_sum_darg0(double a, double b, double c, double d) {
// CHECK-NEXT:     double _d_a = 1;
// CHECK-NEXT:     double _d_b = 0;
// CHECK-NEXT:     double _d_c = 0;
// CHECK-NEXT:     double _d_d = 0;
// CHECK-NEXT:     double _d_A[2][2] = {{[{][{]}}_d_a, _d_b}, {_d_c, _d_d}};
// CHECK-NEXT:     double A[2][2] = {{[{][{]}}a, b}, {c, d}};
// CHECK-NEXT:     double _d_B[2][2] = {{[{][{]}}0, 0}, {0, 0}};
// CHECK-NEXT:     double B[2][2] = {{[{][{]}}1, 2}, {3, 4}};
// CHECK-NEXT:     double _d_C[2][2] = {{[{][{]}}_d_A[0][0] * B[0][0] + A[0][0] * _d_B[0][0] + _d_A[0][1] * B[1][0] + A[0][1] * _d_B[1][0], _d_A[0][0] * B[0][1] + A[0][0] * _d_B[0][1] + _d_A[0][1] * B[1][1] + A[0][1] * _d_B[1][1]}, {_d_A[1][0] * B[0][0] + A[1][0] * _d_B[0][0] + _d_A[1][1] * B[1][0] + A[1][1] * _d_B[1][0], _d_A[1][0] * B[0][1] + A[1][0] * _d_B[0][1] + _d_A[1][1] * B[1][1] + A[1][1] * _d_B[1][1]}};
// CHECK-NEXT:     double C[2][2] = {{[{][{]}}A[0][0] * B[0][0] + A[0][1] * B[1][0], A[0][0] * B[0][1] + A[0][1] * B[1][1]}, {A[1][0] * B[0][0] + A[1][1] * B[1][0], A[1][0] * B[0][1] + A[1][1] * B[1][1]}};
// CHECK-NEXT:     return _d_C[0][0] + _d_C[0][1] + _d_C[1][0] + _d_C[1][1];
// CHECK-NEXT: }

// CHECK: void const_matmul_sum_grad(double a, double b, double c, double d, double *_d_a, double *_d_b, double *_d_c, double *_d_d) {
// CHECK-NEXT:     double _d_A[2][2] = {{[{][{]}}0., 0.}, {0., 0.}};
// CHECK-NEXT:     double A[2][2] = {{[{][{]}}a, b}, {c, d}};
// CHECK-NEXT:     double _d_B[2][2] = {{[{][{]}}0, 0}, {0, 0}};
// CHECK-NEXT:     double B[2][2] = {{[{][{]}}1, 2}, {3, 4}};
// CHECK-NEXT:     double _d_C[2][2] = {{[{][{]}}0., 0.}, {0., 0.}};
// CHECK-NEXT:     double C[2][2] = {{[{][{]}}A[0][0] * B[0][0] + A[0][1] * B[1][0], A[0][0] * B[0][1] + A[0][1] * B[1][1]}, {A[1][0] * B[0][0] + A[1][1] * B[1][0], A[1][0] * B[0][1] + A[1][1] * B[1][1]}};
// CHECK-NEXT:     {
// CHECK-NEXT:         _d_C[0][0] += 1;
// CHECK-NEXT:         _d_C[0][1] += 1;
// CHECK-NEXT:         _d_C[1][0] += 1;
// CHECK-NEXT:         _d_C[1][1] += 1;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         _d_A[0][0] += _d_C[0][0] * B[0][0];
// CHECK-NEXT:         _d_B[0][0] += A[0][0] * _d_C[0][0];
// CHECK-NEXT:         _d_A[0][1] += _d_C[0][0] * B[1][0];
// CHECK-NEXT:         _d_B[1][0] += A[0][1] * _d_C[0][0];
// CHECK-NEXT:         _d_A[0][0] += _d_C[0][1] * B[0][1];
// CHECK-NEXT:         _d_B[0][1] += A[0][0] * _d_C[0][1];
// CHECK-NEXT:         _d_A[0][1] += _d_C[0][1] * B[1][1];
// CHECK-NEXT:         _d_B[1][1] += A[0][1] * _d_C[0][1];
// CHECK-NEXT:         _d_A[1][0] += _d_C[1][0] * B[0][0];
// CHECK-NEXT:         _d_B[0][0] += A[1][0] * _d_C[1][0];
// CHECK-NEXT:         _d_A[1][1] += _d_C[1][0] * B[1][0];
// CHECK-NEXT:         _d_B[1][0] += A[1][1] * _d_C[1][0];
// CHECK-NEXT:         _d_A[1][0] += _d_C[1][1] * B[0][1];
// CHECK-NEXT:         _d_B[0][1] += A[1][0] * _d_C[1][1];
// CHECK-NEXT:         _d_A[1][1] += _d_C[1][1] * B[1][1];
// CHECK-NEXT:         _d_B[1][1] += A[1][1] * _d_C[1][1];
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         *_d_a += _d_A[0][0];
// CHECK-NEXT:         *_d_b += _d_A[0][1];
// CHECK-NEXT:         *_d_c += _d_A[1][0];
// CHECK-NEXT:         *_d_d += _d_A[1][1];
// CHECK-NEXT:     }
// CHECK-NEXT: }

void f25(double x, const double *y) { 
  const_cast<double &>(*y) = 3 * x; 
}

// CHECK: void f25_grad(double x, const double *y, double *_d_x, double *_d_y) {
// CHECK-NEXT:    const_cast<double &>(*y) = 3 * x;
// CHECK-NEXT:    {
// CHECK-NEXT:      double _r_d0 = *_d_y;
// CHECK-NEXT:      *_d_y = 0.;
// CHECK-NEXT:      *_d_x += 3 * _r_d0;
// CHECK-NEXT:    }
// CHECK-NEXT:  }

double arr[2] = {4.0, 5.0};

double f_inner(double* p) {
    return p[0] * p[1];
}

double f_outer(double x) {
    return f_inner(arr) * x;
}

// CHECK: double f_outer_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t0 = f_inner_pushforward(arr, (double{{ *}}[2]){0., 0.});
// CHECK-NEXT:     double &_t1 = _t0.value;
// CHECK-NEXT:     return _t0.pushforward * x + _t1 * _d_x;
// CHECK-NEXT: }

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

  auto dcms = clad::differentiate(const_matmul_sum, 0);
  printf("%.2f\n", dcms.execute(11, 12, 13, 14)); // CHECK-EXEC: 3.00

  auto grad = clad::gradient(const_matmul_sum);
  double result2[4] = {};
  grad.execute(
      11, 12, 13, 14, &result2[0], &result2[1], &result2[2], &result2[3]);
  printf("{%.2f, %.2f, %.2f, %.2f}\n", result2[0], result2[1], result2[2], result2[3]); // CHECK-EXEC: {3.00, 7.00, 3.00, 7.00}

  auto const_output_test = clad::gradient(f25);
  double const_output_test_result[2] = {0.0, 1.0};
  const double y[1] = {4.0};
  const_output_test.execute(3, y, &const_output_test_result[0],
                            &const_output_test_result[1]);
  printf("{%.2f, %.2f}\n", const_output_test_result[0],
         const_output_test_result[1]); // CHECK-EXEC: {3.00, 0.00}

    auto d_global = clad::differentiate(f_outer, "x");

    printf("%.2f\n", d_global.execute(5.0));
    // CHECK-EXEC: 20.00

  return 0;
}
