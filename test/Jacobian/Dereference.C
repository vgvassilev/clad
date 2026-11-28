// RUN: %cladclang %s -I%S/../../include -oDereference.out 2>&1 | %filecheck %s
// RUN: ./Dereference.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

void f_deref(double x, double* y, double* out) {
  *y += x;
  out[0] = *y * 2;
  *out *= x;
}

// CHECK: void f_deref_jac(double x, double *y, double *out, clad::matrix<double> *_d_vector_y, clad::matrix<double> *_d_vector_out) {
// CHECK-NEXT:     unsigned {{int|long|long long}} indepVarCount = _d_vector_y->rows() + _d_vector_out->rows() + {{1U|1UL|1ULL}};
// CHECK-NEXT:     clad::array<double> _d_vector_x = clad::one_hot_vector(indepVarCount, {{0U|0UL|0ULL}});
// CHECK-NEXT:     *_d_vector_y = clad::identity_matrix(_d_vector_y->rows(), indepVarCount, {{1U|1UL|1ULL}});
// CHECK-NEXT:     *_d_vector_out = clad::identity_matrix(_d_vector_out->rows(), indepVarCount, _d_vector_y->rows() + {{1U|1UL|1ULL}});
// CHECK-NEXT:     (*_d_vector_y)[0] += _d_vector_x;
// CHECK-NEXT:     *y += x;
// CHECK-NEXT:     (*_d_vector_out)[0] = (*_d_vector_y)[0] * 2 + *y * (clad::zero_vector(indepVarCount));
// CHECK-NEXT:     out[0] = *y * 2;
// CHECK-NEXT:     clad::array_ref<double> _t0 = (*_d_vector_out)[0];
// CHECK-NEXT:     _t0 = _t0 * x + *out * _d_vector_x;
// CHECK-NEXT:     *out *= x;
// CHECK-NEXT: }

void f_deref_offset(double x, double* y, double* out) {
  int i = 1;
  out[0] = *(y + i) * x;
  *(out + 1) = *(1 + y) + x;
}

// CHECK: void f_deref_offset_jac(double x, double *y, double *out, clad::matrix<double> *_d_vector_y, clad::matrix<double> *_d_vector_out) {
// CHECK-NEXT:     unsigned {{int|long|long long}} indepVarCount = _d_vector_y->rows() + _d_vector_out->rows() + {{1U|1UL|1ULL}};
// CHECK-NEXT:     clad::array<double> _d_vector_x = clad::one_hot_vector(indepVarCount, {{0U|0UL|0ULL}});
// CHECK-NEXT:     *_d_vector_y = clad::identity_matrix(_d_vector_y->rows(), indepVarCount, {{1U|1UL|1ULL}});
// CHECK-NEXT:     *_d_vector_out = clad::identity_matrix(_d_vector_out->rows(), indepVarCount, _d_vector_y->rows() + {{1U|1UL|1ULL}});
// CHECK-NEXT:     clad::array<int> _d_vector_i(clad::zero_vector(indepVarCount));
// CHECK-NEXT:     int i = 1;
// CHECK-NEXT:     (*_d_vector_out)[0] = (*_d_vector_y)[i] * x + *(y + i) * _d_vector_x;
// CHECK-NEXT:     out[0] = *(y + i) * x;
// CHECK-NEXT:     (*_d_vector_out)[1] = (*_d_vector_y)[1] + _d_vector_x;
// CHECK-NEXT:     *(out + 1) = *(1 + y) + x;
// CHECK-NEXT: }

int main() {
  double x = 1.5, y[2] = {2, 3}, out[2] = {0, 0};

  clad::matrix<double> dy(1, 3), dout(1, 3);
  clad::jacobian(f_deref).execute(x, y, out, &dy, &dout);
  printf("{%.2f, %.2f, %.2f}\n", dout[0][0], dout[0][1], dout[0][2]); // CHECK-EXEC: {10.00, 3.00, 0.00}
  printf("{%.2f, %.2f, %.2f}\n", dy[0][0], dy[0][1], dy[0][2]); // CHECK-EXEC: {1.00, 1.00, 0.00}

  y[0] = 2;
  clad::matrix<double> dy2(2, 5), dout2(2, 5);
  clad::jacobian(f_deref_offset).execute(x, y, out, &dy2, &dout2);
  printf("{%.2f, %.2f, %.2f, %.2f, %.2f}\n", dout2[0][0], dout2[0][1], dout2[0][2], dout2[0][3], dout2[0][4]); // CHECK-EXEC: {3.00, 0.00, 1.50, 0.00, 0.00}
  printf("{%.2f, %.2f, %.2f, %.2f, %.2f}\n", dout2[1][0], dout2[1][1], dout2[1][2], dout2[1][3], dout2[1][4]); // CHECK-EXEC: {1.00, 0.00, 1.00, 0.00, 0.00}
}
