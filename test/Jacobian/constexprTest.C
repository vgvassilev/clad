// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr %s -I%S/../../include -std=c++14 -oconstexprTest.out 2>&1 | %filecheck %s
// RUN: ./constexprTest.out | %filecheck_exec %s
// RUN: %cladclang %s -I%S/../../include -std=c++14 -oconstexprTest.out
// RUN: ./constexprTest.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"

#include "../TestUtils.h"

  double result[3] = {0};
  double result1[3] = {0};
  clad::matrix<double> jacobian(3, 2);

constexpr void fn_mul(double i, double j, double *_clad_out_res) {
   _clad_out_res[0] = i*i;
   _clad_out_res[1] = j*j;
   _clad_out_res[2] = i*j;
}

// CHECK: constexpr void fn_mul_jac(double i, double j, double *_clad_out_res, clad::matrix<double> *_d_vector__clad_out_res) {
// CHECK-NEXT:     unsigned long indepVarCount = {{2U|2UL|2ULL}};
// CHECK-NEXT:     clad::array<double> _d_vector_i = clad::one_hot_vector(indepVarCount, {{0U|0UL|0ULL}});
// CHECK-NEXT:     clad::array<double> _d_vector_j = clad::one_hot_vector(indepVarCount, {{1U|1UL|1ULL}});
// CHECK-NEXT:     *_d_vector__clad_out_res = clad::identity_matrix(_d_vector__clad_out_res->rows(), indepVarCount, {{2U|2UL|2ULL}});
// CHECK-NEXT:     (*_d_vector__clad_out_res)[0] = _d_vector_i * i + i * _d_vector_i;
// CHECK-NEXT:     _clad_out_res[0] = i * i;
// CHECK-NEXT:     (*_d_vector__clad_out_res)[1] = _d_vector_j * j + j * _d_vector_j;
// CHECK-NEXT:     _clad_out_res[1] = j * j;
// CHECK-NEXT:     (*_d_vector__clad_out_res)[2] = _d_vector_i * j + i * _d_vector_j;
// CHECK-NEXT:     _clad_out_res[2] = i * j;
// CHECK-NEXT: }


constexpr void f_1(double x, double y, double z, double _clad_out_output[]) {
  _clad_out_output[0] = x * x * x;
  _clad_out_output[1] = x * y * x + y * x * x;
  _clad_out_output[2] = z * x * 10 - y * z;
}

// CHECK: constexpr void f_1_jac(double x, double y, double z, double _clad_out_output[], clad::matrix<double> *_d_vector__clad_out_output) {
// CHECK-NEXT:     unsigned long indepVarCount = {{3U|3UL|3ULL}};
// CHECK-NEXT:     clad::array<double> _d_vector_x = clad::one_hot_vector(indepVarCount, {{0U|0UL|0ULL}});
// CHECK-NEXT:     clad::array<double> _d_vector_y = clad::one_hot_vector(indepVarCount, {{1U|1UL|1ULL}});
// CHECK-NEXT:     clad::array<double> _d_vector_z = clad::one_hot_vector(indepVarCount, {{2U|2UL|2ULL}});
// CHECK-NEXT:     *_d_vector__clad_out_output = clad::identity_matrix(_d_vector__clad_out_output->rows(), indepVarCount, {{3U|3UL|3ULL}});
// CHECK-NEXT:     double _t0 = x * x;
// CHECK-NEXT:     (*_d_vector__clad_out_output)[0] = (_d_vector_x * x + x * _d_vector_x) * x + _t0 * _d_vector_x;
// CHECK-NEXT:     _clad_out_output[0] = _t0 * x;
// CHECK-NEXT:     double _t1 = x * y;
// CHECK-NEXT:     double _t2 = y * x;
// CHECK-NEXT:     (*_d_vector__clad_out_output)[1] = (_d_vector_x * y + x * _d_vector_y) * x + _t1 * _d_vector_x + (_d_vector_y * x + y * _d_vector_x) * x + _t2 * _d_vector_x;
// CHECK-NEXT:     _clad_out_output[1] = _t1 * x + _t2 * x;
// CHECK-NEXT:     double _t3 = z * x;
// CHECK-NEXT:     (*_d_vector__clad_out_output)[2] = (_d_vector_z * x + z * _d_vector_x) * 10 + _t3 * (clad::zero_vector(indepVarCount)) - (_d_vector_y * z + y * _d_vector_z); 
// CHECK-NEXT:     _clad_out_output[2] = _t3 * 10 - y * z;
// CHECK-NEXT: }

int main() {
    
    INIT_JACOBIAN(fn_mul);
    INIT_JACOBIAN(f_1);

    TEST_JACOBIAN(fn_mul, 2, 6, 3, 1, result, &jacobian); // CHECK-EXEC: {6.00, 0.00, 0.00, 2.00, 1.00, 3.00}
    TEST_JACOBIAN(f_1, 3, 9, 4, 5, 6, result1, &jacobian); // CHECK-EXEC: {48.00, 0.00, 0.00, 80.00, 32.00, 0.00, 60.00, -6.00, 35.00}
}

