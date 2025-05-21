// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr -Xclang -plugin-arg-clad -Xclang -enable-sp %s -I%S/../../include -std=c++14 -oSparsity.out 2>&1 | %filecheck %s
// RUN: ./Sparsity.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -enable-sp %s -I%S/../../include -std=c++14 -oSparsity.out
// RUN: ./Sparsity.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"

void f1(double a, double b, double c, double _clad_out_output[]) {
  _clad_out_output[0] = a;
  _clad_out_output[1] = 2*a + 3*b;
  _clad_out_output[2] = 4*a + 5*b + 6*c;
}

// CHECK: void f1_jac(double a, double b, double c, double _clad_out_output[], clad::matrix<double> *_d_vector__clad_out_output) {
// CHECK-NEXT:     unsigned long indepVarCount = 3UL;
// CHECK-NEXT:     clad::array<double> _d_vector_a = clad::one_hot_vector(indepVarCount, 0UL);
// CHECK-NEXT:     clad::array<double> _d_vector_b = clad::one_hot_vector(indepVarCount, 1UL);
// CHECK-NEXT:     clad::array<double> _d_vector_c = clad::one_hot_vector(indepVarCount, 2UL);
// CHECK-NEXT:     *_d_vector__clad_out_output = clad::identity_matrix(_d_vector__clad_out_output->rows(), indepVarCount, 3UL);
// CHECK-NEXT:     (*_d_vector__clad_out_output)[0] = _d_vector_a;
// CHECK-NEXT:     _clad_out_output[0] = a;
// CHECK-NEXT:     (*_d_vector__clad_out_output)[1] = (clad::zero_vector(indepVarCount)) * a + 2 * _d_vector_a + (clad::zero_vector(indepVarCount)) * b + 3 * _d_vector_b;
// CHECK-NEXT:     _clad_out_output[1] = 2 * a + 3 * b;
// CHECK-NEXT:     (*_d_vector__clad_out_output)[2] = (clad::zero_vector(indepVarCount)) * a + 4 * _d_vector_a + (clad::zero_vector(indepVarCount)) * b + 5 * _d_vector_b + (clad::zero_vector(indepVarCount)) * c + 6 * _d_vector_c;
// CHECK-NEXT:     _clad_out_output[2] = 4 * a + 5 * b + 6 * c;
// CHECK-NEXT: }

// CHECK: void f1_sparse_jac(double a, double b, double c, double _clad_out_output[], clad::matrix<double> *_d_vector__clad_out_output, clad::sparsity_pattern<double> *sparsity_patern) {
// CHECK-NEXT:     f1_jac(a, b, c, _clad_out_output, _d_vector__clad_out_output);
// CHECK-NEXT:     int dependency_set[6] = {0, 3, 4, 6, 7, 8};
// CHECK-NEXT:     sparsity_patern->set_col_idx({0, 0, 1, 0, 1, 2});
// CHECK-NEXT:     sparsity_patern->set_row_idx({0, 1, 3, 6});
// CHECK-NEXT:     for (int i = 0; i < 6; ++i)
// CHECK-NEXT:         (*sparsity_patern)[i] = (*_d_vector__clad_out_output)[dependency_set[i] / 3][dependency_set[i] % 3];
// CHECK-NEXT: }


int main(){
    clad::matrix<double> jacobian(3, 3);
    double outputarr[9];
    clad::sparsity_pattern<double> pattern;

    auto df1 = clad::jacobian(f1);
    df1.execute(1, 2, 3, outputarr, &jacobian, &pattern);
    printf("nnz: %.2f, %.2f, %.2f, %.2f, %.2f, %.2f", pattern[0], pattern[1], pattern[2], pattern[3], pattern[4], pattern[5]); // CHECK-EXEC: nnz: 1.00, 2.00, 3.00, 4.00, 5.00, 6.00
}
