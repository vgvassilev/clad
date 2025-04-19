// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr %s -I%S/../../include -oPointers._clad_out_out 2>&1 | %filecheck %s
// RUN: ./Pointers._clad_out_out | %filecheck_exec %s
// RUN: %cladclang %s -I%S/../../include -oPointers._clad_out_out
// RUN: ./Pointers._clad_out_out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"

void nonMemFn(double i, double j, double* _clad_out_out) {
  _clad_out_out[0] = i;
  _clad_out_out[1] = j;
}

// CHECK: void nonMemFn_jac(double i, double j, double *_clad_out_out, clad::matrix<double> *_d_vector__clad_out_out) {
// CHECK-NEXT:     unsigned long indepVarCount = {{2U|2UL|2ULL}};
// CHECK-NEXT:     clad::array<double> _d_vector_i = clad::one_hot_vector(indepVarCount, {{0U|0UL|0ULL}});
// CHECK-NEXT:     clad::array<double> _d_vector_j = clad::one_hot_vector(indepVarCount, {{1U|1UL|1ULL}});
// CHECK-NEXT:     *_d_vector__clad_out_out = clad::identity_matrix(_d_vector__clad_out_out->rows(), indepVarCount, {{2U|2UL|2ULL}});
// CHECK-NEXT:     (*_d_vector__clad_out_out)[0] = _d_vector_i;
// CHECK-NEXT:     _clad_out_out[0] = i;
// CHECK-NEXT:     (*_d_vector__clad_out_out)[1] = _d_vector_j;
// CHECK-NEXT:     _clad_out_out[1] = j;
// CHECK-NEXT: }

#define NON_MEM_FN_TEST(var)\
var.execute(5, 7, _clad_out_out, &res);\
printf("{%.2f %.2f %.2f %.2f}\n", res[0][0], res[0][1],\
                                  res[1][0], res[1][1]);

int main() {
  auto nonMemFnPtr = &nonMemFn;
  auto nonMemFnPtrToPtr = &nonMemFnPtr;

  clad::matrix<double> res(2, 2);
  double _clad_out_out[2];
  auto d_nonMemFn = clad::jacobian(nonMemFn);
  auto d_nonMemFnPar = clad::jacobian((nonMemFn));
  auto d_nonMemFnPtr = clad::jacobian(nonMemFnPtr);
  auto d_nonMemFnPtrToPtr = clad::jacobian(*nonMemFnPtrToPtr);
  auto d_nonMemFnPtrToPtrPar = clad::jacobian((*(nonMemFnPtrToPtr)));

  NON_MEM_FN_TEST(d_nonMemFn); // CHECK-EXEC: {1.00 0.00 0.00 1.00}

  NON_MEM_FN_TEST(d_nonMemFnPar); // CHECK-EXEC: {1.00 0.00 0.00 1.00}

  NON_MEM_FN_TEST(d_nonMemFnPtr); // CHECK-EXEC: {1.00 0.00 0.00 1.00}

  NON_MEM_FN_TEST(d_nonMemFnPtrToPtr); // CHECK-EXEC: {1.00 0.00 0.00 1.00}

  NON_MEM_FN_TEST(d_nonMemFnPtrToPtrPar); // CHECK-EXEC: {1.00 0.00 0.00 1.00}
}

