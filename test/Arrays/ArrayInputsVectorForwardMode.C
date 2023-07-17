// RUN: %cladclang %s -I%S/../../include -oArrayInputsVectorForwardMode.out 2>&1 | FileCheck %s
// RUN: ./ArrayInputsVectorForwardMode.out | FileCheck -check-prefix=CHECK-EXEC %s

// CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

double multiply(double *arr) {
  return arr[0] * arr[1];
}

// CHECK: void multiply_dvec(double *arr, clad::array_ref<double> _d_arr) {
// CHECK-NEXT:   unsigned long indepVarCount = _d_arr.size();
// CHECK-NEXT:   clad::matrix<double> _d_vector_arr = clad::identity_matrix(_d_arr.size(), indepVarCount, 0UL);
// CHECK-NEXT:   {
// CHECK-NEXT:     clad::array<double> _d_vector_return(clad::array<double>(indepVarCount, (_d_vector_arr[0]) * arr[1] + arr[0] * (_d_vector_arr[1])));
// CHECK-NEXT:     _d_arr = _d_vector_return.slice(0UL, _d_arr.size());
// CHECK-NEXT:     return;
// CHECK-NEXT:   }
// CHECK-NEXT: }

double divide(double *arr) {
  return arr[0] / arr[1];
}

// CHECK: void divide_dvec(double *arr, clad::array_ref<double> _d_arr) {
// CHECK-NEXT:   unsigned long indepVarCount = _d_arr.size();
// CHECK-NEXT:   clad::matrix<double> _d_vector_arr = clad::identity_matrix(_d_arr.size(), indepVarCount, 0UL);
// CHECK-NEXT:   {
// CHECK-NEXT:     clad::array<double> _d_vector_return(clad::array<double>(indepVarCount, ((_d_vector_arr[0]) * arr[1] - arr[0] * (_d_vector_arr[1])) / (arr[1] * arr[1])));
// CHECK-NEXT:     _d_arr = _d_vector_return.slice(0UL, _d_arr.size());
// CHECK-NEXT:     return;
// CHECK-NEXT:   }
// CHECK-NEXT: }

double addArr(double *arr, int n) {
  double ret = 0;
  for (int i = 0; i < n; i++) {
    ret += arr[i];
  }
  return ret;
}

// CHECK: void addArr_dvec_0(double *arr, int n, clad::array_ref<double> _d_arr) {
// CHECK-NEXT:   unsigned long indepVarCount = _d_arr.size();
// CHECK-NEXT:   clad::matrix<double> _d_vector_arr = clad::identity_matrix(_d_arr.size(), indepVarCount, 0UL);
// CHECK-NEXT:   clad::array<int> _d_vector_n = clad::zero_vector(indepVarCount);
// CHECK-NEXT:   clad::array<double> _d_vector_ret(clad::array<double>(indepVarCount, 0));
// CHECK-NEXT:   double ret = 0;
// CHECK-NEXT:   {
// CHECK-NEXT:     clad::array<int> _d_vector_i(clad::array<int>(indepVarCount, 0));
// CHECK-NEXT:     for (int i = 0; i < n; i++) {
// CHECK-NEXT:       _d_vector_ret += _d_vector_arr[i];
// CHECK-NEXT:       ret += arr[i];
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   {
// CHECK-NEXT:     clad::array<double> _d_vector_return(clad::array<double>(indepVarCount, _d_vector_ret));
// CHECK-NEXT:     _d_arr = _d_vector_return.slice(0UL, _d_arr.size());
// CHECK-NEXT:     return;
// CHECK-NEXT:   }
// CHECK-NEXT: }

int main() {
  // multiply
  double a1[] = {1, 2, 3};
  double da1[3] = {0};
  clad::array_ref<double> da1_ref(da1, 3);
  auto multiply_darr = clad::differentiate<clad::opts::vector_mode>(multiply, "arr");
  multiply_darr.execute(a1, da1_ref);
  printf("Result = {%.2f, %.2f, %.2f}\n", da1[0], da1[1], da1[2]); // CHECK-EXEC: Result = {2.00, 1.00, 0.00}

  // divide
  double a2[] = {1, 2, 3};
  double da2[3] = {0};
  clad::array_ref<double> da2_ref(da2, 3);
  auto divide_darr = clad::differentiate<clad::opts::vector_mode>(divide, "arr");
  divide_darr.execute(a2, da2_ref);
  printf("Result = {%.2f, %.2f, %.2f}\n", da2[0], da2[1], da2[2]); // CHECK-EXEC: Result = {0.50, -0.25, 0.00}

  // addArr
  double a3[] = {1, 2, 3};
  double da3[3] = {0};
  clad::array_ref<double> da3_ref(da3, 3);
  auto addArr_darr = clad::differentiate<clad::opts::vector_mode>(addArr, "arr");
  addArr_darr.execute(a3, 3, da3_ref);
  printf("Result = {%.2f, %.2f, %.2f}\n", da3[0], da3[1], da3[2]); // CHECK-EXEC: Result = {1.00, 1.00, 1.00}

  return 0;
}