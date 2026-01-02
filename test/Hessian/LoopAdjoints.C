// Compile time test for Hessian of weightedSum, kept separate from runtime tests (Hessians.C)
// RUN: %cladclang -std=c++17 -Xclang -add-plugin -Xclang clad -Xclang -plugin-arg-clad -Xclang \
// RUN: -fdump-derived-fn %s -I%S/../../include -c -o /dev/null 2>&1 | %filecheck %s

#include "clad/Differentiator/Differentiator.h"

double weightedSum(double* arr, const double* weights) {
  double sum = 0;
  for(int i=0; i<10; ++i) {
    sum += arr[i] * weights[i];
  }
  return sum;
}

// Instantiate Hessian to trigger derivation
auto h_weightedSum = clad::hessian(weightedSum, "arr[0:10]");

// Verify that the loop index adjoint '_d_i' is NOT created.
// CHECK: void weightedSum_hessian
// CHECK-NOT: int _d_i
// CHECK: }
