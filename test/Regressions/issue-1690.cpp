// RUN: %clang -fplugin=%clad_plugin -std=c++17 %s -o %t
// RUN: %t | FileCheck %s

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

double test_func(double* x) { return x[0]; }

int main() {
  double arr[2] = {1.0, 2.0};
  double grads[2] = {0.0, 0.0}; // Buffer for the gradient output

  // Use clad::gradient (Reverse Mode), which supports array inputs/outputs
  auto d_test = clad::gradient(test_func);

  // This used to crash/error when passing raw arrays. 
  // Now it should implicitly decay arrays to pointers and execute successfully.
  d_test.execute(arr, grads);

  printf("Success\n");
  // CHECK: Success
  return 0;
}