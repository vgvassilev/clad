// RUN: %clangxx -fplugin=%clad_plugin -std=c++17 %s -o %t
// RUN: %t | FileCheck %s

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

double test_func(double* x) { return x[0]; }

int main() {
  double arr[2] = {1.0, 2.0};
  auto d_test = clad::differentiate(test_func, "x");

  // This used to crash/error. Now it should implicitly decay to pointer and run.
  d_test.execute(arr);

  printf("Success\n");
  // CHECK: Success
  return 0;
}