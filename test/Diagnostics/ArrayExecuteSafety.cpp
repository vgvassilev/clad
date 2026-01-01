// Regression test for array argument crash in execute()
// RUN: not %clang_cc1 -fplugin=%clad_plugin -x c++ -std=c++17 -I include %s 2>&1 | FileCheck %s

#include "clad/Differentiator/Differentiator.h"

double foo(double x, double y) { return x * y; }

void test_array_crash() {
  auto grad = clad::gradient(foo);
  double x = 1.0, y = 2.0;
  double grads[2] = {0, 0};

  // Passing an array used to segfault. Now it must trigger a static_assert.
  grad.execute(x, y, grads);
}

// CHECK: Clad Error: execution failed due to missing arguments. Automatic array unpacking is not supported in execute(). Please provide explicit pointers for all gradient variables.
