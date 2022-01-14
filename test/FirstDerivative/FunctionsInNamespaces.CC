// RUN: %cladclang %s -I%S/../../include -fsyntax-only -Xclang -verify 2>&1 | FileCheck %s
//CHECK-NOT: {{.*error:.*}}
//XFAIL:*
#include "clad/Differentiator/Differentiator.h"

namespace function_namespace10 {
  int func1(int x) {
    return x*x*x + x*x;
  }

  int func2(int x) {
    return x*x + x;
  }

  namespace function_namespace11 {
    int func3(int x, int y) {
      return x*x*x + y*y;
    }

    int func4(int x, int y) {
      return x*x + y;
    }
  }
}

namespace function_namespace2 {
  float func1(float x) {
    return x*x*x + x*x;
  }

  float func2(float x) {
    return x*x + x;
  }

  int func3(int x, int y) {
    return function_namespace10::function_namespace11::func4(x, y);
  }
}

int test_1(int x, int y) {
  return function_namespace2::func3(x, y);
}

// CHECK: int test_1_darg0(int x, int y) {
// CHECK-NEXT: function_namespace2::func3_darg0(int x, int y);
// CHECK-NEXT: }

// CHECK: int test_1_darg1(int x, int y) {
// CHECK-NEXT: function_namespace2::func3_darg1(int x, int y);
// CHECK-NEXT: }


int main () {
  clad::differentiate(test_1, 1); // expected-no-diagnostics

  return 0;
}
