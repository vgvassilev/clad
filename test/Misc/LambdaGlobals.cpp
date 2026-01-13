// RUN: %cladclang -I%S/../../include -I%S/../../build/include -c %s 2>&1 | FileCheck %s

#include "clad/Differentiator/Differentiator.h"

void f2(double i, double j) {
  auto L = []() {
    {
      double a = 1;
      (void)a;
    }
  };
  L();
}

int main() {
  (void)clad::gradient(f2);
}

// Bug #1701: lambda-local `a` must NOT be emitted into the outer derivative fn.
//
// The derivative storage for the lambda should live in the pullback helper,
// not inside f2_grad.

// CHECK-LABEL: {{.*}}operator_call_pullback{{.*}}
// CHECK: double _d_a = 0.
// CHECK: double a = 0.
// CHECK: a = 1

// And f2_grad should NOT contain the zero-initialized 'a' / '_d_a'.
// (It will still contain 'double a = 1;' inside the original lambda.)
//
// CHECK-LABEL: void f2_grad(
// CHECK-NOT: double _d_a = 0.
// CHECK-NOT: double a = 0.
// CHECK: auto L = []() {
// CHECK: double a = 1
