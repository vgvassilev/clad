// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -enable-va %s -I%S/../../include -oNestedLoopsActivity.out -Xclang -verify 2>&1 | %filecheck %s
// RUN: ./NestedLoopsActivity.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

// expected-no-diagnostics

// Test complex nested loops with active and inactive variables.
// `x` is active, `y` is inactive.
double nested_loop_activity(double x, double y) {
  double sum_active = 0;
  double sum_inactive = 0;
  
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      if (i == j) {
        sum_active += x * x;
      } else {
        sum_inactive += y * y;
      }
    }
  }
  
  return sum_active + sum_inactive;
}

// CHECK: double nested_loop_activity_darg0(double x, double y) {
// CHECK: double _d_sum_active = 0;
// CHECK: for (int i = 0; i < 3; ++i) {
// CHECK: for (int j = 0; j < 3; ++j) {
// CHECK: if (i == j) {
// CHECK: _d_sum_active += _d_x * x + x * _d_x;
// CHECK: }
// CHECK-NOT: _d_sum_inactive
// CHECK: }
// CHECK: }
// CHECK: return _d_sum_active + 0.;

int main() {
  auto diff = clad::differentiate(nested_loop_activity, "x");
  double dx = diff.execute(2.0, 3.0);
  printf("dx = %.2f\n", dx);
  // CHECK-EXEC: dx = 12.00
  return 0;
}
