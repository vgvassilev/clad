// RUN: %cladclang %s -I%S/../../include -oLoopIncrementFix.out 2>&1 | %filecheck %s
// RUN: ./LoopIncrementFix.out | %filecheck_exec %s

// Test for segmentation fault fix when for loops have empty increment expressions
// This addresses issue #1436

#include "clad/Differentiator/Differentiator.h"
#include <iostream>

double fn_with_empty_increment(double u, double v) {
  double sum = 0;
  for (int i = 0; i != 1 ;) {  // Empty increment expression - should not crash
    sum += u + v;
    break;
  }
  return sum;
}

double fn_with_normal_increment(double u, double v) {
  double sum = 0;
  for (int i = 0; i < 2; i++) {  // Normal increment - should work as before
    sum += u + v;
  }
  return sum;
}

int main () {
  // Test empty increment loop (previously caused segfault)
  auto grad1 = clad::gradient(fn_with_empty_increment);
  double du1 = 0, dv1 = 0;
  grad1.execute(3.0, 4.0, &du1, &dv1);
  
  // Test normal increment loop (regression test)  
  auto grad2 = clad::gradient(fn_with_normal_increment);
  double du2 = 0, dv2 = 0;
  grad2.execute(3.0, 4.0, &du2, &dv2);
  
  printf("Empty increment test passed - derivatives: du1=%f, dv1=%f\n", du1, dv1); // CHECK-EXEC: Empty increment test passed - derivatives: du1=1.000000, dv1=1.000000
  printf("Normal increment test passed - derivatives: du2=%f, dv2=%f\n", du2, dv2); // CHECK-EXEC: Normal increment test passed - derivatives: du2=2.000000, dv2=2.000000
  
  return 0;
}