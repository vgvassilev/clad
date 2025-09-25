// RUN: %cladclang %s -I%S/../../include -fsyntax-only -Xclang -verify 2>&1 | %filecheck %s
// expected-no-diagnostics

// Test for segmentation fault fix when for loops have empty increment expressions
// This addresses issue #1436

#include "clad/Differentiator/Differentiator.h"

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

int main() {
    // Test that these compile without segfault - the main achievement of this fix
    auto grad1 = clad::gradient(fn_with_empty_increment);
    auto grad2 = clad::gradient(fn_with_normal_increment);
    return 0;
}

// CHECK: void fn_with_empty_increment_grad(double u, double v, double *_d_u, double *_d_v) {
// CHECK: void fn_with_normal_increment_grad(double u, double v, double *_d_u, double *_d_v) {