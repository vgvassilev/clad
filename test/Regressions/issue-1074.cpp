// RUN: %cladclang -std=c++17 -I%S/../../include -c %s 2>&1 | %filecheck %s

#include "clad/Differentiator/Differentiator.h"

double test_zero_init_coverage(double x) {
  double y = 0; 

  double res = 0;
  for(int i = 0; i < 3; ++i) {
      double z = 0;
      int k = 0; 
      z = x * i;
      k = i;
      res += z + k;
  }
  return y + res;
}
auto test_grad = clad::gradient(test_zero_init_coverage);

// CHECK: void test_zero_init_coverage_grad(double x, double *_d_x) {
// CHECK: double _d_z = 0.;
// CHECK: double z = 0.;
// CHECK: int k = 0;
// CHECK: double _d_y = 0.;
// CHECK: double y = 0.;