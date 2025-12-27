// RUN: %cladclang -std=c++17 -I%S/../../include %s -o %t
// RUN: %t | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

double f(double x) {
  return x * x;
}

int main() {
  auto f_diff = clad::differentiate(f);
  
  // Test the new operator() syntax
  double res_diff = f_diff(3.0); 
  printf("Diff result: %.2f\n", res_diff);
  // CHECK-EXEC: Diff result: 6.00

  // 2. Test gradient (Reverse Mode)
  auto f_grad = clad::gradient(f);
  double x = 4.0;
  double d_x = 0.0;
  
  // Test the new operator() syntax
  f_grad(x, &d_x);
  
  printf("Grad result: %.2f\n", d_x);
  // CHECK-EXEC: Grad result: 8.00

  return 0;
}