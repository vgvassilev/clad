// RUN: %cladclang -std=c++17 -O0 -I%S/../../include/ %s -o %t 2>&1 | %filecheck %s
// RUN: %t | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>
#include <vector>

struct CustomObj {
  double v;
  CustomObj() : v(0.0) {} 
  CustomObj(double temp) {
    v = temp * 2.0; 
  }
};

double func1(double x, int n) {
  if (n <= 0) {
    CustomObj obj(x); 
    return obj.v;
  }
  return x*2.0 + func1(x, n - 1);
}

double func(double* A, int n, std::vector<double>& v) {
  if (n <= 1) {
    return A[0] * v[0]; 
  }
  
  double res = 0;
  for (int i = 0; i < n; i++) {
    v[0] += A[i]; 
    res += func(A, n - 1, v); 
  }
  return res;
}

// CHECK: void func_grad_0_2(
// CHECK: clad::tape<clad::restore_tracker> _tracker0 = {};
// CHECK: auto _rev0 = [&] {
// CHECK: clad::back(_tracker0).restore();

int main() {
  double A[] = {1.0, 2.0, 3.0};
  std::vector<double> v = {1.0};
  auto d_fn = clad::gradient(func, "A, v");
  
  double d_A[3] = {0, 0, 0};
  std::vector<double> d_v = {0.0};
  d_fn.execute(A, 3, v, d_A, &d_v);
  printf("d_A = {%.2f, %.2f, %.2f}\n", d_A[0], d_A[1], d_A[2]);
  // CHECK-EXEC: d_A = {74.00, 13.00, 2.00}
  printf("d_v = {%.2f}\n", d_v[0]);
  // CHECK-EXEC: d_v = {6.00}
  
  auto d_func1 = clad::gradient(func1, "x");
  double d_x = 0;
  d_func1.execute(2.0, 3, &d_x);
  printf("d_x = %.2f\n", d_x); 
  // CHECK-EXEC: d_x = 8.00
  
  return 0;
}