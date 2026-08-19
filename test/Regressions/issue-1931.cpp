// RUN: %cladclang %s -I%S/../../include -o %t
// RUN: %t | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"

template <typename T>
double helper(int n, T x) {
  double result = 0.0;
  for (int i = 0; i < n; ++i)
    result += x[i] * x[i];
  return result;
}

double fn(double* params) {
  double sum = 0.0;
  #pragma clad checkpoint loop
  for (int i = 0; i < 3; i++) {
    double arr[]{params[0], params[1]};
    sum += helper(2, arr);
  }
  return sum;
}

int main() {
  auto grad = clad::gradient(fn);
  double params[] = {3.0, 4.0};
  double d_params[] = {0.0, 0.0};
  grad.execute(params, d_params);
  printf("{%.2f, %.2f}\n", d_params[0], d_params[1]);
  // CHECK-EXEC: {18.00, 24.00}
  return 0;
}
