// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -enable-va %s -I%S/../../include -o %t 2>&1 | %filecheck %s
// RUN: %t | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -enable-va -Xclang -plugin-arg-clad -Xclang -disable-tbr %s -I%S/../../include -o %t
// RUN: %t | %filecheck_exec %s
// XFAIL: valgrind

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

double weightedSum(const double p[], const double w[], int n) {
  double sum = 0;
  for (int i = 0; i < n; i++)
    sum += p[i] * w[i];
  return sum;
}
// CHECK-NOT: _d_i

int main() {
  auto hess = clad::hessian(weightedSum, "p[0:1],w[0:1]");
  double p[2] = {1, 2};
  double w[2] = {3, 4};
  double result[16] = {};
  hess.execute(p, w, 2, result);
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
      printf("%.2f%c", result[i * 4 + j], j == 3 ? '\n' : ' ');
  // CHECK-EXEC: 0.00 0.00 1.00 0.00
  // CHECK-EXEC: 0.00 0.00 0.00 1.00
  // CHECK-EXEC: 1.00 0.00 0.00 0.00
  // CHECK-EXEC: 0.00 1.00 0.00 0.00
}
