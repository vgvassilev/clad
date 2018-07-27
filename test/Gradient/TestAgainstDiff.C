// RUN: %cladclang %s -I%S/../../include -oTestAgainstDiff.out 2>&1 | FileCheck %s
// RUN: ./TestAgainstDiff.out | FileCheck -check-prefix=CHECK-EXEC %s

//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

double f(double x, double y) {
  return (x - 1) * (x - 1) + 100 * (y - x * x) * (y - x * x);
}
void f_grad(double x, double y, double * _result);

void f_grad_old(double x, double y, double * _result) {
  auto dx = clad::differentiate(f, 0);
  auto dy = clad::differentiate(f, 1);

  _result[0] = dx.execute(x, y);
  _result[1] = dy.execute(x, y);
}

int main() {
  clad::gradient(f);
  
  auto test = [&] (double x, double y) { // expected-no-diagnostics
    double result_old[2] = {};
    double result_new[2] = {};

    f_grad_old(x, y, result_old);
    f_grad(x, y, result_new);

    for (int i = 0; i < 2; i++)
      if (result_old[i] != result_new[i])
        return false;
    return true;
  };

  printf("Equal? %s\n", test(1, 1) ? "true" : "false"); // CHECK-EXEC: Equal? true
  printf("Equal? %s\n", test(3, 3) ? "true" : "false"); // CHECK-EXEC: Equal? true
  printf("Equal? %s\n", test(5, 5) ? "true" : "false"); // CHECK-EXEC: Equal? true
  
}


