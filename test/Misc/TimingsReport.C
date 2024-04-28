// RUN: %cladclang %s -I%S/../../include -oTimingsReport.out -ftime-report 2>&1 | %filecheck %s

#include "clad/Differentiator/Differentiator.h"
// CHECK-NOT: {{.*error|warning|note:.*}}
// CHECK: Timers for Clad Funcs

double nested1(double c){
  return c*3*c;
}

double nested2(double z){
  return 4*z*z;
}

double test1(double x, double y) {
  return 2*y*nested1(y) * 3 * x * nested1(x);
}

double test2(double a, double b) {
  return 3*a*a + b * nested2(a) + a * b;
}

int main() {
  auto d_fn_1 = clad::differentiate(test1, "x");
  double dp = -1, dq = -1;
  auto f_grad = clad::gradient(test2);
  f_grad.execute(3, 4, &dp, &dq);
  printf("Result is = %f\n", d_fn_1.execute(3,4));
  printf("Result is = %f %f\n", dp, dq);
  return 0;
}