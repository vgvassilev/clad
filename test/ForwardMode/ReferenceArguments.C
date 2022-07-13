// RUN: %cladclang %s -I%S/../../include -oReferenceArguments.out 
// RUN: ./ReferenceArguments.out | FileCheck -check-prefix=CHECK-EXEC %s
// CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

double fn1(double& i, double& j) {
  double res = i * i * j;
  return res;
}

// CHECK: double fn1_darg0(double &i, double &j) {
// CHECK-NEXT:     double _d_i = 1;
// CHECK-NEXT:     double _d_j = 0;
// CHECK-NEXT:     double _t0 = i * i;
// CHECK-NEXT:     double _d_res = (_d_i * i + i * _d_i) * j + _t0 * _d_j;
// CHECK-NEXT:     double res = _t0 * j;
// CHECK-NEXT:     return _d_res;
// CHECK-NEXT: }

#define INIT(fn, ...) auto d_##fn = clad::differentiate(fn, __VA_ARGS__);

#define TEST(fn, ...)                                                          \
  auto res = d_##fn.execute(__VA_ARGS__);                                      \
  printf("{%.2f}\n", res)

int main() {
    INIT(fn1, "i");
    
    double i = 3, j = 5;
    TEST(fn1, i, j);    // CHECK-EXEC: {30.00}
}
