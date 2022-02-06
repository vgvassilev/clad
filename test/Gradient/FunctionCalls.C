// RUN: %cladnumdiffclang -lm -lstdc++ %s  -I%S/../../include -oFunctionCalls.out 2>&1 | FileCheck %s
// RUN: ./FunctionCalls.out | FileCheck -check-prefix=CHECK-EXEC %s

//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

namespace A {
  template <typename T> T constantFn(T i) { return 3; }
  // CHECK: void constantFn_grad(float i, clad::array_ref<float> _d_i) {
  // CHECK-NEXT:     int constantFn_return = 3;
  // CHECK-NEXT:     goto _label0;
  // CHECK-NEXT:   _label0:
  // CHECK-NEXT:     ;
  // CHECK-NEXT: }
} // namespace A

double constantFn(double i) {
  return 5;
}

double fn1(float i) {
  float res = A::constantFn(i);
  double a = res*i;
  return a;
}

// CHECK: void fn1_grad(float i, clad::array_ref<double> _d_i) {
// CHECK-NEXT:     float _t0;
// CHECK-NEXT:     float _d_res = 0;
// CHECK-NEXT:     float _t1;
// CHECK-NEXT:     float _t2;
// CHECK-NEXT:     double _d_a = 0;
// CHECK-NEXT:     _t0 = i;
// CHECK-NEXT:     float res = A::constantFn(_t0);
// CHECK-NEXT:     _t2 = res;
// CHECK-NEXT:     _t1 = i;
// CHECK-NEXT:     double a = _t2 * _t1;
// CHECK-NEXT:     double fn1_return = a;
// CHECK-NEXT:     goto _label0;
// CHECK-NEXT:   _label0:
// CHECK-NEXT:     _d_a += 1;
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r1 = _d_a * _t1;
// CHECK-NEXT:         _d_res += _r1;
// CHECK-NEXT:         double _r2 = _t2 * _d_a;
// CHECK-NEXT:         * _d_i += _r2;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         float _grad0 = 0.F;
// CHECK-NEXT:         constantFn_grad(_t0, &_grad0);
// CHECK-NEXT:         float _r0 = _d_res * _grad0;
// CHECK-NEXT:         * _d_i += _r0;
// CHECK-NEXT:     }
// CHECK-NEXT: }

#define INIT(F)\
  auto F##_grad = clad::gradient(F);

#define TEST1(F, ...)\
  result[0] = 0;\
  F##_grad.execute(__VA_ARGS__, &result[0]);\
  printf("{%.2f}\n", result[0]);

int main() {
  double result[2];
  INIT(fn1);

  TEST1(fn1, 11); // CHECK-EXEC: 3.00
}

