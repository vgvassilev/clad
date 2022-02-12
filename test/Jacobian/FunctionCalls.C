// RUN: %cladclang %s -lm -I%S/../../include -oFunctionCalls.out 2>&1 | FileCheck %s
// RUN: ./FunctionCalls.out | FileCheck -check-prefix=CHECK-EXEC %s

//CHECK-NOT: {{.*error|warning|note:.*}}

#include <cmath>
#include "clad/Differentiator/Differentiator.h"

double outputs[4], results[4];

void fn1(double i, double j, double* output) {
  output[0] = std::pow(i, j);
  output[1] = std::pow(j, i);
}

// CHECK: void fn1_jac(double i, double j, double *output, double *jacobianMatrix) {
// CHECK-NEXT:     double _t0;
// CHECK-NEXT:     double _t1;
// CHECK-NEXT:     double _t2;
// CHECK-NEXT:     double _t3;
// CHECK-NEXT:     _t0 = i;
// CHECK-NEXT:     _t1 = j;
// CHECK-NEXT:     output[0] = std::pow(i, j);
// CHECK-NEXT:     _t2 = j;
// CHECK-NEXT:     _t3 = i;
// CHECK-NEXT:     output[1] = std::pow(j, i);
// CHECK-NEXT:     {
// CHECK-NEXT:         double _jac2 = 0.;
// CHECK-NEXT:         double _jac3 = 0.;
// CHECK-NEXT:         custom_derivatives::pow_pullback(_t2, _t3, 1, &_jac2, &_jac3);
// CHECK-NEXT:         double _r2 = _jac2;
// CHECK-NEXT:         jacobianMatrix[3UL] += _r2;
// CHECK-NEXT:         double _r3 = _jac3;
// CHECK-NEXT:         jacobianMatrix[2UL] += _r3;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         double _jac0 = 0.;
// CHECK-NEXT:         double _jac1 = 0.;
// CHECK-NEXT:         custom_derivatives::pow_pullback(_t0, _t1, 1, &_jac0, &_jac1);
// CHECK-NEXT:         double _r0 = _jac0;
// CHECK-NEXT:         jacobianMatrix[0UL] += _r0;
// CHECK-NEXT:         double _r1 = _jac1;
// CHECK-NEXT:         jacobianMatrix[1UL] += _r1;
// CHECK-NEXT:     }
// CHECK-NEXT: }

#define INIT(F) auto d_##F = clad::jacobian(F);

#define DERIVED_FN(F) d_##F

template <unsigned numOfOutputs, typename Fn, typename... Args>
void test(Fn derivedFn, Args... args) {
  unsigned numOfParameters = sizeof...(args);
  unsigned numOfResults = numOfOutputs * numOfParameters;
  for (unsigned i = 0; i < numOfOutputs; ++i)
    outputs[i] = 0;
  for (unsigned i = 0; i < numOfResults; ++i)
    results[i] = 0;
  derivedFn.execute(args..., outputs, results);
  printf("{");
  for (unsigned i = 0; i < numOfResults; ++i) {
    printf("%.2f", results[i]);
    if (i != numOfResults - 1)
      printf(", ");
  }
  printf("}\n");
}

int main() {
  INIT(fn1);
  
  test<2>(DERIVED_FN(fn1), 3, 5); // CHECK-EXEC: {405.00, 266.96, 201.18, 75.00}
}