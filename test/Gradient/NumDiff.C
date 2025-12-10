// RUN: %cladnumdiffclang %s %S/NumDiffDefs.C -I%S/../../include -oNumDiff.out -Xclang -verify 2>&1 | FileCheck -check-prefix=CHECK %s
// RUN: ./NumDiff.out | %filecheck_exec %s
// XFAIL: valgrind

#include "clad/Differentiator/Differentiator.h"
#include "../TestUtils.h"

double single_arg(double x);

double f1(double x, double y) {
  return single_arg(x * y); // expected-warning {{function 'single_arg' was not differentiated because clad failed to differentiate it and no suitable overload was found in namespace 'custom_derivatives'}}
  // expected-note@11 {{falling back to numerical differentiation for 'single_arg'}}
}

//CHECK: void f1_grad(double x, double y, double *_d_x, double *_d_y) {
//CHECK-NEXT:     {
//CHECK-NEXT:         double _r0 = 0.;
//CHECK-NEXT:         _r0 += 1 * numerical_diff::forward_central_difference(single_arg, x * y, 0, 0, x * y);
//CHECK-NEXT:         *_d_x += _r0 * y;
//CHECK-NEXT:         *_d_y += x * _r0;
//CHECK-NEXT:     }
//CHECK-NEXT: }

double multi_arg(double x, double y);

double f2(double x, double y) {
  return multi_arg(x, y); // expected-warning {{function 'multi_arg' was not differentiated because clad failed to differentiate it and no suitable overload was found in namespace 'custom_derivatives'}}
  // expected-note@27 {{falling back to numerical differentiation for 'multi_arg'}}
}


//CHECK: void f2_grad(double x, double y, double *_d_x, double *_d_y) {
//CHECK-NEXT:     {
//CHECK-NEXT:         double _grad0[2] = {0};
//CHECK-NEXT:         numerical_diff::central_difference(multi_arg, _grad0, 0, x, y);
//CHECK-NEXT:         double _r0 = 1 * _grad0[0];
//CHECK-NEXT:         double _r1 = 1 * _grad0[1];
//CHECK-NEXT:         *_d_x += _r0;
//CHECK-NEXT:         *_d_y += _r1;
//CHECK-NEXT:     }
//CHECK-NEXT: }

void noNumDiff(double& x);

double f3(double x, double y) {
  noNumDiff(x);
  return x + y;
}

int main() {
  double dx = 0, dy = 0;
  INIT_GRADIENT(f1);
  TEST_GRADIENT(f1, /*numOfDerivativeArgs=*/2, 4, 5, &dx, &dy); // CHECK-EXEC: {10.00, 8.00}

  dx = 0; dy = 0;
  INIT_GRADIENT(f2);
  TEST_GRADIENT(f2, /*numOfDerivativeArgs=*/2, 2, 3, &dx, &dy); // CHECK-EXEC: {1.00, 1.00}

  dx = 0; dy = 0;
  INIT_GRADIENT(f3);
  TEST_GRADIENT(f3, /*numOfDerivativeArgs=*/2, 2, 3, &dx, &dy); // CHECK-EXEC: {1.00, 1.00}
  return 0;
}
