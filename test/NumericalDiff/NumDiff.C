// RUN: %cladnumdiffclang %s -I%S/../../include -oNumDiff.out -Xclang -verify 2>&1 | FileCheck -check-prefix=CHECK %s
// RUN: ./NumDiff.out | %filecheck_exec %s
// RUN: %cladnumdiffclang -Xclang -plugin-arg-clad -Xclang -enable-tbr -Xclang -verify %s -I%S/../../include -oNumDiff.out
// RUN: ./NumDiff.out | %filecheck_exec %s
#include "clad/Differentiator/Differentiator.h"

double test_1(double x){
  return std::tgamma(x); // expected-warning {{function 'tgamma' was not differentiated because clad failed to differentiate it and no suitable overload was found in namespace 'custom_derivatives'}}
  // expected-note@8 {{falling back to numerical differentiation for 'tgamma'}}
}

//CHECK: void test_1_grad(double x, double *_d_x) {
//CHECK-NEXT:     {
//CHECK-NEXT:         double _r0 = 0.;
//CHECK-NEXT:         _r0 += 1 * numerical_diff::forward_central_difference(std::tgamma, x, 0, 0, x);
//CHECK-NEXT:         *_d_x += _r0;
//CHECK-NEXT:     }
//CHECK-NEXT: }

int main(){
  auto df = clad::gradient(test_1);

  double x = 0.5, dx = 0;
  df.execute(x, &dx);
  printf("Result is:%f", dx); // CHECK-EXEC: Result is:-3.480231
  
}
