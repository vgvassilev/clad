// RUN: %cladnumdiffclang %s -I%S/../../include -oNumDiff.out -Xclang -verify 2>&1 | FileCheck -check-prefix=CHECK %s
// RUN: ./NumDiff.out | %filecheck_exec %s
// RUN: %cladnumdiffclang -Xclang -plugin-arg-clad -Xclang -enable-tbr -Xclang -verify %s -I%S/../../include -oNumDiff.out
// RUN: ./NumDiff.out | %filecheck_exec %s
#include "clad/Differentiator/Differentiator.h"

double test_1(double x){
  return std::tgamma(x); // expected-warning {{attempted differentiation of function 'tgamma' without definition and no suitable overload was found in namespace 'custom_derivatives'}}
  // expected-note@8 {{falling back to numerical differentiation for 'tgamma'}}
}

//CHECK: void test_1_grad(double x, double *_d_x) {
//CHECK-NEXT:     {
//CHECK-NEXT:         double _r0 = 0.;
//CHECK-NEXT:         _r0 += 1 * numerical_diff::forward_central_difference(std::tgamma, x, 0, 0, x);
//CHECK-NEXT:         *_d_x += _r0;
//CHECK-NEXT:     }
//CHECK-NEXT: }

// In forward mode the primal call is rebuilt and returned next to the
// numerical-diff call, which must clone the callee and arguments instead of
// sharing them with the rebuilt call.
double test_2(double x){
  return x * std::tgamma(x); // expected-warning {{attempted differentiation of function 'tgamma' without definition and no suitable overload was found in namespace 'custom_derivatives'}}
  // expected-note@-1 {{falling back to numerical differentiation for 'tgamma'}}
}

//CHECK: double test_2_darg0(double x) {
//CHECK-NEXT:     double _d_x = 1;
//CHECK-NEXT:     double _t0 = std::tgamma(x);
//CHECK-NEXT:     return _d_x * _t0 + x * (numerical_diff::forward_central_difference(std::tgamma, x, 0, 0, x) * _d_x);
//CHECK-NEXT: }

int main(){
  auto df = clad::gradient(test_1);

  double x = 0.5, dx = 0;
  df.execute(x, &dx);
  printf("Result is:%f\n", dx); // CHECK-EXEC: Result is:-3.480231

  auto df2 = clad::differentiate(test_2, "x");
  printf("Result is:%f\n", df2.execute(3.0)); // CHECK-EXEC: Result is:7.536706
}
