// RUN: %cladnumdiffclang %s -I%S/../../include -oNumDiff.out -Xclang -verify 2>&1 | FileCheck -check-prefix=CHECK %s
// RUN: ./NumDiff.out | %filecheck_exec %s
// RUN: %cladnumdiffclang -Xclang -plugin-arg-clad -Xclang -enable-tbr -Xclang -verify %s -I%S/../../include -oNumDiff.out
// RUN: ./NumDiff.out | %filecheck_exec %s
//CHECK-NOT: {{.*error|warning|note:.*}}
#include "clad/Differentiator/Differentiator.h"

double test_1(double x){
  return tanh(x); // expected-warning {{function 'tanh' was not differentiated because clad failed to differentiate it and no suitable overload was found in namespace 'custom_derivatives'}}
  // expected-note@9 {{falling back to numerical differentiation for 'tanh'}}
}

//CHECK: void test_1_grad(double x, double *_d_x) {
//CHECK-NEXT:     {
//CHECK-NEXT:         double _r0 = 0;
//CHECK-NEXT:         _r0 += 1 * numerical_diff::forward_central_difference(tanh, x, 0, 0, x);
//CHECK-NEXT:         *_d_x += _r0;
//CHECK-NEXT:     }
//CHECK-NEXT: }


double test_2(double x){
   return std::log10(x);// expected-warning {{function 'log10' was not differentiated because clad failed to differentiate it and no suitable overload was found in namespace 'custom_derivatives'}}
  // expected-note@23 {{falling back to numerical differentiation for 'log10'}}
}
//CHECK: double test_2_darg0(double x) {
//CHECK-NEXT:     double _d_x = 1;
//CHECK-NEXT:     return numerical_diff::forward_central_difference(std::log10, x, 0, 0, x) * _d_x;
//CHECK-NEXT: }


double test_3(double x) {
    if (x > 0) {
        double constant = 11.;
        return std::hypot(x, constant); // expected-warning {{function 'hypot' was not differentiated because clad failed to differentiate it and no suitable overload was found in namespace 'custom_derivatives'}}
  // expected-note@35 {{falling back to numerical differentiation for 'hypot'}}
    }
    return 0;
}
//CHECK: void test_3_grad(double x, double *_d_x) {
//CHECK-NEXT:     bool _cond0;
//CHECK-NEXT:     double _d_constant = 0;
//CHECK-NEXT:     double constant = 0;
//CHECK-NEXT:     {
//CHECK-NEXT:     _cond0 = x > 0;
//CHECK-NEXT:     if (_cond0) {
//CHECK-NEXT:         constant = 11.;
//CHECK-NEXT:         goto _label0;
//CHECK-NEXT:     }
//CHECK-NEXT:     }
//CHECK-NEXT:     if (_cond0) {
//CHECK-NEXT:       _label0:
//CHECK-NEXT:         {
//CHECK-NEXT:             double _r0 = 0;
//CHECK-NEXT:             double _r1 = 0;
//CHECK-NEXT:             double _grad0[2] = {0};
//CHECK-NEXT:             numerical_diff::central_difference(std::hypot, _grad0, 0, x, constant);
//CHECK-NEXT:             _r0 += 1 * _grad0[0];
//CHECK-NEXT:             _r1 += 1 * _grad0[1];
//CHECK-NEXT:             *_d_x += _r0;
//CHECK-NEXT:             _d_constant += _r1;
//CHECK-NEXT:         }
//CHECK-NEXT:     }
//CHECK-NEXT: }

int main(){
    clad::gradient(test_1);
    clad::differentiate(test_2, 0);
    auto dtest_3 = clad::gradient(test_3);
    double dx = 0;
    dtest_3.execute(5, &dx);
    printf("Result is = %f\n", dx); // CHECK-EXEC: Result is = 0.413803
}
