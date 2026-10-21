// RUN: %cladnumdiffclang %s -I%S/../../include -oBuiltins.out -Xclang -verify 2>&1 | %filecheck %s
// RUN: ./Builtins.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"

#include <cstdio>

// A builtin cannot be passed as a pointer: the fallback differentiates the
// library function it stands for.
double fexp(double x) { return __builtin_exp(x); } // expected-warning 2{{attempted differentiation of function '__builtin_exp' without definition and no suitable overload was found in namespace 'custom_derivatives'}}
// expected-note@-1 2{{falling back to numerical differentiation for '__builtin_exp'}}

// CHECK: double fexp_darg0(double x) {
// CHECK:     return numerical_diff::forward_central_difference<{{.*}}>(exp, x, 0, 0, x) * _d_x;

// CHECK: void fexp_grad(double x, double *_d_x) {
// CHECK:     numerical_diff::forward_central_difference<{{.*}}>(exp, x, 0, 0, x)

// __builtin_powi stands for no library function, so it has nothing to fall
// back on.
double fpowi(double x) { return __builtin_powi(x, 2); } // expected-warning {{attempted differentiation of function '__builtin_powi' without definition and no suitable overload was found in namespace 'custom_derivatives'}}
// expected-note@-1 {{numerical differentiation is not viable for '__builtin_powi'; considering '__builtin_powi' as 0}}

// CHECK: double fpowi_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     return 0;

int main() {
  printf("%.4f\n", clad::differentiate(fexp, "x").execute(1)); // CHECK-EXEC: 2.7183
  double dx = 0;
  clad::gradient(fexp).execute(1, &dx);
  printf("%.4f\n", dx); // CHECK-EXEC: 2.7183
  printf("%.4f\n", clad::differentiate(fpowi, "x").execute(3)); // CHECK-EXEC: 0.0000
}
