// RUN: %cladclang -std=c++17 -I%S/../../include %s -o %t
// RUN: %t | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"

double f_inner(double *params, double const *obs) {
   double arg = (obs[0] - params[1]) / params[2];
   return arg * arg;
}

double f_outer(double *params, double const *obs) {
   return f_inner(params, obs) + params[0];
}

int main() {
   double params[] = {0.5, -1, 2};
   double obs[] = {-9.9};
   double h[9] = {};

   auto hess = clad::hessian(f_outer, "params[0:2]");
   hess.execute(params, obs, h);

   printf("H_00 : %.2f\n", h[0]); // CHECK-EXEC: H_00 : 0.00
   printf("H_11 : %.2f\n", h[4]); // CHECK-EXEC: H_11 : 0.50
   printf("H_12 : %.2f\n", h[5]); // CHECK-EXEC: H_12 : -4.45
   printf("H_21 : %.2f\n", h[7]); // CHECK-EXEC: H_21 : -4.45

   return 0;
}