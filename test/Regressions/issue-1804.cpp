// RUN: %cladclang -c -std=c++17 -I%S/../../include %s

#include "clad/Differentiator/Differentiator.h"

double f_inner(double *params, double const *obs) {
   double arg = (obs[0] - params[1]) / params[2];
   return arg * arg;
}

double f_outer(double *params, double const *obs) {
   return f_inner(params, obs) + params[0];
}

void check() {
   auto hess = clad::hessian(f_outer, "params[0:2]");
}

// CHECK: f_inner_pushforward_pullback(params, obs, (double[3]){1., 0., 0.}, nullptr, _d_t0, _d_params, (double[3]){0., 0., 0.}, nullptr);