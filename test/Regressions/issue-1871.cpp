// RUN: %cladclang -D_USE_MATH_DEFINES -std=c++17 -I%S/../../include %s -o %t 2>&1
// RUN: %t | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

// The inner function accesses obs[0]; its pushforward will receive
// _d_obs as a pointer.  When the outer derivative represents the
// inactive const-pointer tangent as nullptr, _d_obs[0] dereferences
// nullptr without the null-tangent guard.
double inner_func(const double* obs, double x) {
  return obs[0] * x * x * x;
}

// The outer function forwards the inactive const double* to inner_func.
// clad::differentiate(outer_func, "x") generates a pushforward for
// inner_func where the tangent of obs is nullptr.
double outer_func(const double* obs, double x) {
  return inner_func(obs, x);
}

int main() {
  auto df = clad::differentiate(outer_func, "x");
  double obs[] = {1.0};
  printf("%.2f\n", df.execute(obs, 1.0)); // CHECK-EXEC: 3.00
}
