// RUN: %cladclang %s -I%S/../../include -o %t 2>&1 | %filecheck %s
// RUN: %t | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"

#include <cstdio>

struct Result {
  double value;
  double* storage;
};

Result make_result(double x) { return {x * x, nullptr}; }

namespace clad::custom_derivatives {

clad::ValueAndAdjoint<Result, Result> make_result_reverse_forw(double x,
                                                               double /*d_x*/) {
  return {make_result(x), {0, nullptr}};
}

void make_result_pullback(double x, Result d_result, double* d_x) {
  *d_x += 2 * x * d_result.value;
}

} // namespace clad::custom_derivatives

Result helper(double x) { return make_result(x); }

double loss(double x) {
  auto result = helper(x);
  return result.value;
}

// CHECK: clad::ValueAndAdjoint<Result, Result> helper_reverse_forw(
// CHECK: make_result_reverse_forw(
// CHECK: return {{.*}}.value{{.*}}.adjoint{{.*}};

int main() {
  auto gradient = clad::gradient(loss);
  double dx = 0;
  gradient.execute(3, &dx);
  std::printf("%.1f\n", dx);
  // CHECK-EXEC: 6.0
}
