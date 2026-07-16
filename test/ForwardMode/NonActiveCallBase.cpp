// RUN: %cladclang %s -I%S/../../include -oNonActiveCallBase.out | %filecheck %s
// RUN: ./NonActiveCallBase.out | %filecheck_exec %s

// Regression test: forward mode over a member/operator call whose base object
// has no tangent, i.e. does not depend on the differentiation variable. Such a
// base has no pushforward to call; clad used to take the address of its absent
// tangent and fail with "cannot take the address of an rvalue of type 'void'".

#include "clad/Differentiator/Differentiator.h"

#include <cstdio>

struct Arr {
  double d[3];
  double operator()(int i) const { return d[i]; }
};

namespace clad {
namespace custom_derivatives {
namespace class_functions {
clad::ValueAndPushforward<double, double>
operator_call_pushforward(const Arr* a, int i, const Arr* d_a, int /*d_i*/) {
  return {(*a)(i), (*d_a)(i)};
}
} // namespace class_functions
} // namespace custom_derivatives
} // namespace clad

// g is read but never differentiated with respect to, so it has no tangent.
static Arr g{{2, 3, 4}};

double reads_nonactive(double x) {
  double s = 0;
  for (int i = 0; i < 3; ++i)
    s += x * g(i);
  return s;
}

// CHECK: double reads_nonactive_darg0(double x) {

int main() {
  auto dx = clad::differentiate(reads_nonactive, "x");
  // d/dx sum_i x * g(i) = sum_i g(i) = 9
  printf("%.2f\n", dx.execute(1.5)); // CHECK-EXEC: 9.00
}
