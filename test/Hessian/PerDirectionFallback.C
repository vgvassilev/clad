// RUN: %cladclang %s -I%S/../../include -oPerDirectionFallback.out 2>&1 | %filecheck %s
// RUN: ./PerDirectionFallback.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"

// The hessian-vector-product scheme needs one tangent per parameter, passed
// by position and reseeded between directions. A parameter the pushforward's
// signature filter skips (an enum) breaks the positions; a reference tangent
// cannot be reseeded or zero-initialized; a custom pushforward's author never
// promised the positional call the wrapper would make. The planner recognizes
// each and schedules per-direction derivatives instead -- for that function
// only, as `plain` below shows.

enum Kind { Linear, Quadratic };

double withEnum(double x, double y, Kind k) {
  if (k == Quadratic)
    return x * x * y;
  return x * y;
}

// CHECK: void withEnum_hessian_0_1(double x, double y, Kind k, double *hessianMatrix) {
// CHECK-NEXT: withEnum_darg0_grad_0_1(x, y, k, hessianMatrix + {{0U|0UL|0ULL}}, hessianMatrix + {{1U|1UL|1ULL}});
// CHECK-NEXT: withEnum_darg1_grad_0_1(x, y, k, hessianMatrix + {{2U|2UL|2ULL}}, hessianMatrix + {{3U|3UL|3ULL}});
// CHECK-NEXT: }

double withRef(double x, double y, double& scale) {
  return x * x * y * scale;
}

// CHECK: void withRef_hessian_0_1(double x, double y, double &scale, double *hessianMatrix) {
// CHECK-NEXT: withRef_darg0_grad_0_1(x, y, scale, hessianMatrix + {{0U|0UL|0ULL}}, hessianMatrix + {{1U|1UL|1ULL}});
// CHECK-NEXT: withRef_darg1_grad_0_1(x, y, scale, hessianMatrix + {{2U|2UL|2ULL}}, hessianMatrix + {{3U|3UL|3ULL}});
// CHECK-NEXT: }

double withCustomPushforward(double x, double y) { return x * y * y; }

namespace clad {
namespace custom_derivatives {
clad::ValueAndPushforward<double, double>
withCustomPushforward_pushforward(double x, double y, double _d_x,
                                  double _d_y) {
  return {x * y * y, y * y * _d_x + 2 * x * y * _d_y};
}
} // namespace custom_derivatives
} // namespace clad

// CHECK: void withCustomPushforward_hessian(double x, double y, double *hessianMatrix) {
// CHECK-NEXT: withCustomPushforward_darg0_grad(x, y, hessianMatrix + {{0U|0UL|0ULL}}, hessianMatrix + {{1U|1UL|1ULL}});
// CHECK-NEXT: withCustomPushforward_darg1_grad(x, y, hessianMatrix + {{2U|2UL|2ULL}}, hessianMatrix + {{3U|3UL|3ULL}});
// CHECK-NEXT: }

// The control: nothing above disqualifies this one, and in the same TU it
// takes the hessian-vector-product scheme.
double plain(double a, double b) { return a * a * b; }

// CHECK: void plain_hessian(double a, double b, double *hessianMatrix) {
// CHECK-NEXT: clad::ValueAndPushforward<double, double> _d_y{0., 0.};
// CHECK-NEXT: _d_y.pushforward = 1.;
// CHECK-NEXT: double _d_a(0.);
// CHECK-NEXT: double _d_b(0.);
// CHECK-NEXT: _d_a = 1.;
// CHECK-NEXT: plain_pushforward_pullback(a, b, _d_a, _d_b, _d_y, hessianMatrix + {{0U|0UL|0ULL}}, hessianMatrix + {{1U|1UL|1ULL}});
// CHECK-NEXT: _d_a = 0.;
// CHECK-NEXT: _d_b = 1.;
// CHECK-NEXT: plain_pushforward_pullback(a, b, _d_a, _d_b, _d_y, hessianMatrix + {{2U|2UL|2ULL}}, hessianMatrix + {{3U|3UL|3ULL}});
// CHECK-NEXT: _d_b = 0.;
// CHECK-NEXT: }

int main() {
  double m[4] = {0};
  clad::hessian(withEnum, "x, y").execute(1., 2., Quadratic, m);
  printf("withEnum [%.2f, %.2f, %.2f, %.2f]\n", m[0], m[1], m[2], m[3]);
  // CHECK-EXEC: withEnum [4.00, 2.00, 2.00, 0.00]

  double scale = 3.;
  double m2[4] = {0};
  clad::hessian(withRef, "x, y").execute(1., 2., scale, m2);
  printf("withRef [%.2f, %.2f, %.2f, %.2f]\n", m2[0], m2[1], m2[2], m2[3]);
  // CHECK-EXEC: withRef [12.00, 6.00, 6.00, 0.00]

  double m3[4] = {0};
  clad::hessian(withCustomPushforward).execute(3., 4., m3);
  printf("withCustomPushforward [%.2f, %.2f, %.2f, %.2f]\n", m3[0], m3[1],
         m3[2], m3[3]);
  // CHECK-EXEC: withCustomPushforward [0.00, 8.00, 8.00, 6.00]

  double m4[4] = {0};
  clad::hessian(plain).execute(1., 2., m4);
  printf("plain [%.2f, %.2f, %.2f, %.2f]\n", m4[0], m4[1], m4[2], m4[3]);
  // CHECK-EXEC: plain [4.00, 2.00, 2.00, 0.00]
}
