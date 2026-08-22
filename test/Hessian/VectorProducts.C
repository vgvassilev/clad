// RUN: %cladclang %s -I%S/../../include -oVectorProducts.out 2>&1 | %filecheck %s
// RUN: ./VectorProducts.out | %filecheck_exec %s

// A hessian is assembled from hessian-vector products: one pushforward of the
// function, whose direction is a run-time argument, and one pullback of that
// pushforward. Seeding the pullback with {.value = 0, .pushforward = 1} and a
// direction of e_i yields row i, so the pair serves every direction and the
// generated code stops growing with the number of parameters.

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

double manyDirections(double* p, double q) {
  return p[0] * p[1] * p[2] + p[3] * q * q;
}

// Four array directions and a scalar one, yet a single pullback covers them
// all -- and the wrapper reuses one tangent buffer per parameter.
// CHECK:  void manyDirections_hessian(double *p, double q, double *hessianMatrix) {
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _d_y{0., 0.};
// CHECK-NEXT:     _d_y.pushforward = 1.;
// CHECK-NEXT:     double _d_p[4]{0};
// CHECK-NEXT:     double _d_q(0.);
// CHECK-NEXT:     _d_p[{{0U|0UL|0ULL}}] = 1.;
// CHECK-NEXT:     manyDirections_pushforward_pullback(p, q, _d_p, _d_q, _d_y, hessianMatrix + {{0U|0UL|0ULL}}, hessianMatrix + {{4U|4UL|4ULL}});
// CHECK-NEXT:     _d_p[{{0U|0UL|0ULL}}] = 0.;
// CHECK-NEXT:     _d_p[{{1U|1UL|1ULL}}] = 1.;
// CHECK-NEXT:     manyDirections_pushforward_pullback(p, q, _d_p, _d_q, _d_y, hessianMatrix + {{5U|5UL|5ULL}}, hessianMatrix + {{9U|9UL|9ULL}});
// CHECK-NEXT:     _d_p[{{1U|1UL|1ULL}}] = 0.;
// CHECK-NEXT:     _d_p[{{2U|2UL|2ULL}}] = 1.;
// CHECK-NEXT:     manyDirections_pushforward_pullback(p, q, _d_p, _d_q, _d_y, hessianMatrix + {{10U|10UL|10ULL}}, hessianMatrix + {{14U|14UL|14ULL}});
// CHECK-NEXT:     _d_p[{{2U|2UL|2ULL}}] = 0.;
// CHECK-NEXT:     _d_p[{{3U|3UL|3ULL}}] = 1.;
// CHECK-NEXT:     manyDirections_pushforward_pullback(p, q, _d_p, _d_q, _d_y, hessianMatrix + {{15U|15UL|15ULL}}, hessianMatrix + {{19U|19UL|19ULL}});
// CHECK-NEXT:     _d_p[{{3U|3UL|3ULL}}] = 0.;
// CHECK-NEXT:     _d_q = 1.;
// CHECK-NEXT:     manyDirections_pushforward_pullback(p, q, _d_p, _d_q, _d_y, hessianMatrix + {{20U|20UL|20ULL}}, hessianMatrix + {{24U|24UL|24ULL}});
// CHECK-NEXT:     _d_q = 0.;
// CHECK-NEXT: }

// A parameter no direction runs through keeps a zero tangent; for a pointer
// that is a null tangent, which forward mode reads as a zero derivative.
double someDirections(double* p, const double* x) {
  return p[0] * p[0] * x[0] + p[1] * p[1] * p[1] * x[1];
}

// CHECK:  void someDirections_hessian_0(double *p, const double *x, double *hessianMatrix) {
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _d_y{0., 0.};
// CHECK-NEXT:     _d_y.pushforward = 1.;
// CHECK-NEXT:     double _d_p[2]{0};
// CHECK-NEXT:     _d_p[{{0U|0UL|0ULL}}] = 1.;
// CHECK-NEXT:     someDirections_pushforward_0_pullback(p, x, _d_p, nullptr, _d_y, hessianMatrix + {{0U|0UL|0ULL}});
// CHECK-NEXT:     _d_p[{{0U|0UL|0ULL}}] = 0.;
// CHECK-NEXT:     _d_p[{{1U|1UL|1ULL}}] = 1.;
// CHECK-NEXT:     someDirections_pushforward_0_pullback(p, x, _d_p, nullptr, _d_y, hessianMatrix + {{2U|2UL|2ULL}});
// CHECK-NEXT:     _d_p[{{1U|1UL|1ULL}}] = 0.;
// CHECK-NEXT: }

// Two hessians of one function w.r.t. different single parameters take two
// different pullbacks of the same pushforward. The subset is part of the
// pullback's name: with one name, the two same-signature inline definitions
// would collapse onto one symbol and one body would serve both hessians.
double twoSubsets(double x, double y) { return x * x * y + y * y; }

// CHECK:  void twoSubsets_hessian_0(double x, double y, double *hessianMatrix) {
// CHECK:  twoSubsets_pushforward_0_pullback(x, y, _d_x, 0., _d_y, hessianMatrix + {{0U|0UL|0ULL}});

// CHECK:  void twoSubsets_hessian_1(double x, double y, double *hessianMatrix) {
// CHECK:  twoSubsets_pushforward_1_pullback(x, y, 0., _d_y0, _d_y, hessianMatrix + {{0U|0UL|0ULL}});

int main() {
  double p[4] = {2., 3., 5., 7.};
  double m[25] = {};
  clad::hessian(manyDirections, "p[0:3], q").execute(p, 11., m);
  for (int i = 0; i < 5; ++i)
    printf("%.0f %.0f %.0f %.0f %.0f\n", m[5*i], m[5*i+1], m[5*i+2], m[5*i+3],
           m[5*i+4]);
  // CHECK-EXEC: 0 5 3 0 0
  // CHECK-EXEC: 5 0 2 0 0
  // CHECK-EXEC: 3 2 0 0 0
  // CHECK-EXEC: 0 0 0 0 22
  // CHECK-EXEC: 0 0 0 22 14

  double q[2] = {2., 3.};
  double x[2] = {1.5, 0.5};
  double m2[4] = {};
  clad::hessian(someDirections, "p[0:1]").execute(q, x, m2);
  printf("%.2f %.2f %.2f %.2f\n", m2[0], m2[1], m2[2], m2[3]);
  // CHECK-EXEC: 3.00 0.00 0.00 9.00

  double mx[1] = {};
  double my[1] = {};
  clad::hessian(twoSubsets, "x").execute(2., 3., mx);
  clad::hessian(twoSubsets, "y").execute(2., 3., my);
  printf("%.2f %.2f\n", mx[0], my[0]);
  // CHECK-EXEC: 6.00 2.00
}
