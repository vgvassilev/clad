// RUN: %cladclang %s -I%S/../../include -oZeroTangentCalls.out 2>&1 | %filecheck %s
// RUN: ./ZeroTangentCalls.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr %s -I%S/../../include -oZeroTangentCalls.out
// RUN: ./ZeroTangentCalls.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"

double cube(double u) { return u * u * u; }

// Deriving in one direction makes all tangents but the seeded one
// zero-initialized constants. A call handed such a tangent contributes
// nothing, so no pushforward is requested for it: each direction calls
// cube_pushforward once instead of twice. The parameters have to be const
// for this -- a tangent that can still be assigned to is not folded.
double sumOfCubes(const double x, const double y) {
  return cube(x) + cube(y);
}

// CHECK: double sumOfCubes_darg0(const double x, const double y) {
// CHECK-NEXT:     const double _d_x = 1;
// CHECK-NEXT:     const double _d_y = 0;
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t0 = cube_pushforward(x, _d_x);
// CHECK-NEXT:     return _t0.pushforward + 0.;
// CHECK-NEXT: }

// CHECK: double sumOfCubes_darg1(const double x, const double y) {
// CHECK-NEXT:     const double _d_x = 0;
// CHECK-NEXT:     const double _d_y = 1;
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t0 = cube_pushforward(y, _d_y);
// CHECK-NEXT:     return 0. + _t0.pushforward;
// CHECK-NEXT: }

// A free function's hessian is assembled from hessian-vector products, whose
// direction is a run-time argument, so there is nothing to fold there. The
// wrapper reseeds its tangents between directions, so they drop the const of
// the parameters they mirror.
// CHECK: void sumOfCubes_hessian(const double x, const double y, double *hessianMatrix) {
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _d_y{0., 0.};
// CHECK-NEXT:     _d_y.pushforward = 1.;
// CHECK-NEXT:     double _d_x(0.);
// CHECK-NEXT:     double _d_y0(0.);
// CHECK-NEXT:     _d_x = 1.;
// CHECK-NEXT:     sumOfCubes_pushforward_pullback(x, y, _d_x, _d_y0, _d_y, hessianMatrix + {{0U|0UL|0ULL}}, hessianMatrix + {{1U|1UL|1ULL}});
// CHECK-NEXT:     _d_x = 0.;
// CHECK-NEXT:     _d_y0 = 1.;
// CHECK-NEXT:     sumOfCubes_pushforward_pullback(x, y, _d_x, _d_y0, _d_y, hessianMatrix + {{2U|2UL|2ULL}}, hessianMatrix + {{3U|3UL|3ULL}});
// CHECK-NEXT:     _d_y0 = 0.;
// CHECK-NEXT: }

// An instance method's hessian still derives per direction, so the same
// folding holds one order up, where the hessian rows are differentiated.
struct SumOfCubes {
  double operator()(const double x, const double y) const {
    return cube(x) + cube(y);
  }
};

// CHECK: void operator_call_darg0_grad(const double x, const double y, SumOfCubes *_d_this, double *_d_x, double *_d_y) const {
// CHECK-NEXT:     double _d_d_x = 0.;
// CHECK-NEXT:     const double _d_x0 = 1;
// CHECK-NEXT:     double _d_d_y = 0.;
// CHECK-NEXT:     const double _d_y0 = 0;
// CHECK:          clad::ValueAndPushforward<double, double> _d_t0 = {0., 0.};
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t00 = cube_pushforward(x, _d_x0);
// CHECK-NOT:      cube_pushforward(y

// CHECK: void operator_call_darg1_grad(const double x, const double y, SumOfCubes *_d_this, double *_d_x, double *_d_y) const {
// CHECK-NEXT:     double _d_d_x = 0.;
// CHECK-NEXT:     const double _d_x0 = 0;
// CHECK-NEXT:     double _d_d_y = 0.;
// CHECK-NEXT:     const double _d_y0 = 1;
// CHECK:          clad::ValueAndPushforward<double, double> _d_t0 = {0., 0.};
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t00 = cube_pushforward(y, _d_y0);
// CHECK-NOT:      cube_pushforward(x

// The pullback of the free function's pushforward bounds the CHECK-NOT
// above; it calls cube_pushforward on both parameters, as it serves every
// direction.
// CHECK: void sumOfCubes_pushforward_pullback(const double x, const double y, const double _d_x, const double _d_y, clad::ValueAndPushforward<double, double> _d_y0, double *_d_x0, double *_d_y1) {

int main() {
  auto dx = clad::differentiate(sumOfCubes, "x");
  auto dy = clad::differentiate(sumOfCubes, "y");
  printf("[%.2f, %.2f]\n", dx.execute(2, 3), dy.execute(2, 3));
  // CHECK-EXEC: [12.00, 27.00]

  auto methodHess = clad::hessian(&SumOfCubes::operator());
  double methodMatrix[4] = {0};
  SumOfCubes f;
  methodHess.execute(f, 2, 3, methodMatrix);
  printf("[%.2f, %.2f, %.2f, %.2f]\n", methodMatrix[0], methodMatrix[1],
         methodMatrix[2], methodMatrix[3]);
  // CHECK-EXEC: [12.00, 0.00, 0.00, 18.00]

  auto hess = clad::hessian(sumOfCubes);
  double matrix[4] = {0};
  hess.execute(2, 3, matrix);
  printf("[%.2f, %.2f, %.2f, %.2f]\n", matrix[0], matrix[1], matrix[2],
         matrix[3]); // CHECK-EXEC: [12.00, 0.00, 0.00, 18.00]
}
