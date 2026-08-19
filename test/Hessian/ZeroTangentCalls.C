// RUN: %cladclang %s -I%S/../../include -oZeroTangentCalls.out 2>&1 | %filecheck %s
// RUN: ./ZeroTangentCalls.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr %s -I%S/../../include -oZeroTangentCalls.out
// RUN: ./ZeroTangentCalls.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"

double cube(double u) { return u * u * u; }

// A hessian seeds one direction at a time, so in every row all tangents but
// the seeded one are zero-initialized constants. A call handed such a tangent
// contributes nothing, so no pushforward is requested for it: each direction
// calls cube_pushforward once instead of twice. The parameters have to be
// const for this -- a tangent that can still be assigned to is not folded.
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

// The same holds one order up, where the hessian rows are differentiated.
// CHECK: void sumOfCubes_darg0_grad(const double x, const double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:     double _d_d_x = 0.;
// CHECK-NEXT:     const double _d_x0 = 1;
// CHECK-NEXT:     double _d_d_y = 0.;
// CHECK-NEXT:     const double _d_y0 = 0;
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _d_t0 = {0., 0.};
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t00 = cube_pushforward(x, _d_x0);
// CHECK-NOT:      cube_pushforward(y

// CHECK: void sumOfCubes_darg1_grad(const double x, const double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:     double _d_d_x = 0.;
// CHECK-NEXT:     const double _d_x0 = 0;
// CHECK-NEXT:     double _d_d_y = 0.;
// CHECK-NEXT:     const double _d_y0 = 1;
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _d_t0 = {0., 0.};
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t00 = cube_pushforward(y, _d_y0);
// CHECK-NOT:      cube_pushforward(x

int main() {
  auto hess = clad::hessian(sumOfCubes);
  double matrix[4] = {0};
  hess.execute(2, 3, matrix);
  printf("[%.2f, %.2f, %.2f, %.2f]\n", matrix[0], matrix[1], matrix[2],
         matrix[3]); // CHECK-EXEC: [12.00, 0.00, 0.00, 18.00]
}
