// RUN: %cladclang %s -I%S/../../include -oMixedInstantiations.out 2>&1 | %filecheck %s
// RUN: ./MixedInstantiations.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr %s -I%S/../../include -oMixedInstantiations.out
// RUN: ./MixedInstantiations.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"

#include <cmath>
#include <cstdio>

// A derivative is named after the function it differentiates, so the two
// instantiations below ask for derivatives of the same name. The one built
// first must not be called for the second: its parameter types do not match.
template <typename T> double scale(double x, T n) { return x * x * n; }

double sumOfScales(double x) { return scale(x, 2.) + scale(x, 3); }

// CHECK: inline clad::ValueAndPushforward<double, double> scale_pushforward(double x, double n, double _d_x, double _d_n) {
// CHECK-NEXT:     double _t0 = x * x;
// CHECK-NEXT:     return {_t0 * n, (_d_x * x + x * _d_x) * n + _t0 * _d_n};
// CHECK-NEXT: }

// CHECK: inline clad::ValueAndPushforward<double, double> scale_pushforward(double x, int n, double _d_x, int _d_n) {
// CHECK-NEXT:     double _t0 = x * x;
// CHECK-NEXT:     return {_t0 * n, (_d_x * x + x * _d_x) * n + _t0 * _d_n};
// CHECK-NEXT: }

// CHECK: inline void scale_pushforward_pullback(double x, double n, double _d_x, double _d_n, clad::ValueAndPushforward<double, double> _d_y, double *_d_x0, double *_d_n0, double *_d_d_x, double *_d_d_n);
// CHECK-NEXT: inline void scale_pushforward_pullback(double x, int n, double _d_x, int _d_n, clad::ValueAndPushforward<double, double> _d_y, double *_d_x0, int *_d_n0, double *_d_d_x, int *_d_d_n);

// CHECK: inline void sumOfScales_pushforward_pullback(double x, double _d_x, clad::ValueAndPushforward<double, double> _d_y, double *_d_x0) {
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _d_t0 = {0., 0.};
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t00 = scale_pushforward(x, 2., _d_x, 0.);
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _d_t1 = {0., 0.};
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t10 = scale_pushforward(x, 3, _d_x, 0);
// CHECK-NEXT:     {
// CHECK-NEXT:         _d_t0.value += _d_y.value;
// CHECK-NEXT:         _d_t1.value += _d_y.value;
// CHECK-NEXT:         _d_t0.pushforward += _d_y.pushforward;
// CHECK-NEXT:         _d_t1.pushforward += _d_y.pushforward;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r4 = 0.;
// CHECK-NEXT:         int _r5 = 0;
// CHECK-NEXT:         double _r6 = 0.;
// CHECK-NEXT:         int _r7 = 0;
// CHECK-NEXT:         scale_pushforward_pullback(x, 3, _d_x, 0, _d_t1, &_r4, &_r5, &_r6, &_r7);
// CHECK-NEXT:         *_d_x0 += _r4;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r0 = 0.;
// CHECK-NEXT:         double _r1 = 0.;
// CHECK-NEXT:         double _r2 = 0.;
// CHECK-NEXT:         double _r3 = 0.;
// CHECK-NEXT:         scale_pushforward_pullback(x, 2., _d_x, 0., _d_t0, &_r0, &_r1, &_r2, &_r3);
// CHECK-NEXT:         *_d_x0 += _r0;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// A static member function reaches the same lookup through its class rather
// than a namespace, and it too has to see both overloads.
struct Scaler {
  template <typename T> static double scale(double x, T n) { return x * x * n; }
};

double sumOfStaticScales(double x) {
  return Scaler::scale(x, 2.) + Scaler::scale(x, 3);
}

// CHECK: static inline void scale_pushforward_pullback(double x, double n, double _d_x, double _d_n, clad::ValueAndPushforward<double, double> _d_y, double *_d_x0, double *_d_n0, double *_d_d_x, double *_d_d_n);
// CHECK-NEXT: static inline void scale_pushforward_pullback(double x, int n, double _d_x, int _d_n, clad::ValueAndPushforward<double, double> _d_y, double *_d_x0, int *_d_n0, double *_d_d_x, int *_d_d_n);

// The same clash reached clad through std::pow: an integral exponent and a
// floating-point one are two instantiations sharing the name pow_pushforward.
// This is the shape of RooFit's Bernstein integral, whose Hessian it broke.
double poly(double x, double y) {
  double res = 0;
  for (int j = 0; j < 3; ++j)
    for (int i = 0; i <= j; ++i) {
      double powDiff = std::pow(x, j + 1.) - std::pow(y, j + 1.);
      res += std::pow(-1., j - i) * powDiff;
    }
  return res;
}

int main() {
  auto scalesHess = clad::hessian(sumOfScales);
  double scalesMatrix[1] = {0};
  scalesHess.execute(1.5, scalesMatrix);
  printf("[%.2f]\n", scalesMatrix[0]); // CHECK-EXEC: [10.00]

  double staticMatrix[1] = {0};
  clad::hessian(sumOfStaticScales).execute(1.5, staticMatrix);
  printf("[%.2f]\n", staticMatrix[0]); // CHECK-EXEC: [10.00]

  // poly(x, y) = (x - y) + (x^3 - y^3), so the Hessian is diag(6x, -6y).
  auto polyHess = clad::hessian(poly);
  double polyMatrix[4] = {0};
  polyHess.execute(2, 3, polyMatrix);
  printf("[%.2f, %.2f, %.2f, %.2f]\n", polyMatrix[0], polyMatrix[1],
         polyMatrix[2], polyMatrix[3]);
  // CHECK-EXEC: [12.00, 0.00, 0.00, -18.00]
}
