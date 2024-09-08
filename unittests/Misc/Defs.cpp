#include "clad/Differentiator/Differentiator.h"

double foo(double x, double alpha, double theta, double x0 = 0) {
  return x * alpha * theta * x0;
}

namespace clad {
namespace custom_derivatives {
clad::ValueAndPushforward<double, double> sq_pushforward(double x,
                                                         double _d_x) {
  return {x * x, 2 * x};
}

void sq_pushforward_pullback(double x, double _dx,
                             clad::ValueAndPushforward<double, double> _d_y,
                             double* _d_x, double* _d__d_x) {
  {
    *_d_x += _d_y.value * x;
    *_d_x += x * _d_y.value;
    *_d_x += 2 * _d_y.pushforward;
  }
}

float custom_fn_darg0(float x, float y) { return cos(x); }

void custom_fn_darg0_grad(float x, float y, float* d_x, float* d_y) {
  *d_x -= sin(x);
}

} // namespace custom_derivatives
} // namespace clad