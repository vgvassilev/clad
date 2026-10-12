//--------------------------------------------------------------------*- C++ -*-
// clad - The C++ Clang-based Automatic Differentiator
//
// Minimises the Rosenbrock function with Newton's method. Clad writes both
// the gradient and the hessian it needs.
//
//----------------------------------------------------------------------------//

// Necessary for clad to work include
#include "clad/Differentiator/Differentiator.h"

#include <cstdio>

// The Rosenbrock function. Its minimum is at (1, 1), where the value is 0.
// Around it is a long curved valley with a nearly flat floor. Following the
// slope alone gets into the valley quickly and then crawls, which is why it
// is the standard test for an optimiser.
// docs-begin-newton
double rosenbrock(double x, double y) {
  return (x - 1) * (x - 1) + 100 * (y - x * x) * (y - x * x);
}
// docs-end-newton

int main() {
  // Newton's method uses both derivatives. The gradient says which way is
  // downhill. The hessian says how fast the slope is changing, and that is
  // what lets one step cross the flat floor instead of inching along it.
  // docs-begin-newton-call
  auto grad = clad::gradient(rosenbrock);
  auto hess = clad::hessian(rosenbrock);
  // docs-end-newton-call

  // Rosenbrock's traditional starting point, on the far side of the valley.
  double x = -1.2, y = 1.;

  // Watch step two. It leaves the valley and the value goes up, because
  // Newton assumes the hessian describes the whole function and here it does
  // not. A real optimiser would shorten that step. This one also assumes the
  // hessian can be inverted, which is true along this path and not checked.
  for (int step = 1; step <= 6; ++step) {
    // clad adds into what it is handed, so both start at zero every step.
    double dx = 0., dy = 0., h[4] = {0.};
    grad.execute(x, y, &dx, &dy);
    hess.execute(x, y, h);

    // A Newton step solves H s = -g for the step s. With two variables the
    // inverse is short enough to write out.
    double det = h[0] * h[3] - h[1] * h[2];
    x += -(h[3] * dx - h[1] * dy) / det;
    y += -(h[0] * dy - h[2] * dx) / det;

    printf("step %d: x = % .6f, y = % .6f, f = %.3e\n", step, x, y,
           rosenbrock(x, y));
  }

  return 0;
}
