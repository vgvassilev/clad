//--------------------------------------------------------------------*- C++ -*-
// clad - The C++ Clang-based Automatic Differentiator
//
// Differentiates an answer that is found by iterating, not by a formula.
//
//----------------------------------------------------------------------------//

// Necessary for clad to work include
#include "clad/Differentiator/Differentiator.h"

#include <cmath>
#include <cstdio>

// Kepler's equation is M = E - e*sin(E), and it says where a body is on its
// orbit. It gives M from E. An orbit calculation needs E from M, and there is
// no formula for that, so you iterate until the answer stops changing.
//
// No derivative is written down anywhere below. The loop runs a number of
// times that depends on the arguments, and it stops on a value computed
// inside it. Clad differentiates what the program does.
// docs-begin-kepler
double eccentric_anomaly(double M, double e) {
  double E = M;
  for (int i = 0; i < 40; ++i) {
    double step = (E - e * std::sin(E) - M) / (1 - e * std::cos(E));
    E -= step;
    if (std::fabs(step) < 1e-15)
      break;
  }
  return E;
}
// docs-end-kepler

int main() {
  // docs-begin-kepler-call
  auto grad = clad::gradient(eccentric_anomaly);
  // docs-end-kepler-call

  double M = 0.75, e = 0.3, dM = 0., de = 0.;
  grad.execute(M, e, &dM, &de);

  // By hand you would not differentiate the loop. You would differentiate
  // the equation it solves, dM = (1 - e*cos(E)) dE - sin(E) de, and
  // rearrange. That gives the exact values printed beside clad's.
  double E = eccentric_anomaly(M, e);
  double denom = 1 - e * std::cos(E);
  printf("E     = %.8f\n", E);
  printf("dE/dM = %.8f   exact %.8f\n", dM, 1 / denom);
  printf("dE/de = %.8f   exact %.8f\n", de, std::sin(E) / denom);

  return 0;
}
