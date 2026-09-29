//--------------------------------------------------------------------*- C++ -*-
// clad - The C++ Clang-based Automatic Differentiator
//
// Computes a jacobian: every partial derivative of a function that has
// several outputs, in one pass.
//
//----------------------------------------------------------------------------//

// Necessary for clad to work include
#include "clad/Differentiator/Differentiator.h"

#include <cmath>
#include <cstdio>

// The same point written two ways: three numbers in, a distance and two
// angles, and three numbers out. A gradient is for one output. This function
// has three, and the full table of partial derivatives is the jacobian.
// docs-begin-jacobian
void spherical_to_cartesian(double r, double theta, double phi, double p[]) {
  p[0] = r * std::sin(theta) * std::cos(phi);
  p[1] = r * std::sin(theta) * std::sin(phi);
  p[2] = r * std::cos(theta);
}
// docs-end-jacobian

int main() {
  // docs-begin-jacobian-call
  auto jac = clad::jacobian(spherical_to_cartesian);
  // docs-end-jacobian-call

  double r = 2., theta = .7, phi = 1.1, p[3] = {0.};

  // One row per output. The columns cover every independent scalar: the
  // three arguments, then the three elements of p.
  clad::matrix<double> J(3, 6);
  jac.execute(r, theta, phi, p, &J);

  // The determinant of the 3x3 block says what a small volume in one system
  // becomes in the other. Here it is r*r*sin(theta), the factor in front of
  // every integral written in spherical coordinates. It is printed beside
  // what clad computed.
  double det = J[0][0] * (J[1][1] * J[2][2] - J[1][2] * J[2][1]) -
               J[0][1] * (J[1][0] * J[2][2] - J[1][2] * J[2][0]) +
               J[0][2] * (J[1][0] * J[2][1] - J[1][1] * J[2][0]);
  printf("det J = %.8f   exact %.8f\n", det, r * r * std::sin(theta));

  for (int row = 0; row < 3; ++row)
    printf("  %+.5f %+.5f %+.5f\n", J[row][0], J[row][1], J[row][2]);

  return 0;
}
