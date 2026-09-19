// RUN: %cladclang %s -I%S/../../../include -o%t 2>&1 | %filecheck_nodiag %s
// RUN: %t | %filecheck_prints %s

// docs-begin-non-differentiable-variable
#include "clad/Differentiator/Differentiator.h"
#include <iostream>

class PointData {
public:
  double x;
  double y;
  CLAD_NONDIFFERENTIABLE double weight; // not differentiated
};

double energy(PointData p) { return p.weight * (p.x * p.x + p.y * p.y); }

int main() {
  auto d_energy = clad::gradient(energy, "p");

  PointData p{3, 4, 2}, d_p{};
  d_energy.execute(p, &d_p);

  // No derivative is generated for 'weight', so its adjoint stays zero.
  std::cout << d_p.x << " " << d_p.y << " " << d_p.weight << "\n";
  // prints: 12 16 0
}
// docs-end-non-differentiable-variable
