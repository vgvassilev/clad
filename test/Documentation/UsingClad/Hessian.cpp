// RUN: %cladclang %s -I%S/../../../include -o%t 2>&1 | %filecheck_nodiag %s
// RUN: %t | %filecheck_prints %s

// docs-begin-hessian
#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

double kinetic_energy(double mass, double velocity) {
  return 0.5 * mass * velocity * velocity;
}

int main() {
  // Tells clad to generate a function that computes the hessian matrix of
  // 'kinetic_energy' with respect to all the input parameters.
  auto hessian_one = clad::hessian(kinetic_energy);

  // The independent arguments can also be named explicitly.
  auto hessian_two = clad::hessian(kinetic_energy, "mass, velocity");

  // A matrix per call to store the hessian in. Each must have enough space:
  // 2 independent variables require 4 elements (2 * 2 = 4). clad adds into
  // the matrix, so each one starts at zero.
  double matrix_one[4] = {0}, matrix_two[4] = {0};

  // Substitutes these values into the hessian function and pipes the result
  // into the matrix.
  hessian_one.execute(10, 2, matrix_one);
  printf("%g %g %g %g\n", matrix_one[0], matrix_one[1], matrix_one[2],
         matrix_one[3]);
  // prints: 0 2 2 10

  hessian_two.execute(5, 1, matrix_two);
  printf("%g %g %g %g\n", matrix_two[0], matrix_two[1], matrix_two[2],
         matrix_two[3]);
  // prints: 0 1 1 5
}
// docs-end-hessian
