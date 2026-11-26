// RUN: %cladclang %s -I%S/../../include -oIssue2324.out 2>&1 | %filecheck %s
// RUN: ./Issue2324.out

#include "clad/Differentiator/Differentiator.h"

#include <cassert>

double h(float a, double b) {
  return a > 0 ? a * a : b * b;
}

double f(double x, double y) {
  return h((float)x, y);
}

int main() {
  auto d = clad::differentiate<clad::opts::vector_mode>(f);

  double dx = 0.0, dy = 0.0;

  d.execute(2.0, 3.0, &dx, &dy);
  assert(dx == 4.0);
  assert(dy == 0.0);

  d.execute(-2.0, 3.0, &dx, &dy);
  assert(dx == 0.0);
  assert(dy == 6.0);
}
