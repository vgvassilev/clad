// RUN: %cladclang %s -I%S/../../include -oissue-2322.out
// RUN: ./issue-2322.out | %filecheck_exec %s

// A void return in a pushforward was dropped, or crashed when it returned a void call.

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

// Reproducer 1: A bare return; was dropped in vector pushforward.
void v(double& r, double x) {
  r = x * x;
  if (x > 0)
    return;
  r = 100 * x;
}

double f(double x, double y) {
  double r = 0;
  v(r, x);
  return r * y;
}

// Reproducer 2: return g(...); of a void call crashed pushforward.
void v2(double& r, double x) {
  r = x * x;
}

void w(double& r, double x) {
  return v2(r, x);
}

double g(double x, double y) {
  double r = 0;
  w(r, x);
  return r * y;
}

int main() {
  // Test Reproducer 1
  auto d1 = clad::differentiate(f, "x");
  printf("%.2f\n", d1.execute(2, 3)); // CHECK-EXEC: 12.00

  auto dv1 = clad::differentiate<clad::opts::vector_mode>(f);
  double dx1 = 0, dy1 = 0;
  dv1.execute(2, 3, &dx1, &dy1);
  printf("%.2f %.2f\n", dx1, dy1); // CHECK-EXEC: 12.00 4.00

  // Test Reproducer 2
  auto d2 = clad::differentiate(g, "x");
  printf("%.2f\n", d2.execute(2, 3)); // CHECK-EXEC: 12.00

  auto dv2 = clad::differentiate<clad::opts::vector_mode>(g);
  double dx2 = 0, dy2 = 0;
  dv2.execute(2, 3, &dx2, &dy2);
  printf("%.2f %.2f\n", dx2, dy2); // CHECK-EXEC: 12.00 4.00

  return 0;
}
