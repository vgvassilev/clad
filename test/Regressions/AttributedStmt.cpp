// RUN: %cladclang %s -std=c++20 -I%S/../../include -o%t 2>&1 | %filecheck %s
// RUN: %t | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

double branch(double x) {
  double r = x;
  if (x > 0) [[likely]] r *= 2;
  else [[unlikely]] r *= 3;
  return r;
}

double nested(double x) {
  double r = x;
  if (x > 0) [[likely]] {
    for (int i = 0; i < 2; ++i) [[likely]] r *= 2;
  }
  return r;
}

double labelled(double x, int i) {
  double r = 0;
  switch (i) {
    [[likely]] case 0: r = x * x; break;
    [[unlikely]] default: r = 3 * x; break;
  }
  return r;
}

int main() {
  auto df = clad::differentiate(branch, "x");
  auto gf = clad::gradient(branch);
  auto dn = clad::differentiate(nested, "x");
  auto gn = clad::gradient(nested);
  auto dl = clad::differentiate(labelled, "x");
  auto gl = clad::gradient(labelled, "x");
  for (double x : {2., -2.}) {
    double d = 0, n = 0;
    gf.execute(x, &d);
    gn.execute(x, &n);
    std::printf("%.0f %.0f %.0f %.0f\n", df.execute(x), d, dn.execute(x), n);
  }
  for (int i : {0, 1}) {
    double d = 0;
    gl.execute(2., i, &d);
    std::printf("%.0f %.0f\n", dl.execute(2., i), d);
  }
}

// CHECK-EXEC: 2 2 4 4
// CHECK-EXEC-NEXT: 3 3 1 1

// CHECK-EXEC-NEXT: 4 4
// CHECK-EXEC-NEXT: 3 3

