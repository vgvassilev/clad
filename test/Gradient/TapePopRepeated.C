// RUN: %cladclang %s -I%S/../../include -oTapePopRepeated.out 2>&1 | %filecheck %s
// RUN: ./TapePopRepeated.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"
#include <cmath>
#include <cstdio>

// Regression test for issue #2240: repeated use of a saved intermediate
// in a loop that requires tape storage.
double f(double k) {
  double v = 1.0;
  for (int i = 0; i < 513; ++i)
    v -= k * v * std::fabs(v) * 1e-3;
  return v;
}

// CHECK: void f_grad(double k, double *_d_k) {
int main() {
  auto d = clad::gradient(f, "k");
  double dk = 0.0;
  d.execute(0.1, &dk);

  const double h = 1e-6;
  const double fd = (f(0.1 + h) - f(0.1 - h)) / (2.0 * h);

  if (std::fabs(dk - fd) < 1e-6)
    std::printf("gradient matches finite differences\n");
  else
    std::printf("gradient mismatch\n");
}

// CHECK-EXEC: gradient matches finite differences
