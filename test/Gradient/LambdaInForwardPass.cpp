// RUN: %cladclang %s -I%S/../../include -oLambdaInForwardPass.out
// RUN: ./LambdaInForwardPass.out | %filecheck_exec %s
// UNSUPPORTED: clang-14, clang-15, clang-16

// A function that writes through a pointer gets a forward pass of its own.
// A lambda declared in it, capturing a parameter, must keep compiling there.

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

void scale(double* p, double s) {
  auto mul = [s](double v) { return s * v; };
  p[0] = mul(p[0]);
}

// f = 3 x^2, f' = 6 x.
double f(double x) {
  double o[1] = {x};
  scale(o, 3);
  return o[0] * x;
}

int main() {
  double dx = 0;
  clad::gradient(f).execute(1.5, &dx);
  printf("%.2f\n", dx); // CHECK-EXEC: 9.00
}
