// RUN: %cladclang %s -I%S/../../include -o%t.out 2>&1
// RUN: %t.out

#include "clad/Differentiator/TapeAndPullback.h"

#include <cmath>
#include <cstdio>

int main() {
  clad::tape<double> t = {};
  clad::push<double>(t, 3.0);

  double d_x = 0.0;
  double d_exponent = 0.0;
  clad::custom_derivatives::std::pow_pullback(clad::back<double>(t), 2.0, 1.0,
                                              &d_x, &d_exponent);

  if (std::fabs(d_x - 6.0) > 1e-12) {
    std::printf("unexpected derivative: %f\n", d_x);
    return 1;
  }

  double v = clad::pop<double>(t);
  if (std::fabs(v - 3.0) > 1e-12) {
    std::printf("unexpected tape value: %f\n", v);
    return 2;
  }

  return 0;
}
