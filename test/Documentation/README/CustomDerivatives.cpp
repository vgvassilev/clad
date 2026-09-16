// RUN: %cladclang %s -I%S/../../../include -o%t 2>&1 | %filecheck_nodiag %s
// RUN: %t | %filecheck_prints %s
// docs-readme: custom-derivatives

// docs-begin-custom-derivatives
#include "clad/Differentiator/Differentiator.h"
#include <cmath>
#include <iostream>

// Suppose clad cannot differentiate my_pow's body, but you know the formulas.
double my_pow(double x, double y) { return std::pow(x, y); }

namespace clad::custom_derivatives {
// Forward mode: return the value alongside the directional derivative.
clad::ValueAndPushforward<double, double>
my_pow_pushforward(double x, double y, double d_x, double d_y) {
  return {my_pow(x, y),
          y * my_pow(x, y - 1) * d_x + my_pow(x, y) * ::std::log(x) * d_y};
}

// Reverse mode: add this call's contribution into each adjoint.
void my_pow_pullback(double x, double y, double d_out, double* _d_x,
                     double* _d_y) {
  *_d_x += y * my_pow(x, y - 1) * d_out;
  *_d_y += my_pow(x, y) * ::std::log(x) * d_out;
}
} // namespace clad::custom_derivatives

double f(double x, double y) { return my_pow(x, y); }

int main() {
  auto f_dx = clad::differentiate(f, "x");
  std::cout << f_dx.execute(3, 2) << std::endl; // prints: 6

  auto f_grad = clad::gradient(f);
  double d_x = 0, d_y = 0;
  f_grad.execute(3, 2, &d_x, &d_y);
  std::cout << d_x << " " << d_y << std::endl; // prints: 6 9.88751
}
// docs-end-custom-derivatives
