// RUN: %cladclang %s -I%S/../../../include -o%t 2>&1 | %filecheck_nodiag %s
// RUN: %t | %filecheck_prints %s

#include "clad/Differentiator/Differentiator.h"
#include <cmath>
#include <iostream>

// Stands in for a function clad cannot differentiate, say one whose
// definition lives in a library.
double my_pow(double x, double exponent) { return ::std::pow(x, exponent); }

// docs-begin-custom-pushforward
namespace clad {
namespace custom_derivatives {
ValueAndPushforward<double, double> my_pow_pushforward(double x,
                                                       double exponent,
                                                       double d_x,
                                                       double d_exponent) {
  return {my_pow(x, exponent),
          exponent * my_pow(x, exponent - 1) * d_x +
              (my_pow(x, exponent) * ::std::log(x)) * d_exponent};
}
} // namespace custom_derivatives
} // namespace clad
// docs-end-custom-pushforward

// docs-begin-custom-pullback
namespace clad {
namespace custom_derivatives {
void my_pow_pullback(double x, double exponent, double d_y, double* d_x,
                     double* d_exponent) {
  double t = my_pow(x, exponent - 1);
  *d_x += exponent * t * d_y;
  *d_exponent += t * x * ::std::log(x) * d_y;
}
} // namespace custom_derivatives
} // namespace clad
// docs-end-custom-pullback

// docs-begin-custom-use
double f(double x, double exponent) { return my_pow(x, exponent); }

int main() {
  // Forward mode calls my_pow_pushforward.
  auto d_f = clad::differentiate(f, "x");
  std::cout << d_f.execute(3, 4) << "\n"; // prints: 108

  // Reverse mode calls my_pow_pullback.
  auto f_grad = clad::gradient(f);
  double d_x = 0, d_exponent = 0;
  f_grad.execute(3, 4, &d_x, &d_exponent);
  std::cout << d_x << " " << d_exponent << "\n"; // prints: 108 88.9876
}
// docs-end-custom-use
