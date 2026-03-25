// RUN: %cladclang -std=c++20 -I%S/../../include %s -o %t
// RUN: %t | %filecheck_exec %s
#include <cmath>
#include <iostream>

#include "clad/Differentiator/Differentiator.h"

double gauss_shifted_mean(double* params, const double* obs) {
  double x[] = {obs[0], params[0] - params[1], params[2]};
  const double arg = x[0] - x[1];
  const double sig = x[2];
  return std::exp(-0.5 * arg * arg / (sig * sig));
}

double gaussian_numeric_int(double* params) {
  double output = 0.0;
  double t6[1];
  {
    const int n = 100;
    const double d = 4 - -4;
    const double eps = d / n;
    #pragma clad checkpoint loop
    for (int i = 0; i < n; ++i) {
      t6[0] = -4 + eps * i;
      const double tmpA = gauss_shifted_mean(params, t6);
      t6[0] = -4 + eps * (i + 1);
      const double tmpB = gauss_shifted_mean(params, t6);
      output += (tmpA + tmpB) * 0.5 * eps;
    }
  }
  return output;
}

double gauss_point(double* params, double x) {
  double obs[1] = {x};
  return gauss_shifted_mean(params, obs);
}

double gaussian_numeric_int_no_braces(double* params) {
  double output = 0.0;
  const int n = 16;
  const double eps = (4 - -4) / static_cast<double>(n);
  #pragma clad checkpoint loop
  for (int i = 0; i < n; ++i)
    output += gauss_point(params, -4 + eps * i);
  return output;
}

#pragma clad ON
void gradient_request() {
  clad::gradient(gaussian_numeric_int, "params");
  clad::gradient(gaussian_numeric_int_no_braces, "params");
}
#pragma clad OFF

int main() {
  gradient_request();
  std::cout << "ok\n";
  // CHECK-EXEC: ok
  return 0;
}
