// RUN: %cladclang %s -I%S/../../include -o %t 2>&1 | %filecheck %s
// RUN: %t | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"

#include <cstdio>
#include <optional>

double configured_scale(double value,
                        ::std::optional<int> factor = ::std::nullopt) {
  return value * factor.value_or(2);
}

namespace clad::custom_derivatives {

void configured_scale_pullback(
    [[maybe_unused]] double value, ::std::optional<int> factor, double d_output,
    double* d_value,
    [[maybe_unused]] ::std::optional<int>* d_factor) {
  *d_value += factor.value_or(2) * d_output;
}

} // namespace clad::custom_derivatives

double use_default_factor(double value) { return configured_scale(value); }

// CHECK: void use_default_factor_grad(double value, double *_d_value) {
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r0 = 0.;
// CHECK-NEXT:         std::optional<int> _r1 = {};
// CHECK-NEXT:         clad::custom_derivatives::configured_scale_pullback(value, ::std::nullopt, 1, &_r0, &_r1);
// CHECK-NEXT:         *_d_value += _r0;
// CHECK-NEXT:     }
// CHECK-NEXT: }

int main() {
  auto gradient = clad::gradient(use_default_factor);
  double derivative = 0.0;
  gradient.execute(3.0, &derivative);
  std::printf("%.1f\n", derivative);
}

// CHECK-EXEC: 2.0
