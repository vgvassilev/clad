// RUN: %cladclang %s -I%S/../../include -o %t 2>&1 | %filecheck %s
// RUN: %t

#include "clad/Differentiator/Differentiator.h"

struct DefaultFactor {
  struct Construct {};

  explicit constexpr DefaultFactor(Construct) {}
};

constexpr DefaultFactor defaultFactor{DefaultFactor::Construct{}};

struct OptionalFactor {
  OptionalFactor() = default;
  OptionalFactor(DefaultFactor) {}

  int value_or(int fallback) const { return fallback; }
};

double configured_scale(double value, OptionalFactor factor = defaultFactor) {
  return value * factor.value_or(2);
}

namespace clad::custom_derivatives {

void configured_scale_pullback(
    double /*value*/, OptionalFactor factor, double d_output, double* d_value,
    OptionalFactor* /*d_factor*/) {
  *d_value += factor.value_or(2) * d_output;
}

} // namespace clad::custom_derivatives

double use_default_factor(double value) { return configured_scale(value); }

// CHECK: void use_default_factor_grad(double value, double *_d_value) {
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r0 = 0.;
// CHECK-NEXT:         OptionalFactor _r1 = {};
// CHECK-NEXT:         clad::custom_derivatives::configured_scale_pullback(value, defaultFactor, 1, &_r0, &_r1);
// CHECK-NEXT:         *_d_value += _r0;
// CHECK-NEXT:     }
// CHECK-NEXT: }

int main() {
  auto gradient = clad::gradient(use_default_factor);
  double derivative = 0.0;
  gradient.execute(3.0, &derivative);
  return derivative != 2.0;
}
