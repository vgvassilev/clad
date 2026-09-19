// RUN: %cladclang %s -I%S/../../include -fsyntax-only -Xclang -verify 2>&1

#include "clad/Differentiator/Differentiator.h"

double f_scalar(double x) { // expected-note {{'f_scalar' declared here with return type 'double'}}
  return x * x;
}

// expected-note@+1 {{'model_and_loss' declared here with return type 'double'}}
double model_and_loss(double x0, double x1, double w0, double w1, double b,
                      double t, double* out) {
  double y = w0 * x0 + w1 * x1 + b;
  double diff = y - t;
  return 0.5 * diff * diff;
}

int status_output(double x, double* out) { // expected-note {{'status_output' declared here with return type 'int'}}
  *out = x * x;
  return 0;
}

double f_custom(double x) { // expected-note {{'f_custom' declared here with return type 'double'}}
  return x * x;
}

auto f_trailing(double x) -> double { // expected-note {{'f_trailing' declared here with return type 'double'}}
  return x * x;
}

namespace clad {
namespace custom_derivatives {
clad::ValueAndPushforward<double, clad::array<double>*> f_custom_jacobian(
    double x) {
  return {x * x, nullptr};
}
} // namespace custom_derivatives
} // namespace clad

int main() {
  auto df_scalar = clad::jacobian(f_scalar); // expected-error {{jacobian mode currently requires function 'f_scalar' to return void; provide differentiable outputs through pointer, reference, or array parameters}}
  auto df_loss = clad::jacobian(model_and_loss); // expected-error {{jacobian mode currently requires function 'model_and_loss' to return void; provide differentiable outputs through pointer, reference, or array parameters}}
  auto df_status = clad::jacobian(status_output); // expected-error {{jacobian mode currently requires function 'status_output' to return void; provide differentiable outputs through pointer, reference, or array parameters}}
  auto df_custom = clad::jacobian(f_custom); // expected-error {{jacobian mode currently requires function 'f_custom' to return void; provide differentiable outputs through pointer, reference, or array parameters}}
  auto df_trailing = clad::jacobian(f_trailing); // expected-error {{jacobian mode currently requires function 'f_trailing' to return void; provide differentiable outputs through pointer, reference, or array parameters}}
  return 0;
}
