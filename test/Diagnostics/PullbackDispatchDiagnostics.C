// RUN: %cladclang %s -I%S/../../include -o %t -Xclang -verify
// RUN: %t | %filecheck_exec %s
#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

double dispatch_square(double x) { return x * x; }
namespace clad { namespace custom_derivatives {
clad::ValueAndPushforward<double, double>
dispatch_square_pushforward(double x, double d_x) {
  return {x*x, 2*x*d_x};
}
void dispatch_square_pullback(double x, double seed) { // expected-note {{unused}}
  // This legacy warning candidate must never be selected for execution.
}
}}
double dispatch_outer(double x) {
  return dispatch_square(x); // expected-warning {{unused function 'dispatch_square_pullback';}}
}
int main() {
  auto pb = clad::pullback(dispatch_outer);
  double dx = 0;
  pb.execute(3., 2., &dx);
  std::printf("Dispatch: %.0f\n", dx);
  // CHECK-EXEC: Dispatch: 12
}
