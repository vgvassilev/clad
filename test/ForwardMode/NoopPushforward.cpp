// RUN: %cladclang %s -I%S/../../include -oNoopPushforward.out
// RUN: ./NoopPushforward.out | %filecheck_exec %s

// A custom pushforward declared CLAD_NOOP_DERIVATIVE gives its call a zero
// tangent; it is not called, so it needs no definition.

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

double scale(double x) { return 3 * x; }

int calls = 0;
void note(double) { ++calls; }

namespace clad {
namespace custom_derivatives {
CLAD_NOOP_DERIVATIVE ValueAndPushforward<double, double>
scale_pushforward(double x, double d_x);
CLAD_NOOP_DERIVATIVE void note_pushforward(double x, double d_x);
} // namespace custom_derivatives
} // namespace clad

double f(double x) {
  note(x);
  return x * scale(x);
}

int main() {
  auto df = clad::differentiate(f, "x");
  printf("%.2f %d\n", df.execute(2), calls); // CHECK-EXEC: 6.00 1
}
