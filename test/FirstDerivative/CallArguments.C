// RUN: %cladclang %s -I%S/../../include -fsyntax-only 2>&1 | FileCheck %s

#include "clad/Differentiator/Differentiator.h"

//CHECK-NOT: {{.*error|warning|note:.*}}

namespace custom_derivatives {
  float f_derived_x(float y) { return 2*y; }
}

float f(float y) {
  return y * y - 10;
}

float g(float x) {
  return f(x*x*x);
}

// CHECK: float g_derived_x(float x) {
// CHECK-NEXT: return f_derived_x(x * x * x)  * ((1.F * x + x * 1.F) * x + x * x * 1.F);
// CHECK-NEXT: }


int main () {
  clad::differentiate(g, 1);
  return 0;
}
