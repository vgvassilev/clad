// RUN: %cladclang %s -I%S/../../include -oCallArguments.out -lm 2>&1 | FileCheck %s
// RUN: ./CallArguments.out | FileCheck -check-prefix=CHECK-EXEC %s

#include "clad/Differentiator/Differentiator.h"

//CHECK-NOT: {{.*error|warning|note:.*}}

namespace custom_derivatives {
  float f_dx(float y) { return 2*y; }
}

float f(float y) {
  return y * y - 10;
}

float g(float x) {
  return f(x*x*x);
}

// CHECK: float g_dx(float x) {
// CHECK-NEXT: return f_dx(x * x * x)  * (((1.F * x + x * 1.F) * x + x * x * 1.F));
// CHECK-NEXT: }

float sqrt_func(float x, float y) {
  return sqrt(x * x + y * y) - y;
}

// CHECK: float sqrt_func_dx(float x, float y) {
// CHECK-NEXT: sqrt_dx(x * x + y * y) * ((1.F * x + x * 1.F) + ((0.F * y + y * 0.F))) - (0.F);
// CHECK-NEXT: }

extern "C" int printf(const char* fmt, ...);
int main () {
  auto f = clad::differentiate(g, 0);
  printf("g_dx=%f\n", f.execute(1));
  //CHECK-EXEC: g_dx=6.000000

  clad::differentiate(sqrt_func, 0);

  return 0;
}
