// RUN: %cladclang -std=c++17 %s -I%S/../../include -DCLAD_NO_NUM_DIFF -oVoidPushforward.out 2>&1 | %filecheck %s
// RUN: ./VoidPushforward.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"
#include <iostream>

extern "C" int printf(const char* fmt, ...);

// Global Function Test
double global_fn(double x) {
  return x * x * x; // Derivative would normally be 3*x^2
}

// Custom derivative definitions for the feature to work
namespace clad {
namespace custom_derivatives {
// Void pushforward -> signals non-differentiable
void global_fn_pushforward(double, double) {}
}
}

double test_global(double x) {
  // Clad treats global_fn as non-differentiable -> derivative 0
  return global_fn(x) + x;
}

// Execution Wrapper
#define TEST_FUNC(name, arg)                                  \
  do {                                                        \
    auto d_##name = clad::differentiate(name, "x");           \
    printf("d_" #name "(%.0f) = %.2f\n", arg, d_##name.execute(arg)); \
  } while(0)

int main() {
  // Expected: 0 + 1 = 1.00
  TEST_FUNC(test_global, 2.0); // CHECK-EXEC: d_test_global(2) = 1.00
  return 0;
}

// Verification of generated code

// CHECK-LABEL: double test_global_darg0(double x) {
// CHECK:       double _d_x = 1;
// Primal call optimized away as it's pure and unused
// CHECK-NOT:   global_fn(x)
// CHECK:       return 0. + _d_x;
// CHECK-NEXT: }
