// RUN: %cladclang %s -I%S/../../../include -std=c++20 -o%t 2>&1 | %filecheck_nodiag %s
// RUN: %t | %filecheck_prints %s
// The immediate-mode path in the plugin is compiled only for clang 17 and
// later (tools/ClangPlugin.cpp), so on older clang no derivative is generated
// and the constexpr initialisation below cannot be evaluated.
// UNSUPPORTED: clang-11, clang-12, clang-13, clang-14, clang-15, clang-16

// docs-begin-immediate-mode
#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

constexpr double fn(double x, double y) { return (x + y) / 2; }

constexpr double fn_test() {
  auto dx = clad::differentiate<clad::immediate_mode>(fn, "x");

  return dx.execute(4, 7);
}

int main() {
  constexpr double fn_result = fn_test();

  printf("%.2f\n", fn_result); // prints: 0.50
}
// docs-end-immediate-mode
