// RUN: not %cladclang %s -I%S/../../include -fsyntax-only 2>&1 | FileCheck %s
//
// Forward-mode counterpart of implicit-move-assign-array-member.cpp. Reduced
// from the Hessian of ImpactX's envelope model.

#include "clad/Differentiator/Differentiator.h"

struct V { double m[2]; };

V twice(V a) {
  a.m[0] *= 2;
  return a;
}

double move_assign(double k) {
  V a{};
  a.m[0] = k;
  V b{};
  b = twice(a);
  return b.m[0];
}

int main() { clad::differentiate(move_assign, "k"); }

// CHECK: forward-implicit-assign-array-member.cpp:{{[0-9]+}}:3: error: cannot differentiate the defaulted move assignment of 'V': it copies 'double[2]' with '__builtin_memcpy', which has no derivative
// CHECK: note: provide custom 'operator_equal_pushforward' for 'V'
// CHECK-NOT: error:
// CHECK: 1 error generated.
