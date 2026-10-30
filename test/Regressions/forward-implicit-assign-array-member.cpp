// RUN: %cladclang %s -I%S/../../include -oforward-implicit-assign-array-member.out
// RUN: ./forward-implicit-assign-array-member.out | %filecheck_exec %s
//
// Forward-mode counterpart of implicit-move-assign-array-member.cpp. Reduced
// from the Hessian of ImpactX's envelope model.

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

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

int main() {
  auto d = clad::differentiate(move_assign, "k");
  printf("%.2f\n", d.execute(3)); // CHECK-EXEC: 2.00
}
