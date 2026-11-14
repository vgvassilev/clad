// RUN: %cladclang %s -I%S/../../include -oimplicit-move-assign-array-member.out
// RUN: ./implicit-move-assign-array-member.out | %filecheck_exec %s
//
// Clang's implicit move assignment memcpys an array member from the address of
// an xvalue, which Sema refuses to rebuild; clad crashed on the null. The copy
// is differentiated element-wise.

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

struct V { double m[2]; };

V twice(const V& a) {
  V r{};
  r.m[0] = 2 * a.m[0];
  return r;
}

double move_assign(double k) {
  V a{};
  a.m[0] = k;
  V b{};
  b = twice(a);
  return b.m[0];
}

int main() {
  double dk = 0;
  clad::gradient(move_assign).execute(3, &dk);
  printf("%.2f\n", dk); // CHECK-EXEC: 2.00
}
