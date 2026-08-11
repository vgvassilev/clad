// RUN: %cladclang %s -I%S/../../include -oNonActiveCallBase.out 2>&1 | %filecheck %s
// RUN: ./NonActiveCallBase.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr %s -I%S/../../include -oNonActiveCallBase.out
// RUN: ./NonActiveCallBase.out | %filecheck_exec %s

// Reverse mode over a member call whose base is a pointer baked into the
// source as an integer literal, as JIT-generated wrappers do for objects
// living in the host process. Such a base differentiates to a plain `0` -- an
// rvalue whose address cannot be taken -- which must count as an absent
// adjoint: a zero-initialized placeholder fills the `_d_this` slot of the
// pullback instead (issue #1960).

#include "clad/Differentiator/Differentiator.h"

#include <cstdio>

struct Sq {
  double operator()(double v) const { return v * v; }
};

// The method reads no object state, so the baked-in address is never
// dereferenced -- the call only routes the active argument.
double addr_lit(double x) {
  return reinterpret_cast<Sq const *>(0x40)->operator()(x);
}

// CHECK: void addr_lit_grad(double x, double *_d_x) {
// CHECK: Sq _r0 = {};
// CHECK: reinterpret_cast<const Sq *>(64)->operator_call_pullback(x, 1, &_r0, &_r1);
// CHECK: *_d_x += _r1;

int main() {
  double dx = 0;
  clad::gradient(addr_lit, "x").execute(3., &dx);
  printf("%.2f\n", dx); // CHECK-EXEC: 6.00
}
