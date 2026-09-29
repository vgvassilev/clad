// RUN: %cladclang %s -I%S/../../include -oGlobalCladFunction.out \
// RUN:     | %filecheck_nodiag %s
// RUN: ./GlobalCladFunction.out | %filecheck_exec %s

// A CladFunction kept in a namespace-scope variable still calls the derivative
// clad generated. clad points the variable's function pointer at that
// derivative by rewriting the call which produced it, and does so after the
// compiler has already tried to work the initialiser out for itself. Were that
// attempt to succeed it would store the null pointer clad has not filled in
// yet, and every call would quietly return zero. CladFunction's constructor
// refuses to be a constant expression while the pointer is null, which makes
// that first attempt fail, so the value is worked out again once the rewrite
// has happened.

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

double sq(double x) { return x * x; }

auto dsq = clad::differentiate(sq, "x");

int main() {
  printf("dsq(3)=%.2f\n", dsq.execute(3.));
  //CHECK-EXEC: dsq(3)=6.00
}
