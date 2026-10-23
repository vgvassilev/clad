// A value read through a pointer that has no tangent has none either. In
// vector mode a number read that way is given a vector of zeros; a copy of an
// object, a reference to it, or a pointer to it is left without one rather
// than given a vector of zeros of its type.
// RUN: %cladclang %s -I%S/../../include -oVectorModeNoTangent.out \
// RUN:   -Xclang -verify
// RUN: ./VectorModeNoTangent.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"

#include <cstdio>

struct Inner { double a; };
struct Outer { Inner in; };
Outer outer{{2}};
Outer* outerPtr = &outer;

double readNoTangent(double x, double y) {
  // expected-warning@+1 {{'in' is accessed through an expression without a derivative; its derivative is treated as 0}}
  double v = outerPtr->in.a;
  // expected-warning@+1 {{'in' is accessed through an expression without a derivative; its derivative is treated as 0}}
  Inner copy = outerPtr->in;
  // expected-warning@+1 {{'in' is accessed through an expression without a derivative; its derivative is treated as 0}}
  const Inner& ref = outerPtr->in;
  Outer* ptr = outerPtr;
  return v * x * y;
}

int main() {
  double dx = 0, dy = 0;
  auto d = clad::differentiate<clad::opts::vector_mode>(readNoTangent, "x,y");
  d.execute(1, 2, &dx, &dy);
  printf("{%.2f, %.2f}\n", dx, dy); // CHECK-EXEC: {4.00, 2.00}
}
