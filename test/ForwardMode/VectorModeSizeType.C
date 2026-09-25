// The independent-variable count is a sum of clad::array_ref::size() calls, so
// it is a std::size_t, and the visitors that open a vector derivative used to
// declare it `unsigned long` -- a type that is std::size_t on LP64 and 32 bits
// wide on LLP64, where a 64-bit count narrowed into a variable the user cannot
// edit. The type now comes from the ASTContext, so the spelling follows the
// target: `unsigned long long` on LLP64.
//
// The Jacobian and vector-pushforward spellings are covered by the checks in
// test/Jacobian, which this change updates. This pins the vector forward mode
// one on every target, using the shape test/ForwardMode/VectorModeInterface.C
// already runs rather than a construction of its own.

// RUN: %cladclang -fsyntax-only %s -I%S/../../include 2>&1 \
// RUN:     | %filecheck %s

#include "clad/Differentiator/Differentiator.h"

double f(double x, double y) {
  return x * y;
}

double g(double x, double y, double z) {
  return x * y + y * z;
}

// CHECK: void f_dvec(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT: unsigned {{int|long|long long}} indepVarCount

// CHECK: void g_dvec(
// CHECK-NEXT: unsigned {{int|long|long long}} indepVarCount

int main() {
  clad::differentiate<clad::opts::vector_mode>(f);
  clad::differentiate<clad::opts::vector_mode>(g);
}
