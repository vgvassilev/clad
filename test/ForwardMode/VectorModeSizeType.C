// The independent-variable count is a sum of clad::array_ref::size() calls, so
// it is a std::size_t, and the three visitors that open a vector derivative
// used to declare it `unsigned long` -- a type that is std::size_t on LP64 and
// 32 bits wide on LLP64, where a 64-bit count narrowed into a variable the
// user cannot edit. The type now comes from the ASTContext.
//
// All this can say is that the spelling is the target's size_t, since on LP64
// the old and the new spelling are the same string. A 32-bit target would
// tell them apart -- std::size_t is `unsigned int` there -- but lit's
// `target-i386` probe reports available on macOS, where -m32 does not in fact
// produce one, so a 32-bit expectation is not something this can rely on. The
// Windows rows are what exercise the change: on LLP64 the spelling moves to
// `unsigned long long`.

// RUN: %cladclang -fsyntax-only %s -I%S/../../include 2>&1 \
// RUN:     | %filecheck %s

#include "clad/Differentiator/Differentiator.h"

double leaf(double x) {
  return x * x;
}

double caller(double x, double y) {
  return leaf(x) + leaf(y);
}

void jac(double a, double b, double _clad_out_output[]) {
  _clad_out_output[0] = a * b;
}

// One CHECK-DAG per visitor, then one per generated `indepVarCount`, so each
// site is covered without depending on the order of the dump.
// CHECK-DAG: void caller_dvec
// CHECK-DAG: leaf_vector_pushforward
// CHECK-DAG: unsigned {{int|long|long long}} indepVarCount
// CHECK-DAG: unsigned {{int|long|long long}} indepVarCount
// CHECK-DAG: unsigned {{int|long|long long}} indepVarCount
// CHECK-DAG: void jac_jac

int main() {
  clad::differentiate<clad::opts::vector_mode>(caller);
  clad::jacobian(jac);
}
