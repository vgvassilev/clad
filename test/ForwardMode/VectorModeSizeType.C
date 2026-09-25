// The independent-variable count is a sum of clad::array_ref::size() calls,
// so it is a std::size_t. The visitors that open a vector derivative stored
// it in an `unsigned long`, which is std::size_t on LP64 and 32 bits wide on
// LLP64, where a 64-bit count narrowed into a variable the user cannot edit.
// The type now comes from the ASTContext, so the spelling follows the target.
//
// On a 64-bit target std::size_t *is* `unsigned long` and the two cannot be
// told apart. A 32-bit target tells them apart: there std::size_t is
// `unsigned int` where the hardcoded spelling said `unsigned long`. Nothing is
// linked, so no 32-bit runtime is needed.

// RUN: %cladclang -m32 -fsyntax-only %s -I%S/../../include 2>&1 \
// RUN:     | %filecheck %s
//
// REQUIRES: target-i386

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

// One CHECK-DAG per visitor, then one per generated `indepVarCount`, so the
// count of each is asserted without depending on the order of the dump.
// CHECK-DAG: void caller_dvec
// CHECK-DAG: leaf_vector_pushforward
// CHECK-DAG: unsigned {{int|long long}} indepVarCount
// CHECK-DAG: unsigned {{int|long long}} indepVarCount
// CHECK-DAG: unsigned {{int|long long}} indepVarCount
// CHECK-DAG: void jac_jac

int main() {
  clad::differentiate<clad::opts::vector_mode>(caller);
  clad::jacobian(jac);
}
